"""``search_web`` — one search, ranked hits, nothing else.

Deliberately separate from the research pipeline in
:mod:`prithivi.rag.web_research`: this tool returns *candidates* for the model to
judge, while the researcher cross-checks sources and labels claims. A tool that
both searched and synthesised would be doing the agent's job behind the
permission engine's back.
"""

from __future__ import annotations

from typing import Any

from ..core.schema import Model
from ..security.egress import EgressPolicy
from ..security.injection import sanitize
from ..web.providers import (
    SearchBlocked,
    SearchError,
    SearchNotConfigured,
    SearchProvider,
    SearchQuery,
    SearchRateLimited,
    SearchResult,
    SearchUnavailable,
    build_search_provider,
)
from .base import Evidence, SideEffect, ToolContext, ToolErrorCode, ToolResult, tool

__all__ = ["ResearchWebArgs", "SearchWebArgs", "_research_web", "_search_web", "build_search_registry", "provider_for", "researcher_for"]

_FRESHNESS = {"day", "week", "month", "year"}


class SearchWebArgs(Model):
    """Tool arguments model."""

    query: str
    """What to search for, in your own words. Be specific: 'latest python 3.14 release date'."""
    max_results: int = 5
    """How many results to return, 1-10."""
    freshness: str | None = None
    """Restrict to recent pages: 'day', 'week', 'month', or 'year'."""
    site: str | None = None
    """Restrict to one domain, e.g. 'docs.python.org'."""


def egress_policy(ctx: ToolContext) -> EgressPolicy:
    """Build the egress policy for *fetched* URLs from settings.

    Exported because ``web_fetch`` needs exactly the same policy: one helper, so
    the two web tools cannot disagree about what is reachable.
    """
    settings = ctx.settings
    return EgressPolicy(
        allow_network=bool(getattr(settings, "egress_allow_network", True)),
        allowed_hosts=tuple(getattr(settings, "egress_allowed_hosts", ()) or ()),
    )


def provider_for(ctx: ToolContext) -> SearchProvider:
    """Get the configured provider, caching it on the session context.

    Cached because the provider owns an :class:`httpx.AsyncClient`; building one
    per call would leak a connection pool per tool call.
    """
    scratch = getattr(ctx, "scratch", None)
    if scratch is not None and "search_provider" in scratch:
        cached: SearchProvider = scratch["search_provider"]
        return cached
    provider = build_search_provider(ctx.settings, egress=egress_policy(ctx))
    if scratch is not None:
        scratch["search_provider"] = provider
    return provider


def _probe_search(ctx: ToolContext) -> str | None:
    """Why search is unavailable, or ``None``."""
    if not getattr(ctx.settings, "egress_allow_network", True):
        return "network egress is disabled, so nothing can be searched"
    try:
        provider = provider_for(ctx)
    except SearchError as exc:
        return str(exc)
    if provider.requires_key and not provider.api_key.is_set:  # type: ignore[attr-defined]
        return f"PRITHIVI_SEARCH_API_KEY is not set, which the {provider.name} provider needs"
    return None


@tool(
    name="search_web",
    description=(
        "Search the web and get ranked links with short snippets. Use this for "
        "anything current or factual that you cannot answer from your own "
        "knowledge: today's news, prices, documentation, who won, when something "
        "was released. Do NOT use it for writing, calculating, opening apps, or "
        "questions about this device — and note that a snippet is a stranger's "
        "claim, so read the page with web_fetch before relying on it."
    ),
    input_model=SearchWebArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=True,
    timeout_s=20.0,
    availability_probe=_probe_search,
)
async def _search_web(
    ctx: ToolContext,
    query: str,
    max_results: int = 5,
    freshness: str | None = None,
    site: str | None = None,
) -> ToolResult:
    if not query.strip():
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, "the search query was empty",
            remedy="Search for a specific phrase, not an empty string.",
        )
    if not 1 <= int(max_results) <= 10:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, f"max_results must be between 1 and 10, got {max_results}",
            remedy="Ask for fewer results, or leave it at the default.",
        )
    if freshness is not None and freshness not in _FRESHNESS:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, f"freshness must be one of {sorted(_FRESHNESS)}, got {freshness!r}",
            remedy="Drop freshness, or use 'day', 'week', 'month', or 'year'.",
        )

    try:
        provider = provider_for(ctx)
        results: list[SearchResult] = await provider.search(
            SearchQuery(text=query.strip(), max_results=int(max_results), freshness=freshness, site=site)
        )
    except SearchBlocked as exc:
        return ToolResult.fail(
            ToolErrorCode.PERMISSION, str(exc),
            remedy="Ask the user to allow that host, or to enable network access.",
        )
    except SearchNotConfigured as exc:
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE, str(exc),
            remedy="Set PRITHIVI_SEARCH_API_KEY, or set PRITHIVI_SEARCH_PROVIDER=duckduckgo for keyless search.",
        )
    except SearchRateLimited as exc:
        return ToolResult.fail(ToolErrorCode.RATE_LIMITED, str(exc), retryable=True,
                               remedy="Wait a moment before searching again.")
    except SearchUnavailable as exc:
        return ToolResult.fail(ToolErrorCode.NETWORK, str(exc), retryable=True,
                               remedy="Check the device's internet connection.")
    except SearchError as exc:
        return ToolResult.fail(ToolErrorCode.INTERNAL, str(exc))

    if not results:
        return ToolResult.ok(
            {"result": f"No results for {query!r}.", "results": [], "provider": provider.name},
            evidence=Evidence("web_search", {"provider": provider.name, "count": 0}),
        )

    rows = [_render(r) for r in results]
    spoken = "; ".join(f"{i + 1}. {r.title or r.url}" for i, r in enumerate(results[:3]))
    return ToolResult.ok(
        {
            "result": f"Found {len(results)} result(s) for {query!r}: {spoken}",
            "results": rows,
            "provider": provider.name,
            "count": len(results),
            "freshness_supported": not getattr(provider, "freshness_unsupported", False),
        },
        evidence=Evidence("web_search", {
            "provider": provider.name,
            "count": len(results),
            "top_url": results[0].url,
        }),
        untrusted_source="web search results",
    )


def _render(result: SearchResult) -> dict[str, Any]:
    """One result as the model sees it, with the snippet sanitised.

    The snippet is a stranger's words on a page nobody has read yet, so it goes
    through the same sanitiser as a fetched page.
    """
    cleaned = sanitize(result.snippet)
    return {
        "rank": result.rank,
        "url": result.url,
        "title": cleaned.text[:200],
        "snippet": cleaned.text[:400],
        "published": result.published,
        "trust": cleaned.trust,
    }


# ---------------------------------------------------------------------------
# research_web
# ---------------------------------------------------------------------------


class ResearchWebArgs(Model):
    """Tool arguments model."""

    question: str
    """The question to answer from the live web, in the user's own words."""
    freshness: str | None = None
    """Restrict to recent pages: 'day', 'week', 'month', or 'year'."""
    site: str | None = None
    """Restrict to one domain, e.g. 'arxiv.org'."""
    max_sources: int = 4
    """How many pages to read, 1-8. More pages means a slower, wider answer."""


def researcher_for(ctx: ToolContext) -> Any:
    """Get the researcher, built without an LLM.

    No LLM here on purpose: the agent *is* the model in this turn, and giving the
    tool a second model would mean two models and two sets of instructions for one
    question. This tool returns cross-checked claims and citations; the agent
    turns them into an answer with its own voice.
    """
    # Imported here, not at module scope: rag.web_research imports core.narrator,
    # which imports tools.base, which loads this package. A module-level import
    # would close that loop and break ``import prithivi.core.narrator``.
    from ..rag.web_research import MAX_BUDGET_S, MAX_FETCHES, WebResearcher

    scratch = getattr(ctx, "scratch", None)
    if scratch is not None and "researcher" in scratch:
        cached: WebResearcher = scratch["researcher"]
        return cached
    from .web_fetch import fetcher_for

    settings = ctx.settings
    researcher = WebResearcher(
        provider_for(ctx),
        fetcher_for(ctx),
        llm=None,
        max_sources=int(getattr(settings, "research_max_sources", MAX_FETCHES)),
        budget_s=float(getattr(settings, "research_budget_s", MAX_BUDGET_S)),
        max_chars_per_page=int(getattr(settings, "web_max_chars_per_page", 4000)),
    )
    if scratch is not None:
        scratch["researcher"] = researcher
    return researcher


@tool(
    name="research_web",
    description=(
        "Answer a question from the live web: searches, reads several pages, "
        "extracts the claims they support, and returns them with numbered sources, "
        "marking anything only one source says, and flagging sources that "
        "disagree. Use this for questions that need several pages compared — "
        "reviews, comparisons, current facts, 'what are people saying about X'. "
        "Do NOT use it for a single URL (use web_fetch), for a quick lookup (use "
        "search_web), or for anything about this device."
    ),
    input_model=ResearchWebArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=True,
    timeout_s=45.0,
    availability_probe=_probe_search,
)
async def _research_web(
    ctx: ToolContext,
    question: str,
    freshness: str | None = None,
    site: str | None = None,
    max_sources: int = 4,
) -> ToolResult:
    if not question.strip():
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, "the research question was empty",
            remedy="Ask about a specific topic, in a full sentence.",
        )
    if not 1 <= int(max_sources) <= 8:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, f"max_sources must be between 1 and 8, got {max_sources}",
            remedy="Use a smaller number, or leave it at the default.",
        )

    from ..rag.web_research import ResearchRequest

    try:
        request = ResearchRequest(
            question=question.strip(),
            max_sources=int(max_sources),
            freshness=freshness,
            site=site,
        )
        result = await researcher_for(ctx).research(request)
    except ValueError as exc:
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, str(exc))

    citations = result.citations
    data: dict[str, Any] = {
        "result": result.spoken(),
        "answer": result.answer,
        "question": result.question,
        "sources": [s.to_dict() for s in citations.sources],
        "claims": [c.to_dict() for c in citations.claims],
        "conflicts": [c.describe() for c in citations.conflicts],
        "source_count": result.source_count,
        "verified_claims": len(citations.verified_claims()),
        "unverified_claims": len(citations.unverified_claims()),
        "degraded": result.degraded,
        "notes": list(result.notes),
    }

    if not citations.sources:
        # An empty result set is not a success with nothing in it: the user was
        # told a search would happen.
        return ToolResult.fail(
            ToolErrorCode.NETWORK, result.answer,
            retryable=True,
            remedy="Check the connection, or say the answer is not available.",
            data=data,
        )

    return ToolResult.ok(
        data,
        evidence=Evidence("web_search", {
            "sources": result.source_count,
            "verified_claims": len(citations.verified_claims()),
            "conflicts": len(citations.conflicts),
            "duration_ms": result.duration_ms,
        }),
        untrusted_source="web research pages",
    )


def build_search_registry() -> list:
    """Return the search tool specs."""
    return [_search_web, _research_web]
