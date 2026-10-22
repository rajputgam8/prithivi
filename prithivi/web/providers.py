"""LLM-agnostic web search.

Searching is not a model capability, so it does not belong to ``llm/`` and it must
never be confused with the LLM itself: an outage here degrades Prithivi to
"answers from my own knowledge", whereas an outage in the provider stops it
answering at all. Every provider is therefore the same three-method shape behind
:class:`SearchProvider`, and the agent treats a search failure as a degraded
answer, not a fatal one.

Implemented: ``serper`` (Google), ``brave``, ``tavily``, ``searxng`` (self-hosted,
no key), and ``duckduckgo`` (keyless HTML endpoint, the default).

Transport is plain :mod:`httpx`, as everywhere else in Prithivi. Every request
goes through :class:`~prithivi.security.egress.EgressPolicy`, and each provider's
own API host is pinned so a configured egress allowlist does not accidentally block
search while still constraining page fetches.
"""

from __future__ import annotations

import abc
import asyncio
import json
import re
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from ..security.egress import EgressError, EgressPolicy
from ..security.secrets import Secret, redactor

__all__ = [
    "PROVIDERS",
    "BraveProvider",
    "DuckDuckGoProvider",
    "SearchBlocked",
    "SearchError",
    "SearchNotConfigured",
    "SearchProvider",
    "SearchQuery",
    "SearchRateLimited",
    "SearchResult",
    "SearchUnavailable",
    "SearxngProvider",
    "SerperProvider",
    "TavilyProvider",
    "build_search_provider",
    "dedupe_results",
    "normalise_url",
]

DEFAULT_USER_AGENT = "Prithivi/0.6 (+https://github.com/prithivi)"

#: Provider name → the hosts it must be able to reach. Used to pin
#: ``always_allowed`` and to tell the user which host to allowlist.
PROVIDERS: dict[str, tuple[str, ...]] = {
    "serper": ("google.serper.dev",),
    "brave": ("api.search.brave.com",),
    "tavily": ("api.tavily.com",),
    "searxng": (),  # user-supplied host
    "duckduckgo": ("html.duckduckgo.com", "duckduckgo.com"),
}


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit. A snippet is a claim from a stranger, so it is treated as
    untrusted text everywhere downstream (see :mod:`prithivi.security.injection`)."""

    url: str
    title: str = ""
    snippet: str = ""
    rank: int = 0
    provider: str = ""
    published: str | None = None
    score: float | None = None

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "rank": self.rank,
            "provider": self.provider,
            "published": self.published,
            "score": self.score,
            "host": self.host,
        }


@dataclass(frozen=True)
class SearchQuery:
    """A search request.

    ``freshness`` uses the values the upstream APIs agree on: ``day``, ``week``,
    ``month``, ``year``. ``site`` restricts to a single domain, which is the
    cheapest way to make an answer authoritative ("check the official docs").
    """

    text: str
    max_results: int = 5
    language: str = "en"
    freshness: str | None = None
    site: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("search query text must not be empty")
        if self.max_results < 1:
            raise ValueError("max_results must be >= 1")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class SearchError(RuntimeError):
    """Base class. ``retryable`` mirrors the LLM provider's convention."""

    retryable: bool = False


class SearchUnavailable(SearchError):
    """The provider could not be reached."""

    retryable = True


class SearchRateLimited(SearchError):
    """HTTP 429, or a provider's own quota signal."""

    retryable = True


class SearchNotConfigured(SearchError):
    """The provider needs an API key or endpoint that is not set.

    Not retryable: retrying with the same empty configuration cannot help.
    """


class SearchBlocked(SearchError):
    """The egress policy refused the request (SECURITY.md T6)."""


# ---------------------------------------------------------------------------
# provider base
# ---------------------------------------------------------------------------


class SearchProvider(abc.ABC):
    """What the researcher requires from a search backend."""

    name: str = "abstract"
    requires_key: bool = False
    hosts: tuple[str, ...] = ()

    #: Set at query time when a provider cannot honour ``SearchQuery.freshness``.
    #: The researcher surfaces this rather than pretending results were filtered.
    freshness_unsupported: bool = False

    @abc.abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Return ranked hits for ``query``.

        Raises:
            SearchError: with a subclass precise enough for the agent to decide
                whether to retry, degrade, or report the gap honestly.
        """

    async def available(self) -> tuple[bool, str]:
        """Cheap readiness probe for ``doctor``.

        A keyless provider answers ``(True, "keyless")`` without spending a
        request; the doctor's job is to find missing configuration, not to burn
        the user's quota on startup.
        """
        return True, "keyless"

    async def aclose(self) -> None:
        return None


class _HttpProvider(SearchProvider):
    """Shared HTTP plumbing: one client, egress-guarded, typed error mapping."""

    def __init__(
        self,
        *,
        egress: EgressPolicy | None = None,
        api_key: Secret | None = None,
        timeout_s: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.egress = egress
        self.api_key = api_key or Secret("", name="PRITHIVI_SEARCH_API_KEY")
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(8.0, timeout_s)),
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    # -- helpers -------------------------------------------------------
    def _guard(self, url: str) -> None:
        if self.egress is None:
            return
        try:
            self.egress.validate(url)
        except EgressError as exc:
            raise SearchBlocked(str(exc)) from exc

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        self._guard(url)
        try:
            response = await self._client.request(method, url, headers=headers, params=params, json=json_body)
        except httpx.TimeoutException as exc:
            raise SearchUnavailable(f"{self.name} timed out after {self.timeout_s:g}s") from exc
        except httpx.HTTPError as exc:
            raise SearchUnavailable(f"cannot reach {self.name} ({redactor.scrub(url)}): {exc}") from exc
        self._raise_for_status(response)
        return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        body = redactor.scrub(response.text[:300])
        if status in {401, 403}:
            raise SearchNotConfigured(
                f"{self.name} rejected the API key ({status}). Check PRITHIVI_SEARCH_API_KEY."
            )
        if status == 429:
            raise SearchRateLimited(f"{self.name} rate limit hit (429): {body}")
        if status == 402:
            raise SearchNotConfigured(f"{self.name} rejected the key for billing (402): {body}")
        raise SearchUnavailable(f"{self.name} returned HTTP {status}: {body}")

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise SearchUnavailable(f"{self.name} returned a non-JSON body: {response.text[:200]}") from exc
        if not isinstance(data, dict):
            raise SearchUnavailable(f"{self.name} returned {type(data).__name__}, expected a JSON object")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def _text(value: object) -> str:
    """Coerce a provider field to text, tolerating None and markup."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())


def _clean_html_markers(value: str) -> str:
    """Strip the ``<b>``/``<em>`` highlight tags some providers wrap matches in."""
    return _TAG_RE.sub("", value)


_TAG_RE = re.compile(r"<[^>]{0,200}>")


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class SerperProvider(_HttpProvider):
    """Google results via serper.dev. Needs a key; free tier exists."""

    name = "serper"
    requires_key = True
    hosts = PROVIDERS["serper"]
    endpoint = "https://google.serper.dev/search"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if not self.api_key.is_set:
            raise SearchNotConfigured("PRITHIVI_SEARCH_API_KEY is required for the serper provider")
        body: dict[str, Any] = {"q": query.text, "num": min(20, query.max_results)}
        if query.freshness:
            body["tbs"] = {"qdr": _QDR.get(query.freshness, "d")}
        data = self._json(await self._request(
            "POST", self.endpoint, json_body=body, headers={"X-API-KEY": self.api_key.reveal()}
        ))
        results: list[SearchResult] = []
        for index, raw in enumerate(data.get("organic") or []):
            url = _text(raw.get("link"))
            if not url.startswith(("http://", "https://")):
                continue
            results.append(SearchResult(
                url=url,
                title=_clean_html_markers(_text(raw.get("title"))),
                snippet=_clean_html_markers(_text(raw.get("snippet"))),
                rank=int(raw.get("position") or index + 1),
                provider=self.name,
                published=_text(raw.get("date")) or None,
            ))
        return results[: query.max_results]


_QDR = {"day": "d", "week": "w", "month": "m", "year": "y"}


class BraveProvider(_HttpProvider):
    """Brave Search API. Independent index, no Google dependency."""

    name = "brave"
    requires_key = True
    hosts = PROVIDERS["brave"]
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if not self.api_key.is_set:
            raise SearchNotConfigured("PRITHIVI_SEARCH_API_KEY is required for the brave provider")
        params: dict[str, Any] = {
            "q": query.text,
            "count": min(20, query.max_results),
            "safesearch": "moderate",
        }
        if query.language:
            params["search_lang"] = query.language
        if query.freshness:
            params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}.get(query.freshness, "pd")
        if query.site:
            params["site"] = query.site
        data = self._json(await self._request(
            "GET", self.endpoint, params=params, headers={"X-Subscription-Token": self.api_key.reveal()}
        ))
        web = data.get("web") or {}
        results: list[SearchResult] = []
        for index, raw in enumerate(web.get("results") or []):
            url = _text(raw.get("url"))
            if not url.startswith(("http://", "https://")):
                continue
            results.append(SearchResult(
                url=url,
                title=_clean_html_markers(_text(raw.get("title"))),
                snippet=_clean_html_markers(_text(raw.get("description"))),
                rank=index + 1,
                provider=self.name,
                published=_text(raw.get("age") or raw.get("page_age")) or None,
            ))
        return results[: query.max_results]


class TavilyProvider(_HttpProvider):
    """Tavily. Returns cleaned content alongside the snippet, which makes it the
    cheapest provider when a single fetch is expensive."""

    name = "tavily"
    requires_key = True
    hosts = PROVIDERS["tavily"]
    endpoint = "https://api.tavily.com/search"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if not self.api_key.is_set:
            raise SearchNotConfigured("PRITHIVI_SEARCH_API_KEY is required for the tavily provider")
        body: dict[str, Any] = {
            "api_key": self.api_key.reveal(),
            "query": query.text,
            "max_results": min(20, query.max_results),
        }
        if query.freshness:
            body["days"] = {"day": 1, "week": 7, "month": 30, "year": 365}.get(query.freshness, 7)
        if query.site:
            body["include_domains"] = [query.site]
        data = self._json(await self._request("POST", self.endpoint, json_body=body))
        results: list[SearchResult] = []
        for index, raw in enumerate(data.get("results") or []):
            url = _text(raw.get("url"))
            if not url.startswith(("http://", "https://")):
                continue
            results.append(SearchResult(
                url=url,
                title=_clean_html_markers(_text(raw.get("title"))),
                snippet=_clean_html_markers(_text(raw.get("content"))),
                rank=index + 1,
                provider=self.name,
                published=_text(raw.get("published_date")) or None,
                score=_float_or_none(raw.get("score")),
            ))
        return results[: query.max_results]


class SearxngProvider(_HttpProvider):
    """A self-hosted SearXNG instance. No key: the URL is the credential, which
    is why ``PRITHIVI_SEARCH_ENDPOINT`` is mandatory here."""

    name = "searxng"
    hosts = PROVIDERS["searxng"]
    endpoint = ""

    def __init__(self, *, endpoint: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if endpoint:
            self.endpoint = endpoint.rstrip("/")

    @property
    def effective_hosts(self) -> tuple[str, ...]:
        """The operator's own instance, which the provider table cannot know."""
        host = (urlparse(self.endpoint).hostname or "").lower()
        return (host,) if host else ()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if not self.endpoint:
            raise SearchNotConfigured("PRITHIVI_SEARCH_ENDPOINT is required for the searxng provider")
        params: dict[str, Any] = {
            "q": query.text,
            "format": "json",
            "safesearch": 1,
            "categories": "general",
        }
        if query.language:
            params["language"] = query.language
        if query.freshness:
            params["time_range"] = {"day": "day", "week": "week", "month": "month", "year": "year"}.get(
                query.freshness, "month"
            )
        if query.site:
            params["site"] = query.site
        data = self._json(await self._request("GET", f"{self.endpoint}/search", params=params))
        results: list[SearchResult] = []
        for index, raw in enumerate(data.get("results") or []):
            url = _text(raw.get("url"))
            if not url.startswith(("http://", "https://")):
                continue
            results.append(SearchResult(
                url=url,
                title=_clean_html_markers(_text(raw.get("title"))),
                snippet=_clean_html_markers(_text(raw.get("content"))),
                rank=index + 1,
                provider=self.name,
                published=_text(raw.get("publishedDate")) or None,
                score=_float_or_none(raw.get("score")),
            ))
        return results[: query.max_results]


class DuckDuckGoProvider(_HttpProvider):
    """Keyless HTML endpoint.

    Not the API (there is none), so this parses HTML. It is the default because it
    costs nothing and needs no account, and it is the reason
    ``PRITHIVI_SEARCH_PROVIDER`` defaults to a value that works on first run.
    """

    name = "duckduckgo"
    hosts = PROVIDERS["duckduckgo"]
    endpoint = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        data: dict[str, Any] = {"q": query.text, "kl": query.language or "wt-wt"}
        response = await self._request("POST", self.endpoint, params=data)
        results = _parse_ddg(response.text, self.name)
        if query.site:
            results = [r for r in results if r.host == query.site.lower().removeprefix("www.")]
        if query.freshness:
            # The keyless endpoint has no freshness parameter. Returning an
            # all-time result set while the caller asked for this week's news
            # would be a silent lie, so the limitation is declared on the
            # provider and the researcher reports it.
            self.freshness_unsupported = True
        return results[: query.max_results]


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class _DDGParser(HTMLParser):
    """Pull results out of DuckDuckGo's no-JS results page.

    Only the two anchors Prithivi needs are tracked, which is why this exists
    instead of a BeautifulSoup dependency on a phone.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._pending: dict[str, str] | None = None
        self._buffer: list[str] = []
        self._mode: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        mapping = {key.lower(): (value or "") for key, value in attrs}
        classes = mapping.get("class", "")
        if "result__a" in classes:
            self._begin(_resolve_ddg_url(mapping.get("href", "")), "title")
        elif "result__snippet" in classes:
            self._begin(self._current_url(), "snippet")

    def _current_url(self) -> str:
        for result in reversed(self.results):
            if not result.title or not result.snippet:
                return result.url
        return ""

    def _begin(self, url: str, mode: str) -> None:
        self._flush()
        self._mode = mode
        self._buffer = []
        self._pending = {"url": url} if mode == "title" else None

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer = []
        if self._mode == "title" and self._pending is not None and text:
            self.results.append(SearchResult(
                url=self._pending["url"], title=text, rank=len(self.results) + 1, provider="duckduckgo"
            ))
        elif self._mode == "snippet" and text and self.results:
            last = self.results[-1]
            self.results[-1] = replace(last, snippet=text)
        self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode is not None:
            self._flush()

    def close(self) -> None:
        self._flush()
        super().close()


def _parse_ddg(html: str, provider: str) -> list[SearchResult]:
    parser = _DDGParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - a malformed page must not kill the search
        pass
    return [replace(r, provider=provider) for r in parser.results if r.url.startswith(("http://", "https://"))]


def _resolve_ddg_url(href: str) -> str:
    """DDG wraps outbound links in ``/l/?uddg=<encoded>``; unwrap it.

    Leaving the wrapper in place would put a redirect in every citation, so the
    real destination is recovered here.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/") or "uddg" in (parsed.query or ""):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target.startswith(("http://", "https://")):
            return target
    return href


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def normalise_url(url: str) -> str:
    """Canonical form for deduplication: lowercase host, no fragment, no trailing
    slash, tracking parameters removed.

    Two URLs that differ only by ``?utm_source=x`` are the same document, and
    citing both would inflate the appearance of corroboration — which is exactly
    the illusion this phase is supposed to prevent.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return url.strip()
    path = parsed.path.rstrip("/") or "/"
    keep = [
        (key, value)
        for key, value in parse_qs(parsed.query, keep_blank_values=True).items()
        if not key.lower().startswith(("utm_", "fbclid", "gclid", "ref_", "mc_", "igshid"))
    ]
    query = "&".join(f"{key}={value[0]}" if len(value) == 1 else key for key, value in keep)
    scheme = parsed.scheme or "https"
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Drop duplicate URLs, keeping the highest-ranked occurrence."""
    seen: dict[str, int] = {}
    kept: list[SearchResult] = []
    for result in results:
        key = normalise_url(result.url)
        if key in seen:
            continue
        seen[key] = len(kept)
        kept.append(replace(result, rank=len(kept) + 1))
    return kept


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def build_search_provider(
    settings: object,
    *,
    egress: EgressPolicy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchProvider:
    """Construct the provider named by ``PRITHIVI_SEARCH_PROVIDER``.

    The provider's own host is pinned into ``always_allowed``: it is operator
    intent, configured by the user, whereas URLs arriving inside search results
    are not and must earn their way through ``allowed_hosts``.

    Raises:
        SearchNotConfigured: unknown provider name.
    """
    name = str(getattr(settings, "search_provider", "duckduckgo") or "duckduckgo").strip().lower()
    if name not in PROVIDERS:
        raise SearchNotConfigured(
            f"unknown PRITHIVI_SEARCH_PROVIDER {name!r}; expected one of: {', '.join(sorted(PROVIDERS))}"
        )

    policy = egress
    if policy is not None:
        pinned = tuple(dict.fromkeys((*policy.always_allowed, *_required_hosts(name, settings))))
        if pinned != policy.always_allowed:
            policy = replace(policy, always_allowed=pinned)

    common: dict[str, Any] = {
        "egress": policy,
        "api_key": getattr(settings, "search_api_key", Secret("", name="PRITHIVI_SEARCH_API_KEY")),
        "timeout_s": float(getattr(settings, "search_timeout_s", 15.0) or 15.0),
        "user_agent": str(getattr(settings, "web_user_agent", DEFAULT_USER_AGENT) or DEFAULT_USER_AGENT),
        "transport": transport,
    }

    endpoint = str(getattr(settings, "search_endpoint", "") or "").strip()
    if name == "serper":
        return SerperProvider(**common)
    if name == "brave":
        return BraveProvider(**common)
    if name == "tavily":
        return TavilyProvider(**common)
    if name == "searxng":
        return SearxngProvider(endpoint=endpoint, **common)
    return DuckDuckGoProvider(**common)


def _required_hosts(name: str, settings: object) -> tuple[str, ...]:
    """Hosts that must be reachable for ``name`` to work, including a user's own
    SearXNG host, which is not knowable from the provider table."""
    hosts = PROVIDERS[name]
    if name != "searxng":
        return hosts
    endpoint = str(getattr(settings, "search_endpoint", "") or "").strip()
    host = (urlparse(endpoint).hostname or "").lower()
    return (host,) if host else hosts


async def search_many(
    provider: SearchProvider,
    queries: list[SearchQuery],
    *,
    limit: int = 12,
    concurrency: int = 3,
) -> tuple[list[SearchResult], list[SearchError]]:
    """Run several sub-queries and merge them.

    Returns:
        ``(results, errors)``. Errors are returned rather than raised because one
        dead sub-query must not lose the others; the caller reports the gap.
    """
    if not queries:
        return [], []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(query: SearchQuery) -> tuple[list[SearchResult], SearchError | None]:
        async with semaphore:
            try:
                return await provider.search(query), None
            except SearchError as exc:
                return [], exc

    outcomes = await asyncio.gather(*(run(q) for q in queries))
    merged: list[SearchResult] = []
    errors: list[SearchError] = []
    for results, error in outcomes:
        merged.extend(results)
        if error is not None:
            errors.append(error)
    return dedupe_results(merged)[:limit], errors


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of model output, tolerating prose and fences.

    Used by the query planner and the synthesiser, both of which must degrade to
    the original question rather than fail when the model is chatty.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate.strip())
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
