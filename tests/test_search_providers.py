"""Search provider contract tests.

Every provider is exercised through a real :class:`httpx.MockTransport`, so the
outbound URL, headers, and query parameters are asserted, not just the return
value. These adapters exist to talk to five different APIs with five different
shapes; a mock that only checks the parsed output would pass while every one of
those wire formats was wrong.

No network, and no API keys: :class:`~prithivi.security.secrets.Secret` values are
obviously fake literals.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from prithivi.core.config import Settings
from prithivi.security.egress import EgressPolicy
from prithivi.security.secrets import Secret
from prithivi.web.providers import (
    BraveProvider,
    DuckDuckGoProvider,
    SearchBlocked,
    SearchError,
    SearchNotConfigured,
    SearchProvider,
    SearchQuery,
    SearchRateLimited,
    SearchResult,
    SearchUnavailable,
    SearxngProvider,
    SerperProvider,
    TavilyProvider,
    build_search_provider,
    dedupe_results,
    normalise_url,
    search_many,
)

pytestmark = pytest.mark.integration

ALLOW_ALL = EgressPolicy(allow_network=True, allowed_hosts=("*",))
KEY = Secret("test-key-not-real", name="PRITHIVI_SEARCH_API_KEY")


def build(cls: type[SearchProvider], handler: Any, **kwargs: Any) -> tuple[SearchProvider, list[httpx.Request]]:
    """A provider wired to ``handler``, plus the list of requests it made."""
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    kwargs.setdefault("egress", ALLOW_ALL)
    kwargs.setdefault("api_key", KEY)
    return cls(transport=httpx.MockTransport(_handler), **kwargs), seen


# ---------------------------------------------------------------------------
# serper
# ---------------------------------------------------------------------------


async def test_serper_parses_organic_results_and_strips_highlight_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://google.serper.dev/search"
        assert request.headers["X-API-KEY"] == "test-key-not-real"
        body = json.loads(request.content)
        assert body["q"] == "python 3.14 release date"
        assert body["num"] == 3
        assert body["tbs"] == {"qdr": "w"}
        return httpx.Response(200, json={"organic": [
            {"title": "Python <b>3.14</b> released", "link": "https://docs.python.org/3.14/whatsnew/",
             "snippet": "The release date is October 2025.", "date": "Oct 2025", "position": 1},
            {"title": "no link", "snippet": "dropped: not a URL"},
        ]})

    provider, _ = build(SerperProvider, handler)
    results = await provider.search(SearchQuery("python 3.14 release date", max_results=3, freshness="week"))

    assert [r.url for r in results] == ["https://docs.python.org/3.14/whatsnew/"]
    assert results[0].title == "Python 3.14 released"  # highlight tags removed
    assert results[0].published == "Oct 2025"
    assert results[0].provider == "serper"


async def test_serper_without_a_key_says_which_variable_is_missing() -> None:
    provider, _ = build(SerperProvider, lambda _: httpx.Response(200, json={}), api_key=Secret(""))
    with pytest.raises(SearchNotConfigured, match="PRITHIVI_SEARCH_API_KEY"):
        await provider.search(SearchQuery("anything"))


# ---------------------------------------------------------------------------
# brave
# ---------------------------------------------------------------------------


async def test_brave_sends_the_subscription_token_and_maps_freshness() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["X-Subscription-Token"] == "test-key-not-real"
        params = dict(request.url.params)
        assert params["q"] == "rust borrow checker"
        assert params["count"] == "2"
        assert params["freshness"] == "pm"
        assert params["search_lang"] == "en"
        return httpx.Response(200, json={"web": {"results": [
            {"title": "Understanding the borrow checker", "url": "https://rust-lang.org/book/ch04-02-borrowing.html",
             "description": "Ownership and borrowing in Rust.", "age": "2 months ago"},
        ]}})

    provider, _ = build(BraveProvider, handler)
    results = await provider.search(SearchQuery("rust borrow checker", max_results=2, freshness="month"))

    assert len(results) == 1
    assert results[0].url.startswith("https://rust-lang.org/")
    assert results[0].published == "2 months ago"


# ---------------------------------------------------------------------------
# tavily
# ---------------------------------------------------------------------------


async def test_tavily_sends_the_key_in_the_body_and_keeps_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_key"] == "test-key-not-real"
        assert body["query"] == "monad in functional programming"
        assert body["max_results"] == 2
        assert body["days"] == 7
        return httpx.Response(200, json={"results": [
            {"title": "Monads", "url": "https://en.wikipedia.org/wiki/Monad",
             "content": "A monad is a composition pattern.", "score": 0.91,
             "published_date": "2025-08-02"},
        ]})

    provider, _ = build(TavilyProvider, handler)
    results = await provider.search(SearchQuery("monad in functional programming", max_results=2, freshness="week"))

    assert results[0].score == pytest.approx(0.91)
    assert results[0].published == "2025-08-02"


# ---------------------------------------------------------------------------
# searxng
# ---------------------------------------------------------------------------


async def test_searxng_uses_the_operators_own_instance_and_json_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://searx.internal.example/search")
        params = dict(request.url.params)
        assert params["format"] == "json"
        assert params["categories"] == "general"
        assert params["q"] == "site:example.org uptime"
        return httpx.Response(200, json={"results": [
            {"title": "Status", "url": "https://status.example.org", "content": "All systems go.", "score": 3},
        ]})

    provider, _ = build(SearxngProvider, handler, endpoint="https://searx.internal.example/")
    results = await provider.search(SearchQuery("site:example.org uptime"))

    assert provider.effective_hosts == ("searx.internal.example",)
    assert results[0].score == 3.0


async def test_searxng_without_an_endpoint_refuses_rather_than_guessing() -> None:
    provider, _ = build(SearxngProvider, lambda _: httpx.Response(200, json={}), endpoint="")
    with pytest.raises(SearchNotConfigured, match="PRITHIVI_SEARCH_ENDPOINT"):
        await provider.search(SearchQuery("anything"))


# ---------------------------------------------------------------------------
# duckduckgo (keyless HTML)
# ---------------------------------------------------------------------------

DDG_HTML = """
<html><body>
  <div class="result results_links">
    <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Fwhatsnew">
        What's New In Python 3.14</a></h2>
    <a class="result__snippet">Python 3.14 adds a free-threaded build.</a>
  </div>
  <div class="result results_links">
    <h2><a class="result__a" href="https://example.org/plain">A plain link</a></h2>
    <a class="result__snippet">Nothing special here.</a>
  </div>
  <div class="result results_links"><h2><a class="result__a" href="/relative">relative</a></h2></div>
</body></html>
"""


async def test_duckduckgo_parses_html_and_unwraps_its_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.host == "html.duckduckgo.com"
        return httpx.Response(200, text=DDG_HTML)

    provider, _ = build(DuckDuckGoProvider, handler)
    results = await provider.search(SearchQuery("python 3.14 whats new"))

    assert [r.url for r in results] == [
        "https://docs.python.org/3/whatsnew",
        "https://example.org/plain",
    ]
    # A citation pointing at duckduckgo.com's redirector would be useless.
    assert all("duckduckgo.com/l/" not in r.url for r in results)
    assert results[0].snippet == "Python 3.14 adds a free-threaded build."
    assert results[0].rank == 1


async def test_duckduckgo_declares_that_it_cannot_filter_by_freshness() -> None:
    """Returning an all-time result set for "today's news" would be a silent lie."""
    provider, _ = build(DuckDuckGoProvider, lambda _: httpx.Response(200, text=DDG_HTML))
    await provider.search(SearchQuery("news", freshness="day"))
    assert provider.freshness_unsupported is True


async def test_a_malformed_page_yields_no_results_rather_than_an_exception() -> None:
    provider, _ = build(DuckDuckGoProvider, lambda _: httpx.Response(200, text="<html><body><div"))
    assert await provider.search(SearchQuery("anything")) == []


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,expected", [
    (401, SearchNotConfigured),
    (403, SearchNotConfigured),
    (429, SearchRateLimited),
    (500, SearchUnavailable),
    (503, SearchUnavailable),
])
async def test_http_statuses_map_to_typed_errors(status: int, expected: type[SearchError]) -> None:
    provider, _ = build(SerperProvider, lambda _: httpx.Response(status, text="nope"))
    with pytest.raises(expected):
        await provider.search(SearchQuery("anything"))


async def test_rate_limits_are_marked_retryable_and_other_failures_are_not() -> None:
    limited, _ = build(SerperProvider, lambda _: httpx.Response(429))
    dead, _ = build(SerperProvider, lambda _: httpx.Response(500))
    for provider, expect_retry in ((limited, True), (dead, True)):
        with pytest.raises(SearchError) as caught:
            await provider.search(SearchQuery("x"))
        assert caught.value.retryable is expect_retry

    misconfigured, _ = build(SerperProvider, lambda _: httpx.Response(401))
    with pytest.raises(SearchError) as caught:
        await misconfigured.search(SearchQuery("x"))
    assert caught.value.retryable is False


async def test_a_non_json_body_is_reported_as_such() -> None:
    provider, _ = build(SerperProvider, lambda _: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(SearchUnavailable, match="non-JSON"):
        await provider.search(SearchQuery("anything"))


async def test_a_transport_error_becomes_search_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    provider, _ = build(SerperProvider, handler)
    with pytest.raises(SearchUnavailable, match="cannot reach"):
        await provider.search(SearchQuery("anything"))


# ---------------------------------------------------------------------------
# egress
# ---------------------------------------------------------------------------


async def test_the_egress_policy_blocks_a_search_the_operator_did_not_allow() -> None:
    policy = EgressPolicy(allow_network=True, allowed_hosts=("api.search.brave.com",))
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"organic": []})

    provider, _ = build(SerperProvider, handler, egress=policy)
    with pytest.raises(SearchBlocked, match="not in"):
        await provider.search(SearchQuery("anything"))
    assert called is False, "the request must be refused before it is sent"


async def test_offline_mode_refuses_search_without_touching_the_network() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, text="")

    provider, _ = build(DuckDuckGoProvider, handler, egress=EgressPolicy(allow_network=False))
    with pytest.raises(SearchBlocked, match="offline"):
        await provider.search(SearchQuery("anything"))
    assert called is False


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "search_provider": "duckduckgo",
        "search_api_key": KEY,
        "search_endpoint": "",
        "egress_allow_network": True,
        "egress_allowed_hosts": ("*",),
    }
    base.update(overrides)
    return Settings.for_testing(**base)


def test_the_factory_builds_each_named_provider() -> None:
    assert build_search_provider(settings(search_provider="serper")).name == "serper"
    assert build_search_provider(settings(search_provider="brave")).name == "brave"
    assert build_search_provider(settings(search_provider="tavily")).name == "tavily"
    assert build_search_provider(settings(search_provider="duckduckgo")).name == "duckduckgo"
    searxng = build_search_provider(settings(search_provider="searxng", search_endpoint="https://searx.be"))
    assert isinstance(searxng, SearxngProvider)
    assert searxng.endpoint == "https://searx.be"


def test_an_unknown_provider_name_is_rejected_with_the_valid_ones() -> None:
    with pytest.raises(SearchNotConfigured, match="serper"):
        build_search_provider(settings(search_provider="altavista"))


def test_the_providers_own_host_is_pinned_into_the_egress_policy() -> None:
    """The provider is operator-configured; a page URL is not.

    Without this, an allowlist that names the LLM host would also grant search —
    or, more to the point, a configured allowlist would block the user's own
    search provider while still allowing every page it returns.
    """
    egress = EgressPolicy(allow_network=True, allowed_hosts=("html.duckduckgo.com",))
    provider = build_search_provider(settings(search_provider="duckduckgo"), egress=egress)
    assert provider.egress is not None
    assert "html.duckduckgo.com" in provider.egress.always_allowed
    # Page fetching is not widened by that pin.
    assert provider.egress.allowed_hosts == ("html.duckduckgo.com",)


def test_an_operators_searxng_host_is_pinned_too() -> None:
    egress = EgressPolicy(allow_network=True, allowed_hosts=())
    provider = build_search_provider(
        settings(search_provider="searxng", search_endpoint="https://search.myname.dev"),
        egress=egress,
    )
    assert provider.egress is not None
    assert "search.myname.dev" in provider.egress.always_allowed


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def test_urls_are_normalised_for_deduplication() -> None:
    assert normalise_url("https://Example.ORG/docs/?utm_source=x&id=7#top") == "https://example.org/docs?id=7"
    assert normalise_url("http://example.org/a/") == "http://example.org/a"


def test_the_same_page_from_three_urls_counts_once() -> None:
    """Citing a document three times would look like three sources agreeing."""
    results = dedupe_results([
        SearchResult(url="https://a.example/post", title="One"),
        SearchResult(url="https://a.example/post/?utm_campaign=x", title="One again"),
        SearchResult(url="https://b.example/post", title="Two"),
    ])
    assert [r.url for r in results] == ["https://a.example/post", "https://b.example/post"]


async def test_search_many_merges_and_reports_partial_failures() -> None:
    class Flaky(SearchProvider):
        name = "flaky"

        async def search(self, query: SearchQuery) -> list[SearchResult]:
            if query.text == "bad":
                raise SearchRateLimited("slow down")
            return [SearchResult(url=f"https://{query.text}.example/p", title=query.text)]

    results, errors = await search_many(Flaky(), [SearchQuery("one"), SearchQuery("bad"), SearchQuery("two")])

    assert [r.url for r in results] == ["https://one.example/p", "https://two.example/p"]
    assert len(errors) == 1 and isinstance(errors[0], SearchRateLimited)


async def test_search_many_with_no_queries_returns_nothing() -> None:
    assert await search_many(SerperProvider(transport=httpx.MockTransport(lambda _: httpx.Response(200))), []) == ([], [])
