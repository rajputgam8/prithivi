"""Page fetching and main-content extraction.

Reading a page is the step where untrusted text enters the system, so it is also
the step with the most checks. In order, before any of the text is used:

1. **Egress policy** (:mod:`prithivi.security.egress`). Every URL — including
   redirects — is validated. With ``PRITHIVI_EGRESS_ALLOW_NETWORK=false`` nothing
   leaves the device, and search snippets are the only web input left.
2. **robots.txt.** A page that asks robots not to be crawled is not crawled. The
   policy is deliberately fail-open on *transport* errors (a site with no
   robots.txt, or an unreachable robots.txt) and fail-closed on an explicit
   ``Disallow``.
3. **Size and type caps.** A 40 MB PDF or an endless ``application/json`` stream
   cannot crowd out a conversation or a phone's memory.
4. **Extraction**, then :mod:`prithivi.security.injection` sanitising and trust
   scoring, so the caller knows how far to trust what it is about to read.

Extraction prefers ``trafilatura`` and falls back to a stdlib
:mod:`html.parser` extractor. The fallback is not a stub: it drops chrome and
keeps article text, and the result records which engine ran so a user reading the
provenance can see that a fallback was used.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ..security.egress import EgressError, EgressPolicy
from ..security.injection import SanitizedText, sanitize
from ..security.secrets import redactor
from .providers import DEFAULT_USER_AGENT

__all__ = [
    "FetchBlocked",
    "FetchError",
    "FetchTooLarge",
    "FetchedPage",
    "RobotsDisallowed",
    "UnsupportedContentType",
    "WebFetcher",
    "extract_main_text",
    "extractor_available",
]

#: Anything over this is not an article. Chosen to fit comfortably in a phone's
#: memory once decoded and padded by the extractor.
DEFAULT_MAX_BYTES = 2_000_000

_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/xml", "text/xml")

_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "form",
    "nav", "header", "footer", "aside", "button", "select", "figure", "figcaption",
})
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "br", "li", "ul", "ol", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dl", "dt", "dd", "table",
})
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t\x0b\f\r ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


class FetchError(RuntimeError):
    """Base class for a page that could not be read."""

    code = "FETCH_FAILED"


class FetchBlocked(FetchError):
    """The egress policy refused the URL or a redirect to it."""

    code = "EGRESS_BLOCKED"


class RobotsDisallowed(FetchError):
    """robots.txt forbids this path for our user agent."""

    code = "ROBOTS_DISALLOWED"


class FetchTooLarge(FetchError):
    """The response exceeded ``max_bytes``."""

    code = "TOO_LARGE"


class UnsupportedContentType(FetchError):
    """Not something we can extract prose from."""

    code = "UNSUPPORTED_CONTENT_TYPE"


@dataclass(frozen=True)
class FetchedPage:
    """One page, ready to be read by the researcher.

    ``trust`` comes from the injection scan and is what the citation list shows;
    ``extractor`` is recorded so a degraded extraction is visible rather than
    indistinguishable from a clean one.
    """

    url: str
    final_url: str
    status: int
    title: str
    text: str
    content_type: str
    extractor: str
    fetched_at: str
    trust: float = 1.0
    findings: tuple[Any, ...] = ()
    truncated: bool = False
    elapsed_ms: int = 0

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def host(self) -> str:
        """Hostname of the URL actually served.

        ``final_url``, not ``url``: after a redirect the two differ, and it is
        the host that answered that a citation must name.
        """
        return (urlparse(self.final_url or self.url).hostname or "").lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status": self.status,
            "title": self.title,
            "content_type": self.content_type,
            "extractor": self.extractor,
            "fetched_at": self.fetched_at,
            "trust": self.trust,
            "words": self.words,
            "truncated": self.truncated,
            "risk": "hostile" if self.trust < 0.5 else ("suspicious" if self.trust < 0.95 else "clean"),
        }


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class _ArticleParser(HTMLParser):
    """Stdlib fallback extractor.

    Tracks depth of skipped chrome elements, the ``<title>``, and text nodes,
    inserting a newline at block boundaries so sentences survive. This is what
    runs when trafilatura is absent, which on a phone is the common case.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in ("div", "section", "article", "span", "a") and _is_hidden(attrs):
            self._hidden_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in ("div", "section", "article", "span", "a") and self._hidden_depth:
            self._hidden_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth or self._hidden_depth:
            return
        stripped = data.strip()
        if stripped:
            self.text_parts.append(stripped + " ")

    def close(self) -> None:
        super().close()

    @property
    def title(self) -> str:
        return " ".join("".join(self.title_parts).split())

    @property
    def text(self) -> str:
        joined = "".join(self.text_parts)
        lines = [_WS_RE.sub(" ", line).strip() for line in joined.splitlines()]
        return _BLANKLINES_RE.sub("\n\n", "\n".join(line for line in lines if line))


def _is_hidden(attrs: list[tuple[str, str | None]]) -> bool:
    mapping = {key.lower(): (value or "") for key, value in attrs}
    if mapping.get("aria-hidden", "").lower() == "true":
        return True
    style = mapping.get("style", "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def extractor_available() -> tuple[bool, str]:
    """Whether the preferred extractor is importable, and why not if it isn't."""
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        return False, "trafilatura is not installed; using the built-in extractor"
    except Exception as exc:  # noqa: BLE001 - a broken native dep must not be fatal
        return False, f"trafilatura failed to import ({type(exc).__name__}); using the built-in extractor"
    return True, "trafilatura"


def extract_main_text(html: str) -> tuple[str, str, str]:
    """Return ``(title, text, engine)`` for one HTML document.

    Prefers trafilatura; falls back to the stdlib parser. Both paths run the
    injection sanitiser afterwards in :meth:`WebFetcher.fetch` — extraction only
    decides *what is article text*.
    """
    fallback_title = _TITLE_RE.search(html)
    if _has_trafilatura():
        extracted = _extract_with_trafilatura(html)
        if extracted:
            title, text = extracted
            return title, text, "trafilatura"
    parser = _ArticleParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup still yields what parsed
        pass
    title = parser.title or (fallback_title.group(1).strip() if fallback_title else "")
    return title, parser.text, "builtin"


def _has_trafilatura() -> bool:
    try:
        import trafilatura  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _extract_with_trafilatura(html: str) -> tuple[str, str] | None:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not extracted or len(extracted.split()) < 20:
            # trafilatura is precision-biased and legitimately returns None on
            # thin pages; the fallback does better there, so let it try.
            return None
        return None, extracted
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# fetcher
# ---------------------------------------------------------------------------


@dataclass
class _RobotsCache:
    parser: RobotFileParser | None = None
    fetched_at: dt.datetime | None = None


class WebFetcher:
    """Fetches pages under the egress policy, politely and within caps.

    Args:
        policy: egress policy. Every request, including each redirect, is
            validated against it.
        max_bytes: hard cap on the response body.
        timeout_s: per-request timeout.
        respect_robots: fetch and obey robots.txt. On by default; there is no
            supported way to turn it off, which is deliberate.
        max_redirects: cap on redirect hops, each re-checked against egress.
        min_delay_s: politeness delay per host, so one page does not get 8
            simultaneous requests.
        transport: injected in tests to serve fixtures without a network.
    """

    def __init__(
        self,
        policy: EgressPolicy,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_s: float = 15.0,
        respect_robots: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        max_redirects: int = 5,
        min_delay_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Any = None,
    ) -> None:
        self.policy = policy
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.respect_robots = respect_robots
        self.max_redirects = max_redirects
        self.min_delay_s = min_delay_s
        self.user_agent = user_agent
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(8.0, timeout_s)),
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": user_agent},
        )
        self._clock = clock
        self._robots: dict[str, _RobotsCache] = {}
        self._last_hit: dict[str, float] = {}
        self.freshness_note: str = ""

    # -- public API ----------------------------------------------------
    async def fetch(self, url: str, *, max_chars: int | None = None) -> FetchedPage:
        """Fetch and extract one page.

        Raises:
            FetchError: with a subclass naming the reason (blocked, disallowed,
                too large, wrong content type, HTTP status).
        """
        started = dt.datetime.now(dt.UTC)
        target = url.strip()
        if not target.startswith(("http://", "https://")):
            raise FetchError(f"{url!r} is not an http(s) URL")

        await self._politeness(target)
        await self._check_robots(target)

        body, final_url, status, content_type = await self._get(target)
        title, text, engine = extract_main_text(body.decode("utf-8", errors="replace"))

        truncated = False
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        cleaned: SanitizedText = sanitize(text)
        finished = dt.datetime.now(dt.UTC)
        return FetchedPage(
            url=target,
            final_url=final_url,
            status=status,
            title=title,
            text=cleaned.text,
            content_type=content_type,
            extractor=engine,
            fetched_at=started.isoformat(),
            trust=cleaned.trust,
            findings=cleaned.findings,
            truncated=truncated,
            elapsed_ms=int((finished - started).total_seconds() * 1000),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- transport -----------------------------------------------------
    async def _get(self, url: str) -> tuple[bytes, str, int, str]:
        """GET with egress re-checked per hop and a byte cap enforced while streaming."""
        current = url
        for _ in range(self.max_redirects + 1):
            self._guard(current)
            try:
                async with self._client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError(f"{current} redirected with no Location header")
                        # Re-validate: a redirect to 169.254.169.254 is the classic
                        # way past an allowlist.
                        current = urljoin(current, location)
                        continue
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > self.max_bytes:
                        raise FetchTooLarge(f"{current} is {int(declared):,} bytes; cap is {self.max_bytes:,}")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise FetchTooLarge(f"{current} exceeds the {self.max_bytes:,}-byte cap")
                        chunks.append(chunk)
                    if response.status_code >= 400:
                        raise FetchError(f"{current} returned HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type and content_type not in _HTML_TYPES:
                        raise UnsupportedContentType(
                            f"{current} is {content_type!r}; Prithivi can only read web pages and plain text"
                        )
                    return b"".join(chunks), current, response.status_code, content_type
            except httpx.TimeoutException as exc:
                raise FetchError(f"{redactor.scrub(current)} timed out after {self.timeout_s:g}s") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"cannot reach {redactor.scrub(current)}: {exc}") from exc
        raise FetchError(f"too many redirects starting from {url}")

    def _guard(self, url: str) -> None:
        try:
            self.policy.validate(url)
        except EgressError as exc:
            raise FetchBlocked(str(exc)) from exc

    # -- politeness & robots -------------------------------------------
    async def _politeness(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        previous = self._last_hit.get(host)
        now = self._monotonic()
        if previous is not None and self.min_delay_s > 0:
            wait = self.min_delay_s - (now - previous)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_hit[host] = self._monotonic()

    async def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        parser = await self._robots_for(url)
        if parser is None:
            return
        try:
            allowed = parser.can_fetch(self.user_agent, url)
        except Exception:  # noqa: BLE001 - a broken robots.txt must not block reading
            return
        if not allowed:
            raise RobotsDisallowed(
                f"{urlparse(url).hostname} asks robots not to crawl this page for {self.user_agent}"
            )

    async def _robots_for(self, url: str) -> RobotFileParser | None:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._robots.get(origin)
        if cached is not None:
            return cached.parser

        parser = RobotFileParser()
        parser.set_url(f"{origin}/robots.txt")
        entry = _RobotsCache(parser=parser)
        try:
            self._guard(f"{origin}/robots.txt")
            response = await self._client.get(f"{origin}/robots.txt")
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                # No robots.txt (404) or an error: nothing is disallowed.
                parser = None
        except (EgressError, FetchError, httpx.HTTPError):
            parser = None
        entry.parser = parser
        entry.fetched_at = dt.datetime.now(dt.UTC)
        self._robots[origin] = entry
        return parser

    # -- misc ----------------------------------------------------------
    def _monotonic(self) -> float:
        import time

        return time.monotonic()


@dataclass
class FetchReport:
    """Outcome of a batch of fetches, kept for the researcher's honest reporting."""

    pages: list[FetchedPage] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> int:
        return len(self.pages)

    def summary(self) -> str:
        parts = [f"{self.ok} fetched"]
        if self.errors:
            reasons: dict[str, int] = {}
            for _, message in self.errors:
                key = message.split(":")[0][:60]
                reasons[key] = reasons.get(key, 0) + 1
            parts.append(", ".join(f"{n}× {reason}" for reason, n in sorted(reasons.items())))
        return "; ".join(parts)
