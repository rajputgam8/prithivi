"""Intent router.

Two paths (ARCHITECTURE.md S8):

* **Fast path** — a deterministic regex table for high-frequency, unambiguous
  commands like "open instagram" or "what time is it". Sub-5 ms, and
  *immune to prompt injection*, because no model sees the input.
* **Slow path** — the LLM, for anything that needs understanding.

The fast path exists for a security reason as much as a latency one: a request
that never reaches a model cannot be manipulated by text in a web page.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["IntentDecision", "RouteOutcome", "Router"]

IntentKind = Literal[
    "open_app", "open_url", "calculate", "get_time",
    "remember_memory", "forget_memory", "chat", "unknown",
]


@dataclass(frozen=True)
class IntentDecision:
    """The router's verdict for one utterance."""

    kind: IntentKind
    tool: str | None = None
    arguments: dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0
    via_fast_path: bool = False
    reason: str = ""

    @property
    def is_tool_call(self) -> bool:
        return self.tool is not None


@dataclass(frozen=True)
class RouteOutcome:
    """A decision plus whether it needs clarification."""

    decision: IntentDecision
    needs_clarification: bool = False
    question: str = ""


# Verbs that introduce a destination: an app name or a URL.
_OPEN_VERBS = (
    r"(?:open|launch|start|run|fire up|show me|bring up"
    r"|go to|visit|browse|take me to|navigate to|open up)"
)
_OPEN_RE = re.compile(rf"^{_OPEN_VERBS}\s+(?P<target>.+?)\s*$", re.IGNORECASE)

# (compiled pattern, intent kind, tool name, argument extractor, human reason)
_RULES: tuple[tuple[re.Pattern[str], IntentKind, str | None, object, str], ...] = (
    (
        # Explicit "remember X" is a memory write, not a conversation. The model
        # never has to be involved, so the text cannot be reworded or dropped.
        re.compile(r"^(?:please\s+)?remember(?:\s+that)?\s+(?P<text>\S.*)$", re.IGNORECASE),
        "remember_memory", "remember_memory", lambda m: {"text": m.group("text").strip()},
        "explicit request to remember something",
    ),
    (
        re.compile(r"^(?:please\s+)?forget(?:\s+about|\s+that)?\s+(?P<key>\S.*)$", re.IGNORECASE),
        "forget_memory", "forget_memory", lambda m: {"key": m.group("key").strip()},
        "explicit request to forget a saved memory",
    ),
    (
        re.compile(
            r"^(?:what(?:'s| is| was| are)?\s+)?(?:the\s+)?(?:current\s+)?"
            r"(?:time|date|day|weekday)"
            # trailing filler: "what time is it", "what's the date now"
            r"(?:\s+(?:right\s+)?(?:now|is\s+it|is\s+it\s+right\s+now))?"
            r"(?:\s+and\s+(?:the\s+)?date)?[\s?!.]*$",
            re.IGNORECASE,
        ),
        "get_time", "get_time", lambda m: {"timezone": "local"},
        "asks for the current time or date",
    ),
    (
        re.compile(
            r"^(?:what\s+time\s+is\s+it\s+in\s+)(?P<tz>[\w/\-]+)(?:\s+time)?$", re.IGNORECASE
        ),
        "get_time", "get_time", lambda m: {"timezone": m.group("tz")},
        "asks for the time in a timezone",
    ),
    (
        # Explicit request first: "calculate 2+2" must not be treated as chat.
        re.compile(r"^(?:calculate|compute|solve|work\s+out)\s+(?P<expr>.+)$", re.IGNORECASE),
        "calculate", "calculate", lambda m: {"expression": _clean_expr(m.group("expr"))},
        "explicit calculation request",
    ),
    (
        # "what is 1,250 + 500" — digits and an operator, no other words.
        re.compile(r"^what(?:'s| is| was)\s+(?P<expr>[-+*/().! 0-9%^,]*\d[-+*/().! 0-9%^,]*)$", re.IGNORECASE),
        "calculate", "calculate", lambda m: {"expression": _clean_expr(m.group("expr"))},
        "arithmetic phrased as a question",
    ),
    (
        # Bare arithmetic. The caller additionally requires a digit, which stops
        # "open instagram" from being read as an expression.
        re.compile(r"^(?P<expr>[-+*/().! 0-9%^,]*\d[-+*/().! 0-9%^,]*)\s*=?\s*$"),
        "calculate", "calculate", lambda m: {"expression": _clean_expr(m.group("expr"))},
        "a bare arithmetic expression",
    ),
)

#: Words that make an "open X" request a URL rather than an app, even without a scheme.
_URL_SHAPED = re.compile(r"\.(com|net|org|io|co|in|dev|app|ai|me|gov|edu|shop)(?:/\S*)?$", re.IGNORECASE)

# Filler that voice input sprinkles around a destination.
_FILLER = re.compile(r"\b(?:please|the|app|application)\b", re.IGNORECASE)

# Politeness that voice input puts *in front of* the verb. Stripped from the
# whole utterance, because "please open whatsapp" is far more common from a
# microphone than it is from a keyboard, and without this the anchored open-verb
# pattern simply fails to match and the request falls through to the model.
_LEADING_FILLER = re.compile(
    r"^(?:please|pls|can you|could you|would you|will you|hey|ok|okay)\s+", re.IGNORECASE
)


def _strip_filler(text: str) -> str:
    """Drop politeness words: "please open the whatsapp app" -> "whatsapp"."""
    cleaned = _FILLER.sub(" ", text)
    return " ".join(cleaned.split())


def _looks_like_url(target: str) -> bool:
    lowered = target.strip().lower()
    if lowered.startswith(("http://", "https://", "www.")):
        return True
    if " " in lowered.strip():
        # "google maps" is an app, not a host.
        return False
    return bool(_URL_SHAPED.search(lowered))


def _clean_expr(expression: str) -> str:
    """Trim a spoken expression down to what the parser accepts."""
    text = expression.strip().rstrip("=?").strip()
    # Spoken thousands separators: "1,250 + 500" -> "1250 + 500".
    return re.sub(r"(?<=\d),(?=\d\d\d\b)", "", text)


class Router:
    """Turns an utterance into an :class:`IntentDecision`.

    Args:
        provider: optional LLM used only by the slow path. When ``None``, the
            router only uses the fast path and returns ``chat``/``unknown``
            decisions, which is what a fully offline or test run wants.
    """

    def __init__(
        self,
        *,
        provider: object = None,
        app_names: Sequence[str] = (),
        app_lookup: Callable[[str], str | None] | None = None,
    ) -> None:
        self._provider = provider
        self._app_names = tuple(app_names)
        self._app_lookup = app_lookup

    def route(self, text: str) -> RouteOutcome:
        """Classify ``text``.

        The app/URL decision is made from the *shape of the target* rather than
        by relying on rule order, because the two overlap: "open instagram" is an
        app and "open instagram.com" is a website. Ordering a regex table to get
        that right is what previously made one of them route wrongly.
        """
        utterance = _LEADING_FILLER.sub("", (text or "").strip()).strip()
        if not utterance:
            return RouteOutcome(
                IntentDecision("unknown", reason="empty utterance"), True, "I didn't catch that."
            )

        # Strip only sentence-final punctuation, and deliberately NOT "!": the
        # calculator does not support postfix factorial, so "solve 5!" must reach
        # it and be reported as unsupported rather than be quietly reduced to
        # "5" and answered with a confidently wrong number. (Use ``fact(5)``.)
        normalised = utterance.rstrip("?.").strip()

        # An explicit "open/go to ..." target is resolved first.
        open_match = _OPEN_RE.match(normalised)
        if open_match:
            target = _strip_filler(open_match.group("target"))
            resolved = self._resolve_open_target(target)
            if resolved is not None:
                return resolved

        for pattern, kind, tool_name, extractor, reason in _RULES:
            match = pattern.match(normalised)
            if match is None:
                continue
            # The bare-expression rule must not swallow "open instagram".
            if kind == "calculate" and not re.search(r"\d", normalised):
                continue
            try:
                arguments = extractor(match)  # type: ignore[operator]
            except (IndexError, KeyError):
                continue
            if not arguments:
                continue
            return RouteOutcome(IntentDecision(
                kind=kind, tool=tool_name, arguments=arguments,
                confidence=0.95, via_fast_path=True, reason=reason,
            ))

        return self._slow_path(utterance)

    def _resolve_open_target(self, target: str) -> RouteOutcome | None:
        """Decide whether ``target`` is a URL or an app, or neither."""
        if not target:
            return None

        # URL-shaped wins over app-shaped: a dotted host is never an app name in
        # this catalog, and guessing otherwise would send a website to an app
        # launcher.
        if _looks_like_url(target):
            return RouteOutcome(IntentDecision(
                "open_url", "open_url", {"url": _normalise_url(target)},
                confidence=0.92, via_fast_path=True,
                reason="opens a target that is shaped like a website",
            ))

        app = self._canonical_app(target)
        if app is not None:
            return RouteOutcome(IntentDecision(
                "open_app", "open_application", {"application": app},
                confidence=0.95, via_fast_path=True,
                reason="opens an app from the catalog",
            ))

        return None

    def _canonical_app(self, target: str) -> str | None:
        """Resolve a spoken target to a canonical catalog app name.

        Longest-match first: "whatsapp" must win over a hypothetical "wa"
        keyword, and a bare substring test would otherwise return "whats".
        """
        from ..tools.basic import resolve_app

        needle = " ".join(target.lower().split())
        if not needle:
            return None

        entry = resolve_app(needle)
        if entry is not None:
            return entry.name

        if self._app_lookup is not None:
            resolved = self._app_lookup(needle)
            if resolved:
                return resolved

        # Fall back to the caller's list of known app names.
        if self._app_names:
            lowered = needle.lower()
            matches = [name for name in self._app_names if name.lower() in lowered]
            if matches:
                return max(matches, key=len)
        return None

    # -- slow path -----------------------------------------------------
    def _slow_path(self, utterance: str) -> RouteOutcome:
        """Fall back to the LLM, or to a plain conversation turn."""
        if self._provider is None:
            return RouteOutcome(IntentDecision(
                "chat", confidence=0.3, reason="no LLM configured; treated as conversation",
            ))
        # The model is asked to decide tool *intent* only. Authority over whether
        # the action may run is never the model's (SECURITY.md, principle 1).
        return RouteOutcome(IntentDecision(
            "unknown", confidence=0.0, reason="deferred to the agent loop",
        ))


def _normalise_url(target: str) -> str:
    """Add a scheme when the user said "open instagram.com"."""
    if target.lower().startswith(("http://", "https://")):
        return target
    if target.lower().startswith("www."):
        return f"https://{target}"
    return f"https://{target}"
