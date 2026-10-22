"""Response narration with provenance (ARCHITECTURE.md S3.2).

The narrator is where honesty is enforced. Its central rule:

    Never claim an action succeeded without evidence.

A tool that returned ``success=False`` cannot be described as done, however
optimistic the model's text was. If the model wrote "I opened Instagram" but the
tool returned ``BACKEND_UNAVAILABLE``, the user hears what actually happened.

Provenance answers a different question — *how does Prithivi know this?* — so the
UI can label an answer as parametric, web-retrieved, user-stated, or observed
from a tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..tools.base import ToolResult

__all__ = ["Narration", "Narrator", "Provenance", "ProvenanceKind"]


class ProvenanceKind(str, Enum):
    """Where a claim came from."""

    MODEL_PARAMETRIC = "model_parametric"
    RETRIEVED_CONTEXT = "retrieved_context"
    WEB_LIVE = "web_live"
    USER_STATED = "user_stated"
    TOOL_OBSERVATION = "tool_observation"
    LOCAL_MEMORY = "local_memory"

    @property
    def label(self) -> str:
        return {
            ProvenanceKind.MODEL_PARAMETRIC: "from my own knowledge",
            ProvenanceKind.RETRIEVED_CONTEXT: "from your saved notes",
            ProvenanceKind.WEB_LIVE: "from the web",
            ProvenanceKind.USER_STATED: "from something you told me",
            ProvenanceKind.TOOL_OBSERVATION: "just checked on your phone",
            ProvenanceKind.LOCAL_MEMORY: "from what I remember about you",
        }[self]


@dataclass(frozen=True)
class Provenance:
    """One source backing part of an answer."""

    kind: ProvenanceKind
    source_id: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    confidence: float | None = None

    def chip(self) -> str:
        """Short label for the UI."""
        text = self.kind.label
        return f"{text} ({self.confidence:.0%})" if self.confidence is not None else text


@dataclass
class Narration:
    """What to say, plus how we know it."""

    text: str
    provenance: list[Provenance] = None  # type: ignore[assignment]
    blocked: bool = False
    details: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.provenance is None:
            self.provenance = []
        if self.details is None:
            self.details = {}

    @property
    def chips(self) -> list[str]:
        return [p.chip() for p in self.provenance]


class Narrator:
    """Renders agent output into something honest to read aloud."""

    def render(
        self,
        *,
        text: str,
        tool_results: Sequence[ToolResult] = (),
        tool_names: Sequence[str] = (),
        provenance: Sequence[Provenance] = (),
        blocked: bool = False,
    ) -> Narration:
        """Produce the final user-facing message.

        Args:
            text: the model's own words.
            tool_results: what actually happened, in call order.
            tool_names: the tool names, aligned with ``tool_results``.
            provenance: sources for the claims.
            blocked: True when every action was refused by the permission engine.
        """
        failures = [(n, r) for n, r in zip(tool_names, tool_results, strict=False) if not r.success]
        confirmed = [(n, r) for n, r in zip(tool_names, tool_results, strict=False) if r.success]

        all_provenance = list(provenance)
        if confirmed and not any(p.kind == ProvenanceKind.TOOL_OBSERVATION for p in all_provenance):
            all_provenance.append(Provenance(ProvenanceKind.TOOL_OBSERVATION))

        if blocked and not confirmed:
            spoken = "I did not do that." + self._failure_detail(failures)
        elif failures and not confirmed:
            spoken = self._failure_detail(failures, leading="I could not do that.")
        elif failures and confirmed:
            spoken = self._mixed_detail(confirmed, failures)
        else:
            spoken = text.strip() or "I could not find anything to do for that."

        # A refusal must never be dressed up with a success claim.
        if blocked and not confirmed:
            spoken = _strip_success_claims(text, spoken)

        return Narration(
            text=spoken,
            provenance=all_provenance,
            blocked=blocked and not confirmed,
            details={
                "tool_calls": [
                    {"tool": n, "success": r.success, "summary": r.summary(), "result": r.to_dict()}
                    for n, r in zip(tool_names, tool_results, strict=False)
                ],
                "confirmed": [n for n, _ in confirmed],
                "failed": [n for n, _ in failures],
            },
        )

    # -- failure phrasing ---------------------------------------------
    @staticmethod
    def _failure_detail(failures: Sequence[tuple[str, ToolResult]], *, leading: str = "") -> str:
        if not failures:
            return leading
        name, result = failures[0]
        error = result.error
        assert error is not None
        others = len(failures) - 1

        prefix = leading or "I could not do that."
        message = _humanise(error.code, error.message)
        remedy = f" {error.remedy}" if error.remedy else ""
        extra = f" ({others} other action{'s' if others != 1 else ''} also failed.)" if others else ""
        return f"{prefix} {message}{remedy}{extra}"

    @staticmethod
    def _mixed_detail(
        confirmed: Sequence[tuple[str, ToolResult]],
        failures: Sequence[tuple[str, ToolResult]],
    ) -> str:
        """Say what worked and what did not; never blur them together."""
        done = [n.replace("_", " ") for n, _ in confirmed]
        not_done = [n.replace("_", " ") for n, _ in failures]
        parts = [f"I did: {', '.join(done)}."]
        if len(not_done) == 1:
            name, result = failures[0]
            assert result.error is not None
            parts.append(f" But {name.replace('_', ' ')} failed: {_humanise(result.error.code, result.error.message)}")
        else:
            parts.append(f" But these failed: {', '.join(not_done)}.")
        return " ".join(parts)


# Error codes a user should never see verbatim.
_CODE_PHRASES: dict[str, str] = {
    "BACKEND_UNAVAILABLE": "that capability is not available on this device right now.",
    "CONFIRMATION_REQUIRED": "it needs your confirmation first.",
    "DENIED": "it was blocked by your permission settings.",
    "TIMEOUT": "it took too long and I stopped waiting.",
    "NOT_FOUND": "I could not find it.",
    "PERMISSION": "the system would not let me.",
    "INVALID_INPUT": "I could not work out what you meant.",
    "VALIDATION_ERROR": "I filled in the details incorrectly.",
    "UNKNOWN_TOOL": "that action is not one I can perform.",
    "RATE_LIMITED": "the service is rate-limiting me; try again shortly.",
    "NETWORK": "the network call failed.",
    "INTERNAL_ERROR": "something went wrong on my side.",
}


def _humanise(code: str, message: str) -> str:
    """Prefer the friendly phrase, fall back to the tool's own message."""
    return _CODE_PHRASES.get(code, message)


_SUCCESS_CLAIMS = (
    "i have opened", "i opened", "i've opened", "i have sent", "i sent",
    "i have deleted", "i deleted", "i have done", "i've done", "done.",
    "successfully", "i have started", "i started",
)


def _strip_success_claims(model_text: str, fallback: str) -> str:
    """Discard model text that asserts success we have no evidence for."""
    lowered = model_text.lower()
    if not any(claim in lowered for claim in _SUCCESS_CLAIMS):
        return fallback
    return fallback
