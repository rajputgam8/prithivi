"""Short-term memory: the rolling conversation buffer.

Long conversations exceed the model's context window. Short-term memory keeps
the recent turns verbatim and folds everything older into a running summary,
so the transcript stays bounded without losing the thread.

The summary is produced by :mod:`prithivi.memory.summarizer`, which has a
deterministic extractive implementation and an optional model-backed one. The
deterministic path is the default because it needs no network and cannot
hallucinate: it selects the sentences that carry the most content words.

Nothing here writes to the database. The agent owns the transcript; this
module owns the policy for how much of it to keep.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ShortTermMemory", "Turn", "estimate_tokens"]


def estimate_tokens(text: str) -> int:
    """Rough token count for a string.

    Four characters per token is the usual English approximation. It is used
    for budgeting, never for billing, so a coarse estimate is fine — and it
    keeps the tokenizer dependency-free.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class Turn:
    """One exchange in the buffer."""

    role: str
    content: str
    at: str = ""
    tokens: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "at": self.at, "tokens": self.tokens}


class ShortTermMemory:
    """A bounded, summarising message buffer.

    Args:
        max_turns: how many recent turns to keep verbatim.
        max_tokens: soft token budget for the verbatim window. When exceeded,
            the oldest turns are summarised away.
        summarizer: the :class:`~prithivi.memory.summarizer.Summarizer` used
            when the window overflows.
    """

    def __init__(
        self,
        *,
        max_turns: int = 20,
        max_tokens: int = 4000,
        summarizer: Any = None,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.max_tokens = max(1, max_tokens)
        self._turns: list[Turn] = []
        self._summary: str = ""
        self._summary_points: list[str] = []
        self._summarizer = summarizer

    # -- writes --------------------------------------------------------
    def add(self, role: str, content: str, *, at: str = "") -> Turn:
        """Append a turn and compact if the budget is now exceeded."""
        turn = Turn(role=role, content=content, at=at)
        self._turns.append(turn)
        self._compact()
        return turn

    def extend(self, turns: Sequence[Turn]) -> None:
        for turn in turns:
            self._turns.append(turn)
        self._compact()

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""
        self._summary_points = []

    # -- reads ---------------------------------------------------------
    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def summary_points(self) -> list[str]:
        return list(self._summary_points)

    @property
    def token_count(self) -> int:
        return sum(t.tokens for t in self._turns)

    def messages(self) -> list[dict[str, Any]]:
        """The buffer as model messages: summary first (as a system note), then turns."""
        out: list[dict[str, Any]] = []
        if self._summary:
            out.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self._summary}",
                }
            )
        out.extend({"role": t.role, "content": t.content} for t in self._turns)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self._turns],
            "summary": self._summary,
            "summary_points": self._summary_points,
            "tokens": self.token_count,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
        }

    # -- internals -----------------------------------------------------
    def _compact(self) -> None:
        """Drop and summarise old turns until the window fits.

        The most recent turns are never summarised: the last exchange is what
        the model needs most, and a summary of "what did you just ask?" is
        useless.
        """
        while len(self._turns) > self.max_turns or (
            self.token_count > self.max_tokens and len(self._turns) > 1
        ):
            oldest = self._turns.pop(0)
            self._absorb(oldest)

    def _absorb(self, turn: Turn) -> None:
        if self._summarizer is not None:
            result = self._summarizer.summarize(turn.content, existing=self._summary)
            self._summary = result.summary
            self._summary_points = result.key_points
            return
        # No summarizer: keep a crude running digest so the context still
        # carries the gist rather than losing the turn entirely.
        snippet = " ".join(turn.content.split())[:200]
        self._summary = (self._summary + "\n" + snippet).strip()
        self._summary_points = self._summary_points[-9:] + [snippet]
