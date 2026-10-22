"""Agent and session state.

``AgentState`` is the eight-state contract the UI renders (ARCHITECTURE.md S15).
Keeping it in one enum is what stops the HUD from inventing a ninth state.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["AgentState", "Modality", "SessionState", "TurnSource", "new_session_state"]


class AgentState(str, enum.Enum):
    """What Prithivi is doing right now."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SEARCHING = "searching"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    WAITING_USER = "waiting_user"
    ERROR = "error"

    @property
    def is_busy(self) -> bool:
        return self in {AgentState.THINKING, AgentState.SEARCHING, AgentState.EXECUTING, AgentState.SPEAKING}

    @property
    def blocks_input(self) -> bool:
        """True when a new user turn must not be started."""
        return self.is_busy or self is AgentState.WAITING_USER


class Modality(str, enum.Enum):
    """How a turn entered the system."""

    TEXT = "text"
    VOICE = "voice"
    GESTURE = "gesture"
    AUTOMATION = "automation"
    API = "api"


class TurnSource(str, enum.Enum):
    """Which component triggered a turn. Recorded for the audit trail.

    ``SECURITY.md`` invariant 3: a turn originating from ``GESTURE`` can never
    satisfy a permission gate on its own.
    """

    CLI = "cli"
    VOICE = "voice"
    GESTURE = "gesture"
    ROUTER = "router"
    AGENT = "agent"
    AUTOMATION = "automation"
    API = "api"
    UI = "ui"

    @property
    def is_untrusted_for_l3(self) -> bool:
        """Untrusted surfaces may not complete an L3/L4 action."""
        return self in {TurnSource.GESTURE, TurnSource.API, TurnSource.AUTOMATION}


@dataclass
class SessionState:
    """Mutable per-session state, replaced wholesale rather than edited in place.

    Immutability-by-convention: :meth:`evolve` returns a new object so a listener
    that captured an earlier state cannot observe a later one.
    """

    session_id: str
    state: AgentState = AgentState.IDLE
    modality: Modality = Modality.TEXT
    turn_count: int = 0
    active_tool: str | None = None
    pending_confirmation: str | None = None
    last_error: str | None = None
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    history: list[dict[str, Any]] = field(default_factory=list)

    def evolve(self, **changes: Any) -> SessionState:
        return replace(self, **changes)

    def set_state(self, state: AgentState) -> SessionState:
        """Transition state, clearing tool/confirmation fields that no longer apply."""
        changes: dict[str, Any] = {"state": state}
        if state is not AgentState.EXECUTING:
            changes["active_tool"] = None
        if state is not AgentState.WAITING_USER:
            changes["pending_confirmation"] = None
        if state is not AgentState.ERROR:
            changes["last_error"] = None
        return self.evolve(**changes)

    def record_turn(self, role: str, content: str) -> None:
        """Append one message to the transcript.

        Deliberately does **not** touch ``turn_count``. The agent increments that
        once per turn, and incrementing it here too counted every exchange three
        times over (once in ``evolve`` plus one per recorded message), so the HUD
        reported "3 turns" after a single question.
        """
        self.history.append({"role": role, "content": content, "at": dt.datetime.now(dt.UTC).isoformat()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "modality": self.modality.value,
            "turn_count": self.turn_count,
            "active_tool": self.active_tool,
            "pending_confirmation": self.pending_confirmation,
            "last_error": self.last_error,
            "started_at": self.started_at.isoformat(),
        }


def new_session_state(session_id: str, *, modality: Modality = Modality.TEXT) -> SessionState:
    return SessionState(session_id=session_id, modality=modality)
