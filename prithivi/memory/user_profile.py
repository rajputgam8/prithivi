"""The user profile.

A profile is not a separate store. It is a *view* over memories of type
``profile`` and ``preference`` — the durable, self-declared facts about the
user, assembled into a document the agent can read at the start of a turn.

Keeping it a view rather than a table has a single purpose: there is exactly
one source of truth. If the profile were its own table, the two would drift,
and the agent would end up contradicting itself — "you said you are in Berlin"
from the profile and "you said you are in Berlin" from a memory, with one of
them months old and neither marked as such.
"""

from __future__ import annotations

from typing import Any

from ..db.connection import Database
from ..db.repositories.memories import MEMORY_TYPES, MemoryRecord
from .long_term import MemoryStore

__all__ = ["UserProfile"]


class UserProfile:
    """Reads and edits the user's durable self-description.

    Args:
        store: the memory store the profile reads through.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # -- reads ---------------------------------------------------------
    def entries(self) -> list[MemoryRecord]:
        """Profile and preference memories, most confident first."""
        records = self._store.list_memories(limit=200)
        wanted = (MEMORY_TYPES.PROFILE, MEMORY_TYPES.PREFERENCE)
        return [r for r in records if r.type in wanted]

    def facts(self) -> dict[str, Any]:
        """Profile entries as a plain mapping, keyed by memory key."""
        return {record.key: record.value if record.value is not None else record.text for record in self.entries()}

    def get(self, key: str, default: Any = None) -> Any:
        record = self._store.get(key)
        if record is None:
            return default
        return record.value if record.value is not None else record.text

    def as_prompt(self) -> str:
        """A short block the agent can put in its system prompt.

        Empty string when nothing is known, so the caller does not have to
        special-case a missing profile.
        """
        entries = self.entries()
        if not entries:
            return ""
        lines = ["What I know about the user:"]
        for record in entries:
            value = record.value if record.value is not None else record.text
            lines.append(f"- {record.key}: {value}")
        return "\n".join(lines)

    # -- writes --------------------------------------------------------
    def set(self, key: str, value: Any, *, text: str | None = None) -> MemoryRecord:
        """Record a profile fact or preference."""
        return self._store.remember(
            key=f"profile.{key}",
            text=text if text is not None else f"{key}: {value}",
            type=MEMORY_TYPES.PROFILE,
            value=value,
            source="user",
            confidence=0.9,
        )

    def set_preference(self, key: str, value: Any, *, text: str | None = None) -> MemoryRecord:
        """Record a preference — how the user wants things done."""
        return self._store.remember(
            key=f"preference.{key}",
            text=text if text is not None else f"{key}: {value}",
            type=MEMORY_TYPES.PREFERENCE,
            value=value,
            source="user",
            confidence=0.9,
            pinned=True,
        )

    def forget(self, key: str) -> bool:
        """Remove a profile entry, with or without the ``profile.`` prefix."""
        if self._store.forget(key):
            return True
        return self._store.forget(f"profile.{key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [record.to_dict() for record in self.entries()],
            "facts": self.facts(),
        }


def build_profile(db: Database) -> UserProfile:
    """Convenience constructor used by the app wiring."""
    return UserProfile(MemoryStore(db))
