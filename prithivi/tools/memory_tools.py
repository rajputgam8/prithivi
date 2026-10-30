"""Tools for the user's long-term memory.

Memory is an explicit contract: the user says "remember X", Prithivi stores a
structured, inspectable row with provenance, and the next session can recall it.
Nothing here writes a memory the user did not ask for, and every write is
reversible (a memory is soft-deleted and its vector removed).

The tools receive the memory service through :class:`ToolContext`, never by
importing the agent. When memory is disabled the tools are not registered at all,
and a direct call degrades to a typed failure instead of silently doing nothing.
"""

from __future__ import annotations

import re

from ..core.schema import Model
from .base import Evidence, SideEffect, ToolContext, ToolErrorCode, ToolResult, tool

__all__ = ["ForgetMemoryArgs", "RecallMemoryArgs", "RememberMemoryArgs", "build_memory_registry"]

_PROFILE_RE = re.compile(
    r"\b(?:my name is|call me|i am|i'm|my birthday is|i live in|i work as)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+(?:prefer|like|love|hate|always|never|use|want)|prefer|always|never)\b",
    re.IGNORECASE,
)
_VALUE_AFTER_RE = re.compile(
    r"\b(?:prefer|like|love|hate|always|never|use|want)\s+(.+)$",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b(?:my name is|call me)\s+(.+)$", re.IGNORECASE)


def _slug(text: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (cleaned[:limit].rstrip("_") or "memory")


def _classify(text: str) -> tuple[str, str, str]:
    """Return ``(type, key, value)`` for an explicit memory statement."""
    if _PROFILE_RE.search(text):
        name = _NAME_RE.search(text)
        value = name.group(1).strip(" .") if name else text.strip()
        key = "profile.name" if name else f"profile.{_slug(value, 32)}"
        return "profile", key, value

    if _PREFERENCE_RE.search(text):
        match = _VALUE_AFTER_RE.search(text)
        value = match.group(1).strip(" .") if match else text.strip()
        return "preference", f"preference.{_slug(value, 40)}", value

    return "fact", f"note.{_slug(text, 40)}", text.strip()


class RememberMemoryArgs(Model):
    """Tool arguments model."""

    text: str
    """The fact, preference, or profile detail to remember, in the user's words."""
    key: str | None = None
    """Optional stable key; derived from the text when omitted."""
    type: str = "auto"
    """One of fact, preference, profile, procedure, event, insight, or auto."""
    pinned: bool | None = None
    """Keep this memory at the top of recall (defaults to true for preferences)."""


@tool(
    name="remember_memory",
    description=(
        "Store something the user explicitly asked you to remember — a preference, "
        "a profile detail, or a durable fact. Use ONLY when the user says 'remember', "
        "'note that', 'don't forget', or clearly states a lasting preference. Do NOT "
        "use it for secrets, passwords, one-off task details, or anything the user did "
        "not ask to keep."
    ),
    input_model=RememberMemoryArgs,
    permission_level=1,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=True,
    timeout_s=10.0,
)
async def _remember_memory(
    ctx: ToolContext,
    text: str,
    key: str | None = None,
    type: str = "auto",
    pinned: bool | None = None,
) -> ToolResult:
    store = getattr(ctx, "memory", None)
    if store is None or not getattr(store, "enabled", False):
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "long-term memory is disabled",
            remedy="Set PRITHIVI_MEMORY_ENABLED=true and restart Prithivi.",
        )

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "there was nothing to remember")
    if len(cleaned) > 4000:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT,
            "that is too long for a single memory; store it as a document instead",
            remedy="Use the knowledge base for long text.",
        )

    detected_type, derived_key, value = _classify(cleaned)
    memory_type = detected_type if type in {"", "auto", None} else str(type)
    memory_key = key or derived_key
    if pinned is None:
        pinned = memory_type in {"preference", "profile"}

    try:
        record = store.remember(
            key=memory_key,
            text=cleaned,
            type=memory_type,
            value=value,
            source="user_stated",
            confidence=1.0,
            pinned=bool(pinned),
        )
    except (ValueError, RuntimeError) as exc:
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, str(exc))

    return ToolResult.ok(
        {
            "result": f"Remembered: {cleaned}",
            "key": record.key,
            "type": record.type,
            "text": record.text,
            "id": record.id,
        },
        evidence=Evidence("memory_row", {"id": record.id, "key": record.key, "type": record.type}),
    )


class RecallMemoryArgs(Model):
    """Tool arguments model."""

    query: str
    """What to search for in the user's memories."""
    limit: int = 5
    """Maximum number of memories to return (1-8)."""


@tool(
    name="recall_memory",
    description=(
        "Search the user's saved memories for something relevant. Use it when the "
        "answer depends on a preference or fact the user told you earlier. Do NOT use "
        "it for general world knowledge or for documents in the knowledge base."
    ),
    input_model=RecallMemoryArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=True,
    timeout_s=10.0,
)
async def _recall_memory(ctx: ToolContext, query: str, limit: int = 5) -> ToolResult:
    store = getattr(ctx, "memory", None)
    if store is None or not getattr(store, "enabled", False):
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "long-term memory is disabled",
            remedy="Set PRITHIVI_MEMORY_ENABLED=true and restart Prithivi.",
        )
    if not query.strip():
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "the memory query was empty")
    limit = max(1, min(int(limit), 8))
    try:
        found = store.recall(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a search failure is a message, not a crash
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"memory recall failed: {exc}")

    items = [item.to_dict() for item in found]
    if not items:
        return ToolResult.ok(
            {"result": "I do not have a memory about that.", "memories": []},
            evidence=Evidence("memory_search", {"query": query, "count": 0}),
        )
    summary = "\n".join(f"- {item.get('text', '')}" for item in items)
    return ToolResult.ok(
        {
            "result": summary or f"Found {len(items)} memory item(s).",
            "memories": items,
        },
        evidence=Evidence("memory_search", {"query": query, "count": len(items)}),
    )


class ForgetMemoryArgs(Model):
    """Tool arguments model."""

    key: str
    """The memory key, or a phrase describing the memory to forget."""


@tool(
    name="forget_memory",
    description=(
        "Delete one of the user's saved memories. Use it only when the user says to "
        "forget or remove something they previously asked you to remember. Do NOT "
        "use it to delete documents from the knowledge base."
    ),
    input_model=ForgetMemoryArgs,
    permission_level=1,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=True,
    timeout_s=10.0,
)
async def _forget_memory(ctx: ToolContext, key: str) -> ToolResult:
    store = getattr(ctx, "memory", None)
    if store is None or not getattr(store, "enabled", False):
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "long-term memory is disabled",
            remedy="Set PRITHIVI_MEMORY_ENABLED=true and restart Prithivi.",
        )
    target = (key or "").strip()
    if not target:
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "no memory key was given")

    record = store.get(target) or store.get(f"profile.{target}")
    if record is None:
        matches = store.recall(target, limit=1)
        record = matches[0].record if matches else None
    if record is None:
        return ToolResult.fail(
            ToolErrorCode.NOT_FOUND,
            f"I could not find a memory matching {target!r}",
            remedy="Use the memory list to see the exact key.",
        )

    if not store.forget(record.key):
        return ToolResult.fail(ToolErrorCode.NOT_FOUND, f"{record.key!r} was already gone")
    return ToolResult.ok(
        {"result": f"Forgot: {record.text}", "key": record.key},
        evidence=Evidence("memory_deleted", {"key": record.key, "id": record.id}),
    )


def build_memory_registry() -> list:
    """Return the memory tool specs."""
    return [_remember_memory, _recall_memory, _forget_memory]
