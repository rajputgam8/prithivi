"""Scripted LLM provider — determinism for tests and offline demo mode.

Replays a fixed list of :class:`Completion` objects, one per turn. This turns a
multi-step agent turn into an ordinary assertion::

    provider = ScriptedProvider([
        Completion(tool_calls=(ToolCall("id1", "open_url", '{"url":"https://x.com"}'),)),
        Completion(text="Opening it now."),
    ])

Two uses:

* **Tests** — no network, no flake, asserts the exact tool/args chain.
* **``prithivi demo``** — lets someone try Prithivi on a phone with no API key
  and see the permission prompt and audit trail working for real.

It deliberately does *not* pretend to be a language model: with no scripted
response it returns an honest "no script" completion rather than inventing one.
"""

from __future__ import annotations

from collections.abc import Sequence

from .provider import (
    Completion,
    DeltaCallback,
    LLMError,
    LLMProvider,
    Message,
    ToolSchema,
)

__all__ = ["ScriptedProvider"]


class ScriptedProvider(LLMProvider):
    """Replays prepared completions and records what it was asked.

    Attributes:
        calls: every ``(messages, tool_names)`` pair received, for assertions.
    """

    name = "scripted"

    def __init__(self, script: Sequence[Completion] = ()) -> None:
        self._script: list[Completion] = list(script)
        self._index = 0
        self.calls: list[tuple[list[Message], tuple[str, ...]]] = []

    def push(self, completion: Completion) -> None:
        self._script.append(completion)

    def reset(self) -> None:
        self._index = 0
        self.calls.clear()

    @property
    def remaining(self) -> int:
        return max(0, len(self._script) - self._index)

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._script)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> Completion:
        self.calls.append((list(messages), tuple(t.name for t in tools)))
        if self._index >= len(self._script):
            raise LLMError(
                f"ScriptedProvider exhausted after {len(self._script)} completion(s); "
                "the agent made more model calls than the script allows"
            )
        completion = self._script[self._index]
        self._index += 1
        # No transport here, so there is nothing to stream *from* — but a caller
        # that opted into streaming must still receive the text, or a voice turn
        # would go silent. Emit it in one piece.
        if on_delta is not None and completion.text:
            await on_delta(completion.text)
        return Completion(
            text=completion.text,
            tool_calls=completion.tool_calls,
            finish_reason=completion.finish_reason or "stop",
            provider=self.name,
            model=completion.model or "scripted",
            raw=completion.raw,
        )

    async def available(self) -> tuple[bool, str]:
        return True, "scripted provider (no network)"
