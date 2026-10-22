"""LLM provider abstraction.

The agent talks to this protocol, never to a vendor SDK. One protocol means
OpenAI, Anthropic, Groq, Together, Ollama, LM Studio, and vLLM are all
interchangeable behind a single interface, and tests can substitute a scripted
provider with no network.

Per ARCHITECTURE.md S4: no vendor SDK dependency — plain :mod:`httpx` against
HTTP APIs. On this device (3.7 GB RAM, ~750 MB free) a local model is not
viable, so the cloud adapter is the primary path and ``local`` is offered for
Ollama/LM Studio when the user runs them elsewhere.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


#: A streaming sink. Called with each incremental piece of assistant text as the
#: model produces it. ``await``ed inline, so a slow consumer (text-to-speech)
#: naturally applies backpressure to the HTTP read loop rather than buffering the
#: whole reply in memory.
DeltaCallback = Callable[[str], Awaitable[None]]

__all__ = [
    "Completion",
    "DeltaCallback",
    "LLMError",
    "LLMProvider",
    "LLMTimeout",
    "LLMUnavailable",
    "Message",
    "Role",
    "ToolCall",
    "ToolCallValidationError",
    "ToolSchema",
]

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(RuntimeError):
    """Base class for provider failures."""

    retryable: bool = False


class LLMUnavailable(LLMError):
    """The endpoint could not be reached (no network, DNS, refused, down)."""

    retryable = True


class LLMTimeout(LLMError):
    """The request exceeded its timeout."""

    retryable = True


class ToolCallValidationError(LLMError):
    """The model returned a tool call we cannot dispatch.

    Not retryable: retrying the same request usually reproduces the same bad
    call, and the agent should be told so it can correct itself.
    """


@dataclass(frozen=True)
class Message:
    """One chat message.

    ``tool_calls`` is set on assistant messages that requested tools;
    ``tool_call_id`` is set on ``tool`` messages carrying results back.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments_json}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model.

    Accepts either form an OpenAI-compatible provider can return — a JSON *string*
    or an already-parsed mapping — because constructing one of these by hand in a
    test or a scripted provider should not require the caller to serialise first.

    The original text is kept alongside the parsed mapping so a confirmation
    prompt can show exactly what the model asked for, even when parsing failed.
    """

    id: str
    name: str
    arguments: str | dict[str, Any] = field(default_factory=dict)
    arguments_json: str = field(default="", init=False)
    parsed: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Normalise ``arguments`` into ``arguments_json`` + ``parsed``.

        Kept as an explicit two-field normalisation rather than a computed
        property because a ``frozen`` dataclass property would recompute the JSON
        on every access, and this sits on the hot path for every tool call.
        """
        raw = self.arguments
        if isinstance(raw, dict):
            parsed = dict(raw)
            text = _dumps(parsed)
        else:
            text = raw or "{}"
            try:
                decoded = json.loads(text)
            except ValueError:
                # Preserve the malformed text for the validator to report; never
                # silently degrade to {}.
                decoded = {}
            parsed = decoded if isinstance(decoded, dict) else {}
        object.__setattr__(self, "arguments_json", text)
        object.__setattr__(self, "parsed", parsed)

    @property
    def arguments_mapping(self) -> dict[str, Any]:
        """The parsed arguments. Read-only alias for :attr:`parsed`."""
        return self.parsed

    @classmethod
    def from_api(cls, *, id: str, name: str, arguments: str | dict[str, Any]) -> ToolCall:
        return cls(id=id, name=name, arguments=arguments)


@dataclass(frozen=True)
class ToolSchema:
    """An OpenAI-style function schema generated from a tool's input model.

    ``description`` is written *for the model* and should state both when to use
    the tool and when not to (ARCHITECTIVE.md S9).
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


@dataclass(frozen=True)
class Completion:
    """One model response: either tool calls, or text, or both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.tool_calls


class LLMProvider(abc.ABC):
    """What the agent requires from a model."""

    name: str = "abstract"

    @abc.abstractmethod
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
        """Return a :class:`Completion`.

        When ``on_delta`` is provided, the provider *may* stream the assistant
        text through it as it arrives, so a caller can begin acting — speaking,
        rendering — before the reply is finished. The returned
        :class:`Completion` is always the complete, aggregated result, so callers
        that ignore streaming are unaffected. A provider without a streaming
        transport may call ``on_delta`` once with the whole text, or not at all.

        Raises:
            LLMError: subclasses for transport/format problems.
        """

    async def available(self) -> tuple[bool, str]:
        """Cheap readiness probe used at startup.

        Returns:
            ``(ok, reason)`` — ``reason`` is a user-facing explanation, not a
            stack trace, because it is printed verbatim.
        """
        return True, "ok"

    async def aclose(self) -> None:
        return None
