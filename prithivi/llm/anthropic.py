"""Anthropic Messages API adapter.

Native implementation of ``POST /v1/messages`` rather than a second route
through :mod:`openai_compat`. Anthropic does serve an OpenAI-compatible
endpoint, but it is a different dialect with different tool-call semantics, so
routing it through the chat-completions client would mean carrying workarounds
in the wrong module. This is the adapter ARCHITECTURE.md S4 asks for.

The wire format differs from OpenAI in four ways that all matter here:

1. ``system`` is a **top-level request field**, not a message with ``role:
   "system"``. Passing it as a message is a hard 400.
2. Tool declarations are ``{"name", "description", "input_schema"}`` — not
   wrapped in ``{"type": "function", "function": {...}}``.
3. Assistant tool calls come back as ``tool_use`` *content blocks*, and tool
   results go back as ``tool_result`` blocks on a **user** message. There is no
   ``tool`` role. Several results for one assistant turn must share a single
   user message, which is why :func:`_to_anthropic_messages` batches them.
4. ``max_tokens`` is **required**; omitting it is a 400.

``tool_choice`` stays ``auto`` unconditionally. The model's preference for a
tool is a routing hint, never an authorisation (ARCHITECTURE.md S8): the
permission engine and confirmation broker decide what actually runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from ..security.egress import EgressError, EgressPolicy
from ..security.secrets import Secret, redactor
from .provider import (
    Completion,
    DeltaCallback,
    LLMError,
    LLMProvider,
    LLMTimeout,
    LLMUnavailable,
    Message,
    ToolCall,
    ToolSchema,
)

__all__ = ["ANTHROPIC_VERSION", "DEFAULT_BASE_URL", "AnthropicProvider"]

#: Pinned API version. Anthropic requires the header and rejects unknown values,
#: so it is a constant rather than a setting — an unpinned client breaks when
#: the vendor adds a field.
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicProvider(LLMProvider):
    """Client for the native Anthropic Messages API.

    Args:
        base_url: defaults to the public API; override for a proxy.
        api_key: an ``sk-ant-…`` secret, masked until the header is built.
        model: default model id, overridable per call.
        egress: applied to the endpoint before any request, same as every other
            outbound path (ARCHITECTURE.md S12).
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: Secret,
        model: str = "claude-sonnet-4-5",
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        egress: EgressPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.egress = egress
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(15.0, timeout_s)),
            transport=transport,
            follow_redirects=False,
        )
        self.last_usage: dict[str, int] = {}

    # -- plumbing ------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        # Anthropic authenticates with x-api-key, NOT Authorization: Bearer.
        # Sending the bearer form returns 401 and is easy to misdiagnose as a
        # bad key, so the two header sets stay deliberately separate.
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.api_key.is_set:
            headers["x-api-key"] = self.api_key.reveal()
        return headers

    def _guard(self, url: str) -> None:
        from .openai_compat import urlparse_host

        if self.egress is not None and urlparse_host(url) not in self.egress.always_allowed:
            self.egress.validate(url)

    # -- the API -------------------------------------------------------
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
        if not messages:
            raise LLMError("complete() requires at least one message")

        system, wire = _to_anthropic_messages(messages)
        if not wire:
            raise LLMError("no user or assistant message to send after system extraction")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": wire,
            "max_tokens": max_tokens or self.max_tokens,  # required by the API
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_tool_decl(t) for t in tools]
            payload["tool_choice"] = {"type": "auto"}

        url = self._endpoint("/v1/messages")
        self._guard(url)

        try:
            response = await self._client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"LLM request timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"cannot reach {redactor.scrub(self.base_url)}: {exc}") from exc

        if response.status_code >= 400:
            raise _map_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM returned non-JSON body: {response.text[:200]}") from exc

        completion = self._parse(data)
        # The Messages API is spoken one whole turn at a time here; the text is
        # simply handed to the sink whole rather than incrementally. That keeps a
        # streaming caller (voice) correct on Anthropic without a second, subtle
        # SSE parser to maintain.
        if on_delta is not None and completion.text:
            await on_delta(completion.text)
        return completion

    def _parse(self, data: dict[str, Any]) -> Completion:
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise LLMError(
                f"LLM returned no content blocks: {redactor.scrub(json.dumps(data)[:300])}"
            )

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                texts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                raw_input = block.get("input")
                calls.append(ToolCall.from_api(
                    id=str(block.get("id") or f"call_{len(calls)}"),
                    name=str(block.get("name") or ""),
                    # ``input`` arrives as an object, never a JSON string;
                    # ToolCall normalises both forms.
                    arguments=raw_input if isinstance(raw_input, dict) else {},
                ))

        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
        }
        return Completion(
            text="\n".join(t for t in texts if t).strip(),
            tool_calls=tuple(calls),
            finish_reason=_STOP_REASONS.get(str(data.get("stop_reason") or ""), "stop"),
            provider=self.name,
            model=str(data.get("model") or self.model),
            prompt_tokens=self.last_usage["prompt_tokens"],
            completion_tokens=self.last_usage["completion_tokens"],
            raw=data,
        )

    async def available(self) -> tuple[bool, str]:
        """GET /v1/models — proves key validity and reachability in one call."""
        if not self.api_key.is_set:
            return False, "PRITHIVI_LLM_API_KEY is not set (expected an sk-ant-… key)"
        url = self._endpoint("/v1/models")
        try:
            self._guard(url)
        except EgressError as exc:
            return False, str(exc)
        try:
            response = await self._client.get(url, headers=self._headers())
        except httpx.TimeoutException:
            return False, f"timed out reaching {self.base_url}"
        except httpx.HTTPError as exc:
            return False, f"cannot reach {self.base_url}: {exc}"
        if response.status_code == 401:
            return False, "Anthropic rejected the API key (401)"
        if response.status_code >= 400:
            return False, f"{self.base_url} returned HTTP {response.status_code}"
        return True, f"Anthropic reachable ({redactor.scrub(self.base_url)})"

    async def aclose(self) -> None:
        await self._client.aclose()


#: Anthropic stop reasons are not OpenAI finish reasons. The agent only reads
#: ``finish_reason`` for narration, so collapsing to the OpenAI vocabulary keeps
#: that code provider-agnostic.
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _tool_decl(tool: ToolSchema) -> dict[str, Any]:
    """OpenAI ToolSchema -> Anthropic tool declaration."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters or {"type": "object", "properties": {}},
    }


def _to_anthropic_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Translate the provider-agnostic transcript into ``(system, messages)``.

    Two subtleties drive the shape of this function:

    * ``system`` messages are hoisted out of the list, joined, and returned
      separately. Anthropic rejects them inline with a 400.
    * Anthropic has no ``tool`` role. A result goes back as a ``user`` message
      carrying a ``tool_result`` block, and *all* results for one assistant
      turn must land in a single user message — otherwise the API sees two
      consecutive user turns and rejects the request. The agent appends one
      ``tool`` message per tool call, so batching is required whenever the
      model requests more than one tool at a time.
    """
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            wire.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        if message.role == "system":
            if message.content.strip():
                system_parts.append(message.content.strip())
            continue

        if message.role == "tool":
            content, is_error = _tool_result_payload(message.content)
            pending.append({
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": content,
                **({"is_error": True} if is_error else {}),
            })
            continue

        flush()

        if message.role == "user":
            if message.content.strip():
                wire.append({"role": "user", "content": message.content})
            continue

        # assistant: text and tool_use coexist in one content-block list.
        blocks: list[dict[str, Any]] = []
        if message.content.strip():
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.parsed,
            })
        if blocks:
            wire.append({"role": "assistant", "content": blocks})

    flush()
    return "\n\n".join(system_parts), wire


def _tool_result_payload(raw: str) -> tuple[str, bool]:
    """Render a serialised :class:`ToolResult` for a ``tool_result`` block.

    A failure is surfaced as the error message plus remedy with ``is_error`` set,
    because that is the shape Anthropic uses to tell the model its call failed.
    Feeding back the whole JSON envelope instead buries the message and teaches
    the model that failing is normal.
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return raw, False
    if not isinstance(data, dict):
        return raw, False
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        text = str(error["message"])
        if error.get("remedy"):
            text = f"{text} {error['remedy']}"
        return text, True
    return json.dumps(data.get("data", data), default=str), False


def _map_error(response: httpx.Response) -> LLMError:
    """Translate Anthropic HTTP failures into actionable, user-facing messages.

    Anthropic returns ``{"type": "error", "error": {"type": ...}}``, so the
    classification comes from the body rather than the status code alone —
    ``overloaded_error`` arrives as 529 and is worth retrying.
    """
    status = response.status_code
    kind = ""
    message = ""
    try:
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            kind = str(err.get("type") or "")
            message = str(err.get("message") or "")
    except ValueError:
        pass
    detail = redactor.scrub(message or response.text[:400])

    if status == 401:
        return LLMError("Anthropic rejected the API key (401). Check PRITHIVI_LLM_API_KEY.")
    if status == 404:
        return LLMError(f"Anthropic model or endpoint not found (404): {detail}")
    if status == 429 or kind == "rate_limit_error":
        return LLMUnavailable(f"Anthropic rate limit hit: {detail}")
    if kind == "overloaded_error" or status == 529:
        return LLMUnavailable(f"Anthropic is overloaded ({status}), try again shortly: {detail}")
    if status in {500, 502, 503, 504} or kind == "api_error":
        return LLMUnavailable(f"Anthropic service error ({status}): {detail}")
    return LLMError(f"Anthropic returned HTTP {status}: {detail}")