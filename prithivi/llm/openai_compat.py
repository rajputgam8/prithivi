"""OpenAI-compatible chat-completions adapter.

One implementation covers every provider that speaks the
``POST {base_url}/chat/completions`` dialect: OpenAI itself, Groq, Together,
Fireworks, DeepSeek, Perplexity, and — importantly for a device with no room for
a local model — Ollama, LM Studio, vLLM, and llama.cpp's server.

Transport is plain :mod:`httpx`. There is no vendor SDK to upgrade or break, and
no import-time cost on a phone.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import replace
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

__all__ = ["OpenAICompatProvider", "build_provider_from_settings"]


class OpenAICompatProvider(LLMProvider):
    """Chat-completions client.

    Args:
        base_url: e.g. ``https://api.openai.com/v1`` or ``http://localhost:11434/v1``.
        api_key: masked; only revealed into the header.
        model: default model id.
        egress: applied to the endpoint before any request. Providers are
            allowlisted by the caller via ``always_allowed``.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Secret,
        model: str,
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
        headers = {"Content-Type": "application/json"}
        if self.api_key.is_set:
            headers["Authorization"] = f"Bearer {self.api_key.reveal()}"
        return headers

    def _guard(self, url: str) -> None:
        """Apply the egress policy unless the operator pinned this endpoint."""
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

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_api() for t in tools]
            payload["tool_choice"] = "auto"
            # Most providers reject temperature != 1 with some reasoning models.
            payload.pop("temperature", None)

        url = self._endpoint("/chat/completions")
        self._guard(url)

        if on_delta is not None:
            return await self._stream_complete(url, payload, on_delta)

        try:
            response = await self._client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"LLM request timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"cannot reach {redactor.scrub(self.base_url)}: {exc}") from exc

        if response.status_code >= 400:
            raise _map_http_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM returned non-JSON body: {response.text[:200]}") from exc

        return self._parse(data)

    async def _stream_complete(
        self,
        url: str,
        payload: dict[str, Any],
        on_delta: DeltaCallback,
    ) -> Completion:
        """Consume an SSE stream, forwarding text as it arrives.

        Text pieces are handed to ``on_delta`` the moment they arrive so the
        caller can start speaking after the first token instead of after the
        whole reply — the difference between "waited, then talked" and "talked
        almost immediately" on a voice turn. Tool-call fragments arrive split
        across events and are reassembled by ``index`` here, so the returned
        :class:`Completion` is identical to the non-streaming one.
        """
        payload = dict(payload)
        payload["stream"] = True

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason = "stop"

        try:
            async with self._client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _map_http_error(response)
                async for line in response.aiter_lines():
                    done, finish_reason = await _consume_sse_line(
                        line, on_delta, text_parts, calls, finish_reason
                    )
                    if done:
                        break
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"LLM request timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"cannot reach {redactor.scrub(self.base_url)}: {exc}") from exc

        tool_calls = tuple(
            ToolCall.from_api(
                id=calls[index]["id"] or f"call_{index}",
                name=calls[index]["name"],
                arguments=calls[index]["arguments"] or "{}",
            )
            for index in sorted(calls)
        )
        return Completion(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            provider=self.name,
            model=str(payload.get("model") or self.model),
        )

    def _parse(self, data: dict[str, Any]) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"LLM returned no choices: {redactor.scrub(json.dumps(data)[:300])}")
        choice = choices[0]
        message = choice.get("message") or {}

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            calls.append(ToolCall.from_api(
                id=str(raw.get("id") or f"call_{len(calls)}"),
                name=str(fn.get("name") or ""),
                arguments=fn.get("arguments") or "{}",
            ))

        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
        return Completion(
            text=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            provider=self.name,
            model=str(data.get("model") or self.model),
            prompt_tokens=self.last_usage["prompt_tokens"],
            completion_tokens=self.last_usage["completion_tokens"],
            raw=data,
        )

    async def available(self) -> tuple[bool, str]:
        """GET /models — the cheapest call that still proves key + reachability."""
        if not self.api_key.is_set:
            return False, "PRITHIVI_LLM_API_KEY is not set"
        url = self._endpoint("/models")
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
            return False, "LLM rejected the API key (401)"
        if response.status_code == 404:
            # Some compatible servers expose no /models; chat may still work.
            return True, f"{self.base_url} has no /models endpoint; assuming chat works"
        if response.status_code >= 400:
            return False, f"{self.base_url} returned HTTP {response.status_code}"
        return True, f"{self.base_url} reachable"

    async def aclose(self) -> None:
        await self._client.aclose()


def urlparse_host(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).hostname or "").lower()


async def _consume_sse_line(
    line: str,
    on_delta: DeltaCallback,
    text_parts: list[str],
    calls: dict[int, dict[str, str]],
    finish_reason: str,
) -> tuple[bool, str]:
    """Fold one Server-Sent-Events line into the running completion.

    Returns ``(done, finish_reason)``. ``done`` is set by the ``[DONE]`` sentinel
    or when a provider closes a choice. Unknown lines (``event:``, comment
    heartbeats, partial JSON) are skipped rather than raising: a malformed keep-
    alive must never abort a reply that is otherwise arriving fine.
    """
    if not line or not line.startswith("data:"):
        return False, finish_reason
    data = line[len("data:"):].strip()
    if not data:
        return False, finish_reason
    if data == "[DONE]":
        return True, finish_reason

    try:
        chunk = json.loads(data)
    except ValueError:
        return False, finish_reason

    choices = chunk.get("choices") or []
    if not choices:
        return False, finish_reason
    choice = choices[0]
    if choice.get("finish_reason"):
        finish_reason = str(choice["finish_reason"])

    delta = choice.get("delta") or {}
    piece = delta.get("content")
    if piece:
        text_parts.append(piece)
        await on_delta(piece)

    for raw in delta.get("tool_calls") or []:
        _accumulate_tool_delta(calls, raw)
    return False, finish_reason


def _accumulate_tool_delta(calls: dict[int, dict[str, str]], raw: object) -> None:
    """Merge one streamed tool-call fragment into its slot, keyed by ``index``.

    OpenAI-compatible providers split a single tool call across several events:
    the id and name arrive on the first, then the JSON arguments arrive in
    pieces that must be concatenated in order. Grouping by ``index`` (not by
    arrival order) is what keeps parallel tool calls from interleaving.
    """
    if not isinstance(raw, dict):
        return
    try:
        index = int(raw.get("index") or 0)
    except (TypeError, ValueError):
        index = 0
    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if raw.get("id"):
        slot["id"] = str(raw["id"])
    function = raw.get("function") or {}
    if not isinstance(function, dict):
        return
    if function.get("name"):
        slot["name"] += str(function["name"])
    if function.get("arguments"):
        slot["arguments"] += str(function["arguments"])


def _map_http_error(response: httpx.Response) -> LLMError:
    """Translate provider HTTP failures into actionable, user-facing messages."""
    status = response.status_code
    body = redactor.scrub(response.text[:400])
    if status == 401:
        return LLMError("LLM rejected the API key (401). Check PRITHIVI_LLM_API_KEY.")
    if status == 403:
        return LLMError(f"LLM refused access (403): {body}")
    if status == 429:
        return LLMUnavailable(f"LLM rate limit hit (429): {body}")
    if status in {500, 502, 503, 504}:
        return LLMUnavailable(f"LLM service error ({status}): {body}")
    if status == 404:
        return LLMError(f"LLM endpoint not found (404). Check PRITHIVI_LLM_BASE_URL and PRITHIVI_LLM_MODEL. Body: {body}")
    return LLMError(f"LLM returned HTTP {status}: {body}")


def build_egress_policy(settings: object) -> EgressPolicy:
    """Build the egress policy for *fetched* URLs, from settings.

    Deliberately excludes the LLM endpoint's own host from ``allowed_hosts``: an
    operator who allowlists ``api.openai.com`` should not thereby grant the web
    tools permission to read arbitrary pages from it.
    """
    return EgressPolicy(
        allow_network=bool(getattr(settings, "egress_allow_network", True)),
        allowed_hosts=tuple(getattr(settings, "egress_allowed_hosts", ()) or ()),
    )


def build_provider_from_settings(
    settings: object,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    egress: EgressPolicy | None = None,
) -> LLMProvider:
    """Construct the provider named by ``settings.llm_provider``.

    Kept in one place so ``main.py`` never imports a vendor module.

    The provider still receives an egress policy, but its own endpoint host is
    pinned in ``always_allowed``. Rationale: ``llm_base_url`` is operator intent,
    whereas a URL arriving inside a tool result or a web page is *not* and must
    earn its way through ``allowed_hosts``. Skipping this wiring would mean a
    configured allowlist silently fails to constrain anything.
    """
    provider = getattr(settings, "llm_provider", "openai_compatible")
    base_url = getattr(settings, "llm_base_url", "")
    model = getattr(settings, "llm_model", "")

    policy = egress if egress is not None else build_egress_policy(settings)
    llm_host = urlparse_host(base_url)
    if llm_host and llm_host not in policy.always_allowed:
        policy = replace(policy, always_allowed=(*policy.always_allowed, llm_host))

    if provider == "anthropic":
        # Imported lazily: anthropic.py borrows ``urlparse_host`` from this
        # module, so a top-level import here would be circular.
        from .anthropic import DEFAULT_BASE_URL, AnthropicProvider

        return AnthropicProvider(
            api_key=settings.llm_api_key,
            model=model or "claude-sonnet-4-5",
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=getattr(settings, "llm_temperature", 0.2),
            max_tokens=getattr(settings, "llm_max_tokens", 1024),
            timeout_s=getattr(settings, "llm_timeout_s", 60.0),
            transport=transport,
            egress=policy,
        )
    if provider in {"local", "openai_compatible", "openai"}:
        # ``local`` is the same wire format (Ollama/LM Studio/vLLM/llama.cpp).
        return OpenAICompatProvider(
            base_url=base_url or "http://localhost:11434/v1",
            api_key=settings.llm_api_key,
            model=model or "qwen2.5:7b-instruct",
            temperature=getattr(settings, "llm_temperature", 0.2),
            max_tokens=getattr(settings, "llm_max_tokens", 1024),
            timeout_s=getattr(settings, "llm_timeout_s", 60.0),
            transport=transport,
            egress=policy,
        )
    raise LLMError(
        f"unknown PRITHIVI_LLM_PROVIDER {provider!r}; expected one of: "
        "openai_compatible, anthropic, local"
    )


def default_env_key_name() -> str:
    return os.environ.get("PRITHIVI_LLM_KEY_ENV", "PRITHIVI_LLM_API_KEY")
