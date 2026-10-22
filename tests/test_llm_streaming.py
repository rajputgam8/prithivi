"""Streaming path for the OpenAI-compatible provider.

The value of streaming is the *order and timing* of the text pieces that reach
the caller, so these tests assert exactly what the sink saw — not merely that the
aggregated text came out right. A regression that buffered the whole reply and
handed it over in one lump would still produce the correct ``Completion.text``
and would still pass a value-only check; the delta list is what catches it.

No network: :class:`httpx.MockTransport` answers every call with an SSE body.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from prithivi.llm.openai_compat import OpenAICompatProvider
from prithivi.llm.provider import Message
from prithivi.security.secrets import Secret

pytestmark = pytest.mark.integration


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _chunk(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _provider(body: bytes) -> OpenAICompatProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    return OpenAICompatProvider(
        base_url="https://example.test/v1",
        api_key=Secret("test-key-not-real", name="PRITHIVI_LLM_API_KEY"),
        model="test-model",
        transport=httpx.MockTransport(handler),
    )


async def test_streaming_forwards_text_deltas_in_order() -> None:
    provider = _provider(_sse(_chunk("Hello"), _chunk(", "), _chunk("world.")))
    seen: list[str] = []

    async def on_delta(piece: str) -> None:
        seen.append(piece)

    completion = await provider.complete(
        [Message(role="user", content="hi")], on_delta=on_delta
    )

    assert seen == ["Hello", ", ", "world."]
    assert completion.text == "Hello, world."


async def test_streaming_reassembles_tool_calls_split_across_events() -> None:
    provider = _provider(
        _sse(
            _chunk(
                tool_calls=[{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "get_time", "arguments": '{"time'},
                }]
            ),
            _chunk(
                tool_calls=[{
                    "index": 0,
                    "function": {"arguments": 'zone": "local"}'},
                }],
                finish="tool_calls",
            ),
        )
    )
    seen: list[str] = []

    async def on_delta(piece: str) -> None:
        seen.append(piece)

    completion = await provider.complete(
        [Message(role="user", content="hi")], on_delta=on_delta
    )

    assert seen == []
    assert completion.text == ""
    assert completion.tool_calls[0].name == "get_time"
    assert completion.tool_calls[0].parsed == {"timezone": "local"}


async def test_streaming_skips_malformed_and_non_data_lines() -> None:
    body = (
        b": keep-alive\n\n"
        b"event: message\n\n"
        b"data: not-json\n\n"
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )
    provider = _provider(body)
    seen: list[str] = []

    async def on_delta(piece: str) -> None:
        seen.append(piece)

    completion = await provider.complete(
        [Message(role="user", content="hi")], on_delta=on_delta
    )

    assert seen == ["ok"]
    assert completion.text == "ok"


async def test_without_a_sink_the_request_is_not_streamed() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}
        )

    provider = OpenAICompatProvider(
        base_url="https://example.test/v1",
        api_key=Secret("test-key-not-real", name="PRITHIVI_LLM_API_KEY"),
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    completion = await provider.complete([Message(role="user", content="hi")])

    assert completion.text == "done"
    assert "stream" not in captured
