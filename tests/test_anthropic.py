"""Anthropic Messages API adapter tests.

Every request body is captured and asserted, because the whole value of a
native adapter is that the *wire format* is right. A mock that only checks the
return value would pass even if the outbound JSON were wrong in every field,
and the first sign of that would be a live request rejected with a 400.

No network: :class:`httpx.MockTransport` answers every call.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from prithivi.llm.anthropic import (
    ANTHROPIC_VERSION,
    DEFAULT_BASE_URL,
    AnthropicProvider,
    _to_anthropic_messages,
)
from prithivi.llm.openai_compat import build_provider_from_settings
from prithivi.llm.provider import (
    LLMError,
    LLMProvider,
    LLMTimeout,
    LLMUnavailable,
    Message,
    ToolCall,
    ToolSchema,
)
from prithivi.security.secrets import Secret

pytestmark = pytest.mark.integration


def provider(handler, **kwargs: Any) -> tuple[AnthropicProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    kwargs.setdefault("api_key", Secret("sk-ant-test-not-real", name="PRITHIVI_LLM_API_KEY"))
    kwargs.setdefault("model", "claude-sonnet-4-5")
    p = AnthropicProvider(transport=httpx.MockTransport(_handler), **kwargs)
    return p, seen


def body(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content)


def text_response(text: str = "hello", *, prompt_tokens: int = 10, completion_tokens: int = 5) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    })


# -- request shape ----------------------------------------------------
async def test_it_posts_to_the_messages_endpoint() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([Message(role="user", content="hi")])

    assert str(seen[0].url) == f"{DEFAULT_BASE_URL}/v1/messages"
    assert seen[0].method == "POST"


async def test_it_authenticates_with_x_api_key_and_a_pinned_version() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([Message(role="user", content="hi")])

    headers = seen[0].headers
    assert headers["x-api-key"] == "sk-ant-test-not-real"
    assert headers["anthropic-version"] == ANTHROPIC_VERSION
    # The OpenAI bearer form is a 401 here; sending it by mistake is the single
    # most common way to misdiagnose an Anthropic key as invalid.
    assert "authorization" not in headers


async def test_max_tokens_is_always_sent_because_the_api_requires_it() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([Message(role="user", content="hi")])
    assert body(seen[0])["max_tokens"] == 1024

    await p.complete([Message(role="user", content="hi")], max_tokens=64)
    assert body(seen[1])["max_tokens"] == 64


async def test_the_system_prompt_is_hoisted_out_of_the_message_list() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="system", content="You are Prithivi."),
        Message(role="user", content="hi"),
    ])

    payload = body(seen[0])
    assert payload["system"] == "You are Prithivi."
    # Sending a system message inline is a hard 400 from Anthropic.
    assert [m["role"] for m in payload["messages"]] == ["user"]


async def test_several_system_messages_are_joined() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="system", content="One."),
        Message(role="user", content="hi"),
        Message(role="system", content="Two."),
    ])
    assert body(seen[0])["system"] == "One.\n\nTwo."


async def test_tools_use_input_schema_not_the_openai_function_wrapper() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete(
        [Message(role="user", content="what time?")],
        tools=[ToolSchema(name="get_time", description="Current time.", parameters={
            "type": "object", "properties": {"tz": {"type": "string"}},
        })],
    )

    payload = body(seen[0])
    assert payload["tools"] == [{
        "name": "get_time",
        "description": "Current time.",
        "input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}},
    }]
    assert payload["tool_choice"] == {"type": "auto"}
    # The OpenAI nesting must not leak through.
    assert "function" not in payload["tools"][0]


async def test_a_tool_with_no_schema_still_sends_a_valid_object() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete(
        [Message(role="user", content="go")],
        tools=[ToolSchema(name="ping", description="Ping.", parameters={})],
    )
    assert body(seen[0])["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


async def test_an_assistant_tool_call_is_echoed_back_as_a_tool_use_block() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="user", content="what time?"),
        Message(role="assistant", content="", tool_calls=(
            ToolCall("toolu_1", "get_time", {}),
        )),
        Message(role="tool", name="get_time", tool_call_id="toolu_1",
                content=json.dumps({"success": True, "data": {"iso": "2026-01-01T00:00:00Z"}})),
    ])

    roles = [m["role"] for m in body(seen[0])["messages"]]
    # Anthropic has no "tool" role: the result returns on a user message.
    assert roles == ["user", "assistant", "user"]
    assert body(seen[0])["messages"][1]["content"][0] == {
        "type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {},
    }


async def test_parallel_tool_results_share_one_user_message() -> None:
    """Two results for one assistant turn must be batched, or Anthropic 400s."""
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="user", content="time and date?"),
        Message(role="assistant", content="", tool_calls=(
            ToolCall("toolu_1", "get_time", {}),
            ToolCall("toolu_2", "get_date", {}),
        )),
        Message(role="tool", tool_call_id="toolu_1", content=json.dumps({"success": True, "data": {}})),
        Message(role="tool", tool_call_id="toolu_2", content=json.dumps({"success": True, "data": {}})),
        Message(role="assistant", content="It is Tuesday."),
        Message(role="user", content="thanks"),
    ])

    messages = body(seen[0])["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["toolu_1", "toolu_2"]


async def test_a_failed_tool_result_is_marked_is_error() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="user", content="delete /"),
        Message(role="assistant", tool_calls=(ToolCall("toolu_1", "erase", {}),)),
        Message(role="tool", tool_call_id="toolu_1", content=json.dumps({
            "success": False,
            "data": {},
            "error": {"code": "DENIED", "message": "You did not confirm.", "remedy": "Ask again."},
        })),
    ])

    block = body(seen[0])["messages"][2]["content"][0]
    assert block["is_error"] is True
    # The message plus remedy, not the JSON envelope: the model needs the reason.
    assert block["content"] == "You did not confirm. Ask again."
    assert "remedy" not in block["content"]


async def test_a_successful_tool_result_drops_the_envelope() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="user", content="6*7?"),
        Message(role="assistant", tool_calls=(ToolCall("toolu_1", "calculate", {"expression": "6*7"}),)),
        Message(role="tool", tool_call_id="toolu_1", content=json.dumps({
            "success": True, "data": {"result": 42},
        })),
    ])
    block = body(seen[0])["messages"][2]["content"][0]
    assert "is_error" not in block
    assert json.loads(block["content"]) == {"result": 42}


async def test_text_and_tool_use_coexist_in_one_assistant_turn() -> None:
    p, seen = provider(lambda r: text_response())
    await p.complete([
        Message(role="user", content="hi"),
        Message(role="assistant", content="Let me check.", tool_calls=(
            ToolCall("toolu_1", "get_time", {}),
        )),
    ])
    blocks = body(seen[0])["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]


# -- response shape ---------------------------------------------------
async def test_it_parses_text_and_maps_stop_reasons() -> None:
    p, _ = provider(lambda r: text_response("hi there"))
    completion = await p.complete([Message(role="user", content="hi")])

    assert completion.text == "hi there"
    assert completion.finish_reason == "stop"
    assert completion.provider == "anthropic"
    assert completion.model == "claude-sonnet-4-5"


async def test_a_tool_use_response_reports_tool_calls() -> None:
    p, _ = provider(lambda r: httpx.Response(200, json={
        "id": "msg_2", "model": "claude-sonnet-4-5", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_9", "name": "get_time", "input": {"tz": "UTC"}}],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }))
    completion = await p.complete([Message(role="user", content="what time?")])

    assert completion.text == "Checking."
    assert completion.wants_tools
    assert completion.finish_reason == "tool_calls"
    call = completion.tool_calls[0]
    assert (call.id, call.name, call.parsed) == ("toolu_9", "get_time", {"tz": "UTC"})


async def test_it_reads_input_and_output_tokens() -> None:
    p, _ = provider(lambda r: text_response(prompt_tokens=30, completion_tokens=9))
    completion = await p.complete([Message(role="user", content="hi")])
    assert (completion.prompt_tokens, completion.completion_tokens) == (30, 9)


async def test_multiple_text_blocks_are_joined() -> None:
    p, _ = provider(lambda r: httpx.Response(200, json={
        "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        "usage": {},
    }))
    assert (await p.complete([Message(role="user", content="hi")])).text == "first\nsecond"


# -- failures ---------------------------------------------------------
async def test_a_missing_key_is_reported_without_a_request() -> None:
    p, seen = provider(lambda r: text_response(), api_key=Secret("", name="K"))
    ok, reason = await p.available()
    assert ok is False
    assert "sk-ant" in reason
    assert seen == []


async def test_available_rejects_a_bad_key() -> None:
    p, _ = provider(lambda r: httpx.Response(401, json={"error": {"type": "authentication_error"}}))
    ok, reason = await p.available()
    assert ok is False
    assert "401" in reason


async def test_available_reports_reachability() -> None:
    p, _ = provider(lambda r: httpx.Response(200, json={"data": []}))
    ok, _reason = await p.available()
    assert ok is True


async def test_a_timeout_is_retryable() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow")

    p, _ = provider(handler)
    with pytest.raises(LLMTimeout) as exc:
        await p.complete([Message(role="user", content="hi")])
    assert exc.value.retryable


async def test_a_transport_failure_is_reported_as_unavailable() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    p, _ = provider(handler)
    with pytest.raises(LLMUnavailable) as exc:
        await p.complete([Message(role="user", content="hi")])
    assert exc.value.retryable
    assert "no route" in str(exc.value)


@pytest.mark.parametrize("status,kind,expected", [
    (401, "authentication_error", LLMError),
    (400, "invalid_request_error", LLMError),
    (404, "not_found_error", LLMError),
    (429, "rate_limit_error", LLMUnavailable),
    (529, "overloaded_error", LLMUnavailable),
    (500, "api_error", LLMUnavailable),
])
async def test_http_failures_are_classified(status: int, kind: str, expected: type) -> None:
    p, _ = provider(lambda r: httpx.Response(status, json={"error": {"type": kind, "message": "nope"}}))
    with pytest.raises(expected):
        await p.complete([Message(role="user", content="hi")])


async def test_an_overloaded_error_is_retryable_even_though_it_is_a_529() -> None:
    p, _ = provider(lambda r: httpx.Response(529, json={"error": {"type": "overloaded_error"}}))
    with pytest.raises(LLMUnavailable) as exc:
        await p.complete([Message(role="user", content="hi")])
    assert exc.value.retryable


async def test_a_non_json_body_is_reported_clearly() -> None:
    p, _ = provider(lambda r: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(LLMError, match="non-JSON"):
        await p.complete([Message(role="user", content="hi")])


async def test_a_response_with_no_content_blocks_is_rejected() -> None:
    p, _ = provider(lambda r: httpx.Response(200, json={"model": "m", "usage": {}}))
    with pytest.raises(LLMError, match="content blocks"):
        await p.complete([Message(role="user", content="hi")])


async def test_the_key_never_appears_in_an_error_message() -> None:
    secret = Secret("sk-ant-supersecretvalue000111", name="PRITHIVI_LLM_API_KEY")
    p, _ = provider(
        lambda r: httpx.Response(500, json={"error": {"type": "api_error", "message": f"bad {secret.reveal()}"}}),
        api_key=secret,
    )
    with pytest.raises(LLMError) as exc:
        await p.complete([Message(role="user", content="hi")])
    assert "sk-ant-supersecretvalue000111" not in str(exc.value)


async def test_it_refuses_an_empty_transcript() -> None:
    p, _ = provider(lambda r: text_response())
    with pytest.raises(LLMError):
        await p.complete([])


async def test_a_system_only_transcript_is_refused_before_the_request() -> None:
    p, seen = provider(lambda r: text_response())
    with pytest.raises(LLMError, match="no user or assistant"):
        await p.complete([Message(role="system", content="only system")])
    assert seen == []


# -- wiring -----------------------------------------------------------
def test_the_factory_builds_the_native_adapter() -> None:
    class Settings:
        llm_provider = "anthropic"
        llm_base_url = ""
        llm_model = ""
        llm_api_key = Secret("sk-ant-x", name="K")
        llm_temperature = 0.3
        llm_max_tokens = 512
        llm_timeout_s = 30.0
        egress_allow_network = True
        egress_allowed_hosts = ()

    p = build_provider_from_settings(Settings())
    assert isinstance(p, AnthropicProvider)
    assert p.base_url == DEFAULT_BASE_URL
    assert p.max_tokens == 512
    assert p.temperature == 0.3


def test_the_factory_honours_a_proxy_base_url() -> None:
    class Settings:
        llm_provider = "anthropic"
        llm_base_url = "https://proxy.internal/anthropic"
        llm_model = "claude-opus-4-1"
        llm_api_key = Secret("sk-ant-x", name="K")
        llm_temperature = 0.2
        llm_max_tokens = 1024
        llm_timeout_s = 60.0
        egress_allow_network = True
        egress_allowed_hosts = ()

    p = build_provider_from_settings(Settings())
    assert p.base_url == "https://proxy.internal/anthropic"
    assert p.model == "claude-opus-4-1"


def test_an_unknown_provider_still_lists_anthropic_as_valid() -> None:
    class Settings:
        llm_provider = "nope"
        llm_base_url = ""
        llm_model = ""
        llm_api_key = Secret("", name="K")
        egress_allow_network = True
        egress_allowed_hosts = ()

    with pytest.raises(LLMError, match="anthropic"):
        build_provider_from_settings(Settings())


def test_the_adapter_satisfies_the_provider_protocol() -> None:
    assert issubclass(AnthropicProvider, LLMProvider)


# -- pure translation -------------------------------------------------
def test_translation_joins_system_and_drops_empty_turns() -> None:
    system, wire = _to_anthropic_messages([
        Message(role="system", content="Sys."),
        Message(role="user", content=""),
        Message(role="user", content="Real."),
    ])
    assert system == "Sys."
    assert wire == [{"role": "user", "content": "Real."}]


def test_translation_survives_a_malformed_tool_result() -> None:
    """A non-JSON observation must still reach the model, not crash the turn."""
    system, wire = _to_anthropic_messages([
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=(ToolCall("t1", "erase", {}),)),
        Message(role="tool", tool_call_id="t1", content="not json at all"),
    ])
    assert system == ""
    assert wire[-1]["content"][0]["content"] == "not json at all"
    assert "is_error" not in wire[-1]["content"][0]