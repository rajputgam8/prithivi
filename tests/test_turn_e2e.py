"""End-to-end turn tests driven by a scripted provider.

TESTING.md §3: the scripted provider makes a whole turn deterministic, so one
test can assert the entire chain — tool chosen, arguments validated, permission
checked, executor invoked, narration produced, audit row written — with no
network and no real model.

These are the tests that would catch a regression in the *seam* between
components, which is exactly where unit tests cannot see.
"""

from __future__ import annotations

import pytest

from prithivi.core.agent import Agent, TurnCancelled
from prithivi.core.events import EventBus
from prithivi.core.schema import Model as SchemaModel
from prithivi.core.state import AgentState, Modality, TurnSource
from prithivi.llm.provider import Completion, ToolCall
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.confirmation import ConfirmationBroker, ConfirmationOutcome
from prithivi.security.permissions import Effect, PermissionEngine, Policy
from prithivi.tools.base import SideEffect, ToolContext, ToolResult, tool
from prithivi.tools.basic import build_registry
from prithivi.tools.registry import ToolRegistry


def scripted(*items: Completion) -> ScriptedProvider:
    return ScriptedProvider(list(items))


def broker_returning(outcome: ConfirmationOutcome) -> ConfirmationBroker:
    async def handler(request):
        return outcome

    return ConfirmationBroker(handler)


class Args(SchemaModel):
    """Tool arguments model."""

    target: str


@tool(
    name="erase_everything",
    description="Test-only irreversible action.",
    input_model=Args,
    permission_level=3,
    side_effects=SideEffect.IRREVERSIBLE,
    requires_confirmation=True,
    idempotent=False,
)
async def _erase(ctx: ToolContext, target: str) -> ToolResult:
    return ToolResult.ok({"erased": target})


@tool(
    name="catastrophe",
    description="Test-only L4 action.",
    input_model=Args,
    permission_level=4,
    side_effects=SideEffect.IRREVERSIBLE,
    idempotent=False,
)
async def _catastrophe(ctx: ToolContext, target: str) -> ToolResult:
    return ToolResult.ok({"wiped": target})


def build_agent(
    provider: ScriptedProvider,
    *,
    audit,
    settings,
    confirmations: ConfirmationBroker | None = None,
    permissions: PermissionEngine | None = None,
    extra_tools=(),
) -> Agent:
    registry = ToolRegistry(audit=audit)
    registry.register_all(build_registry())
    for spec in extra_tools:
        registry.register(spec, replace=True)
    return Agent(
        provider=provider,
        registry=registry,
        permissions=permissions or PermissionEngine(default_level=2),
        confirmations=confirmations or broker_returning(ConfirmationOutcome.decline("no")),
        audit=audit,
        settings=settings,
        bus=EventBus(),
    )


# --- a plain turn -----------------------------------------------------------


async def test_a_plain_question_needs_no_tool(audit, settings) -> None:
    agent = build_agent(
        scripted(Completion(text="A monad is a monoid in the category of endofunctors.")),
        audit=audit, settings=settings,
    )
    turn = await agent.run("what is a monad?")

    assert turn.succeeded
    assert turn.iterations == 1
    assert turn.tool_results == []
    assert "monad" in turn.text
    assert turn.state is AgentState.IDLE


async def test_the_session_records_both_sides_of_the_turn(audit, settings) -> None:
    """Regression guard: a user message must be stored exactly once.

    The CLI used to record the user turn and then hand the same utterance to the
    agent, which recorded it again. History grew as ``[user, user, assistant]``
    and the duplicate was fed back to the model on every later turn.
    """
    agent = build_agent(scripted(Completion(text="hi")), audit=audit, settings=settings)
    await agent.run("hello")

    roles = [entry["role"] for entry in agent.session.history]
    assert roles == ["user", "assistant"]
    assert [e["content"] for e in agent.session.history] == ["hello", "hi"]
    # One exchange is one turn, not three.
    assert agent.session.turn_count == 1


# --- a tool turn, end to end ------------------------------------------------


async def test_a_tool_turn_runs_the_whole_chain(audit, settings) -> None:
    """One assertion set over the whole pipeline."""
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "21*2"}),)),
            Completion(text="That is 42."),
        ),
        audit=audit, settings=settings,
    )
    turn = await agent.run("what is 21 times 2")

    assert turn.succeeded
    assert [n for n, _ in turn.tool_results] == ["calculate"]
    assert turn.tool_results[0][1].data["result"] == 42
    assert turn.iterations == 2

    # The permission engine was consulted and allowed it.
    checks = [r for r in audit.all() if r.action == "permission.check"]
    assert len(checks) == 1
    assert checks[0].detail["effect"] == Effect.ALLOW
    assert checks[0].detail["tool"] == "calculate"

    # The tool call itself was recorded, with its evidence.
    calls = [r for r in audit.all() if r.action == "tool.execute"]
    assert calls[0].outcome == "success"
    assert calls[0].detail["result"]["evidence"] is not None

    # And the whole trail is verifiable.
    assert audit.verify() == (True, None)


async def test_provenance_records_what_was_actually_observed(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "get_time", {"timezone": "utc"}),)),
            Completion(text="It is now."),
        ),
        audit=audit, settings=settings,
    )
    turn = await agent.run("what time is it")

    kinds = {p.kind for p in turn.narration.provenance}
    assert "tool_observation" in {k.value for k in kinds}


async def test_several_tools_in_one_turn_all_run(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(
                ToolCall("c1", "calculate", {"expression": "1+1"}),
                ToolCall("c2", "get_time", {"timezone": "utc"}),
            )),
            Completion(text="Both done."),
        ),
        audit=audit, settings=settings,
    )
    turn = await agent.run("add one and check the time")

    assert [n for n, _ in turn.tool_results] == ["calculate", "get_time"]
    assert turn.succeeded


# --- honesty ----------------------------------------------------------------


async def test_narration_never_claims_an_unrun_action(audit, settings) -> None:
    """Rule 2: the model may not assert success for a tool that did not run."""
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "erase_everything", {"target": "/"}),)),
            Completion(text="I have deleted everything on your device."),
        ),
        audit=audit,
        settings=settings,
        confirmations=broker_returning(ConfirmationOutcome.decline("no")),
        extra_tools=(_erase,),
    )
    turn = await agent.run("delete everything")

    assert turn.blocked is True
    assert turn.tool_results[0][1].success is False
    lowered = turn.text.lower()
    for claim in ("deleted everything", "i deleted", "done"):
        assert claim not in lowered, f"narration repeated the model's false claim: {claim!r}"


async def test_a_failed_tool_is_reported_as_failed(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "2 +"}),)),
            Completion(text="The expression was malformed."),
        ),
        audit=audit, settings=settings,
    )
    turn = await agent.run("add two to nothing")

    result = turn.tool_results[0][1]
    assert result.success is False
    assert result.error.code == "INVALID_INPUT"


async def test_an_unknown_tool_is_refused_with_alternatives(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "launch_missiles", {"target": "all"}),)),
            Completion(text="I cannot do that."),
        ),
        audit=audit, settings=settings,
    )
    turn = await agent.run("launch everything")

    result = turn.tool_results[0][1]
    assert result.error.code == "UNKNOWN_TOOL"
    assert "launch_missiles" in result.error.message
    assert turn.blocked is False, "an unknown tool is a model error, not a refusal"


# --- confirmation -----------------------------------------------------------


async def test_a_declined_confirmation_blocks_the_action(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "erase_everything", {"target": "/tmp/a"}),)),
            Completion(text="Understood."),
        ),
        audit=audit,
        settings=settings,
        confirmations=broker_returning(ConfirmationOutcome.decline("changed my mind")),
        extra_tools=(_erase,),
    )
    turn = await agent.run("delete /tmp/a")

    assert turn.tool_results[0][1].success is False
    assert turn.blocked is True
    assert [r.outcome for r in audit.all() if r.action == "confirmation"] == ["denied"]


async def test_an_approved_confirmation_lets_the_action_run(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "erase_everything", {"target": "/tmp/a"}),)),
            Completion(text="Done."),
        ),
        audit=audit,
        settings=settings,
        confirmations=broker_returning(ConfirmationOutcome.approve("ui")),
        extra_tools=(_erase,),
    )
    turn = await agent.run("delete /tmp/a")

    assert turn.tool_results[0][1].success is True
    assert turn.tool_results[0][1].data == {"erased": "/tmp/a"}


async def test_a_denied_policy_stops_a_tool_without_asking(audit, settings) -> None:
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "1+1"}),)),
            Completion(text="I can't do that."),
        ),
        audit=audit,
        settings=settings,
        permissions=PermissionEngine(default_level=2, policies=[
            Policy("calculate", effect=Effect.DENY, source="user"),
        ]),
    )
    turn = await agent.run("add one and one")

    assert turn.tool_results[0][1].success is False
    assert turn.tool_results[0][1].error.code == "DENIED"
    assert turn.blocked is True


async def test_an_l4_action_is_confirmed_with_a_typed_phrase(audit, settings) -> None:
    seen: list[str] = []

    async def handler(request):
        seen.append(request.typed_phrase)
        return ConfirmationOutcome.approve("typed", typed_phrase=request.typed_phrase)

    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "catastrophe", {"target": "/"}),)),
            Completion(text="As you confirmed, done."),
        ),
        audit=audit,
        settings=settings,
        confirmations=ConfirmationBroker(handler),
        extra_tools=(_catastrophe,),
    )
    turn = await agent.run("do the catastrophic thing")

    assert seen and len(seen[0].split()) == 2, "the user must be shown a phrase to repeat"
    assert turn.tool_results[0][1].success is True


async def test_an_l4_action_fails_when_the_phrase_is_wrong(audit, settings) -> None:
    async def handler(request):
        return ConfirmationOutcome.approve("typed", typed_phrase="something else")

    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "catastrophe", {"target": "/"}),)),
            Completion(text="It is done."),
        ),
        audit=audit,
        settings=settings,
        confirmations=ConfirmationBroker(handler),
        extra_tools=(_catastrophe,),
    )
    turn = await agent.run("do the catastrophic thing")

    assert turn.tool_results[0][1].success is False
    assert turn.blocked is True


# --- untrusted sources ------------------------------------------------------


@pytest.mark.security
async def test_a_gesture_turn_cannot_authorise_itself(audit, settings) -> None:
    """SECURITY.md invariant 3, asserted end to end.

    The model asks for a reversible tool from a gesture turn with no grant in
    place. The turn must come back asking for confirmation rather than acting.
    """
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "open_application", {"application": "chrome"}),)),
            Completion(text="Opening Chrome."),
        ),
        audit=audit,
        settings=settings,
        confirmations=broker_returning(ConfirmationOutcome.decline("no")),
    )
    turn = await agent.run(
        "open chrome", source=TurnSource.GESTURE, modality=Modality.GESTURE
    )

    checks = [r for r in audit.all() if r.action == "permission.check"]
    assert checks[0].detail["source"] == "gesture"
    assert checks[0].detail["effect"] == Effect.CONFIRM


@pytest.mark.security
async def test_a_gesture_turn_is_allowed_by_a_preexisting_grant(audit, settings) -> None:
    """The documented exception: the grant came from the user, not the gesture."""
    agent = build_agent(
        scripted(
            Completion(tool_calls=(ToolCall("c1", "open_application", {"application": "chrome"}),)),
            Completion(text="Opening Chrome."),
        ),
        audit=audit,
        settings=settings,
        permissions=PermissionEngine(default_level=2, policies=[
            Policy("open_application:chrome", effect=Effect.ALLOW, max_uses=1),
        ]),
    )
    turn = await agent.run("open chrome", source=TurnSource.GESTURE)

    checks = [r for r in audit.all() if r.action == "permission.check"]
    assert checks[0].detail["effect"] == Effect.ALLOW


# --- failure handling -------------------------------------------------------


async def test_an_llm_failure_is_an_error_turn_not_a_crash(audit, settings) -> None:
    from prithivi.llm.provider import LLMUnavailable

    class BrokenProvider(ScriptedProvider):
        async def complete(self, messages, **kwargs):
            raise LLMUnavailable("cannot reach the endpoint")

    agent = build_agent(BrokenProvider([]), audit=audit, settings=settings)
    turn = await agent.run("hello")

    assert turn.succeeded is False
    assert "cannot reach the endpoint" in turn.error
    assert turn.state is AgentState.ERROR


async def test_a_turn_can_be_cancelled(audit, settings) -> None:
    import asyncio

    cancel = asyncio.Event()
    cancel.set()

    agent = build_agent(scripted(Completion(text="hi")), audit=audit, settings=settings)
    with pytest.raises(TurnCancelled):
        await agent.run("hello", cancel=cancel)


async def test_exhausting_the_iteration_budget_is_a_failure(audit, settings) -> None:
    """Falling out of the loop is not an answer, and must not be reported as one."""
    looping = scripted(*[
        Completion(tool_calls=(ToolCall(f"c{i}", "get_time", {"timezone": "utc"}),))
        for i in range(50)
    ])
    agent = build_agent(looping, audit=audit, settings=settings)
    turn = await agent.run("loop forever")

    assert turn.iterations == settings.agent_max_iterations
    assert turn.succeeded is False
    assert "without reaching an answer" in turn.error
    assert turn.state is AgentState.ERROR


async def test_an_empty_provider_response_is_handled(audit, settings) -> None:
    """No text and no tool calls must not raise."""
    agent = build_agent(scripted(Completion()), audit=audit, settings=settings)
    turn = await agent.run("say nothing")

    assert turn.iterations == 1
    assert turn.text is not None


# --- session hygiene --------------------------------------------------------


async def test_reset_clears_the_session(audit, settings) -> None:
    agent = build_agent(
        scripted(Completion(text="one"), Completion(text="two")), audit=audit, settings=settings
    )
    await agent.run("first")
    first_session = agent.session.session_id

    agent.reset()
    assert agent.session.session_id != first_session
    assert agent.session.history == []
    assert agent.session.turn_count == 0


async def test_a_turn_is_traceable_by_id(audit, settings) -> None:
    agent = build_agent(scripted(Completion(text="hi")), audit=audit, settings=settings)
    turn = await agent.run("hello")

    assert turn.trace_id.startswith("t-")
    assert all(
        r.detail.get("trace") == turn.trace_id
        for r in audit.all()
        if r.action in {"turn.start", "turn.end"}
    )


async def test_state_transitions_return_to_idle(audit, settings) -> None:
    states: list[AgentState] = []

    class RecordingBus(EventBus):
        def publish(self, topic, payload=None, *, source=""):
            if topic == "agent.state":
                states.append(AgentState(payload["state"]))
            super().publish(topic, payload, source=source)

    agent = build_agent(
        scripted(Completion(text="hi")), audit=audit, settings=settings
    )
    agent.bus = RecordingBus()
    await agent.run("hello")

    assert states, "state changes must be published for the UI"
    assert states[-1] is AgentState.IDLE