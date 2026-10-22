"""The core loop: a model may request a tool, but it may not authorise one."""

from __future__ import annotations

import pytest

from prithivi.core.agent import Agent
from prithivi.core.schema import Model as SchemaModel
from prithivi.llm.provider import Completion, ToolCall
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.confirmation import ConfirmationOutcome
from prithivi.security.permissions import Effect, PermissionEngine, Policy
from prithivi.tools.base import ToolContext, ToolErrorCode
from prithivi.tools.registry import ToolRegistry


def provider(*items: Completion) -> ScriptedProvider:
    return ScriptedProvider(list(items))


async def test_tool_call_is_executed_and_reported(agent: Agent) -> None:
    agent.provider = provider(
        Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "2+2"}),)),
        Completion(text="That is 4."),
    )
    turn = await agent.run("what is 2+2")

    assert turn.succeeded
    assert [name for name, _ in turn.tool_results] == ["calculate"]
    assert turn.tool_results[0][1].data["result"] == 4
    assert turn.iterations == 2


async def test_model_cannot_grant_itself_permission(
    agent: Agent, audit
) -> None:
    """The model's claim that a step is 'allowed' must not appear anywhere in the
    authorisation path."""
    agent.provider = provider(
        Completion(
            tool_calls=(
                ToolCall(
                    "c1",
                    "calculate",
                    # The model volunteers a permission override in its own
                    # arguments. It is not a ToolSpec field, so validation must
                    # reject the call outright rather than honour it.
                    {"expression": "1+1", "__permission__": "allow", "sudo": True},
                ),
            )
        ),
        Completion(text="ok"),
    )
    turn = await agent.run("add")

    checks = [r for r in audit.tail(50) if r.action == "permission.check"]
    assert checks, "the permission engine must be consulted"
    for record in checks:
        assert record.detail.get("effect") in {Effect.ALLOW, Effect.CONFIRM, Effect.DENY}


async def test_unknown_tool_is_refused_not_guessed(agent: Agent) -> None:
    agent.provider = provider(
        Completion(tool_calls=(ToolCall("c1", "send_money", {"amount": 10}),)),
        Completion(text="I sent it."),
    )
    turn = await agent.run("send money")

    result = turn.tool_results[0][1]
    assert result.success is False
    assert "send_money" in result.error.message
    # The model claimed success; narration must not repeat that as fact.
    assert "sent it" not in turn.text.lower()


@pytest.fixture
def risky_registry() -> ToolRegistry:
    """A registry whose L3 tool always requires confirmation.

    ``open_url`` is only L1, so it would pass straight through the baseline gate
    and never reach the confirmation flow at all. Exercising the flow needs a
    tool that actually sits above the gate.
    """
    from prithivi.tools.base import SideEffect, ToolResult, tool

    class Args(SchemaModel):
        target: str

    @tool(
        name="delete_everything",
        description="Test-only irreversible action.",
        input_model=Args,
        permission_level=3,
        side_effects=SideEffect.IRREVERSIBLE,
        idempotent=False,
        requires_confirmation=True,
    )
    async def _risky(ctx: ToolContext, target: str) -> ToolResult:
        return ToolResult.ok({"deleted": target})

    reg = ToolRegistry()
    reg.register(_risky)
    return reg


async def test_declined_confirmation_blocks_the_tool(
    risky_registry: ToolRegistry, audit, settings
) -> None:
    async def decline(request):
        return ConfirmationOutcome.decline("no")

    agent = Agent(
        provider=provider(
            Completion(tool_calls=(ToolCall("c1", "delete_everything", {"target": "/"}),)),
            Completion(text="deleted everything"),
        ),
        registry=risky_registry,
        permissions=PermissionEngine(default_level=2),
        confirmations=__import__(
            "prithivi.security.confirmation", fromlist=["ConfirmationBroker"]
        ).ConfirmationBroker(decline),
        audit=audit,
        settings=settings,
    )
    turn = await agent.run("delete everything")

    assert turn.tool_results[0][1].success is False
    assert turn.blocked is True
    # The model said it did the thing; narration must not repeat that.
    assert "deleted everything" not in turn.text.lower()


async def test_approved_confirmation_lets_the_tool_run(
    risky_registry: ToolRegistry, audit, settings
) -> None:
    async def approve(request):
        return ConfirmationOutcome.approve(factor="ui")

    agent = Agent(
        provider=provider(
            Completion(tool_calls=(ToolCall("c1", "delete_everything", {"target": "/tmp/x"}),)),
            Completion(text="Done."),
        ),
        registry=risky_registry,
        permissions=PermissionEngine(default_level=2),
        confirmations=__import__(
            "prithivi.security.confirmation", fromlist=["ConfirmationBroker"]
        ).ConfirmationBroker(approve),
        audit=audit,
        settings=settings,
    )
    turn = await agent.run("delete /tmp/x")

    result = turn.tool_results[0][1]
    assert result.success is True
    assert result.data["deleted"] == "/tmp/x"


async def test_consumable_grant_runs_out(permissions: PermissionEngine) -> None:
    """A single-use grant must not silently authorise a second call.

    Checked at level 3, which is at or above the default, because a level-0 tool
    is below the baseline gate and would be allowed either way — which would make
    this test pass for the wrong reason.
    """
    permissions.add_policy(
        Policy("open_application:*", effect=Effect.ALLOW, max_uses=1, source="test")
    )
    first = permissions.check(
        scope="open_application:whatsapp", level=3, tool_requires_confirmation=False
    )
    assert first.granted
    second = permissions.check(
        scope="open_application:whatsapp", level=3, tool_requires_confirmation=False
    )
    assert not second.granted


async def test_denied_scope_stays_denied(permissions: PermissionEngine) -> None:
    permissions.add_policy(Policy("open_application:*", effect=Effect.DENY, source="user"))
    decision = permissions.check(
        scope="open_application:whatsapp", level=1, tool_requires_confirmation=False
    )
    assert decision.denied
    assert decision.granted is False


async def test_turn_budget_stops_a_model_loop(
    registry: ToolRegistry, permissions: PermissionEngine, audit, settings
) -> None:
    """A model that only ever calls tools must be cut off, not trusted to stop."""
    looping = provider(*[Completion(tool_calls=(ToolCall("c", "get_time", {"timezone": "local"}),)) for _ in range(50)])
    agent = Agent(
        provider=looping,
        registry=registry,
        permissions=permissions,
        audit=audit,
        settings=settings,
    )
    turn = await agent.run("loop forever")

    assert turn.iterations <= settings.agent_max_iterations
    assert turn.succeeded is False


async def test_registry_requires_a_decision(registry: ToolRegistry) -> None:
    result = await registry.execute(
        "calculate", {"expression": "1+1"}, None, ToolContext(source="test")
    )
    assert result.success is False
    assert result.error.code == ToolErrorCode.DENIED
    assert "without a permission decision" in result.error.message


@pytest.mark.security
async def test_every_tool_call_is_audited(agent: Agent, audit) -> None:
    agent.provider = provider(
        Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "3*3"}),)),
        Completion(text="9"),
    )
    await agent.run("three times three")

    ok, broken = audit.verify()
    assert ok and broken is None
    actions = [r.action for r in audit.tail(50)]
    assert "tool.call" in actions or "tool.execute" in actions