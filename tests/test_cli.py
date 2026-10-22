"""CLI smoke tests.

TESTING.md §4 requires a CLI smoke test. The terminal interface is the only
interface guaranteed to exist on a phone, so "it starts, prints something, and
exits cleanly" is a real acceptance criterion rather than a nicety.

Input is injected as a stream, so nothing here touches a real TTY, and the
provider is scripted, so no turn reaches a network.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from prithivi.core.agent import Agent
from prithivi.core.events import EventBus
from prithivi.core.router import Router
from prithivi.core.state import TurnSource
from prithivi.interface.cli import BANNER, CLI, HELP, Application
from prithivi.llm.provider import Completion
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.confirmation import ConfirmationBroker, ConfirmationOutcome
from prithivi.security.permissions import PermissionEngine, Policy
from prithivi.tools.basic import build_registry
from prithivi.tools.registry import ToolRegistry


def build_app(*completions: Completion, audit=None, permissions=None, researcher=None) -> tuple[Application, Console]:
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    registry = ToolRegistry(audit=audit)
    registry.register_all(build_registry())
    bus = EventBus()

    async def decline(request):
        return ConfirmationOutcome.decline("no")

    agent = Agent(
        provider=ScriptedProvider(list(completions)),
        registry=registry,
        permissions=permissions or PermissionEngine(default_level=2),
        confirmations=ConfirmationBroker(decline),
        audit=audit,
        bus=bus,
    )
    app = Application(
        agent=agent,
        router=Router(),
        permissions=agent.permissions,
        console=console,
        bus=bus,
        audit=audit,
        researcher=researcher,
    )
    return app, console


async def run_cli(app: Application, console: Console, script: str):
    cli = CLI(app, stream=io.StringIO(script), interactive=True)
    code = await cli.run()
    return code, console.file.getvalue()


def test_the_banner_names_the_project() -> None:
    # The name is figlet ASCII art, so assert on the tagline instead of a literal.
    assert "VOICE ASSISTANT" in BANNER.upper()
    assert "LOCAL-FIRST" in BANNER.upper()


async def test_the_repl_starts_and_exits_cleanly() -> None:
    app, console = build_app()
    code, out = await run_cli(app, console, "/quit\n")
    assert code == 0
    assert "VOICE ASSISTANT" in out.upper()


async def test_typing_a_message_produces_an_answer() -> None:
    app, console = build_app(Completion(text="A monad is a monoid in the category of endofunctors."))
    code, out = await run_cli(app, console, "what is a monad?\n/quit\n")
    assert code == 0
    assert "monoid" in out


async def test_a_tool_call_is_traced_in_the_output() -> None:
    from prithivi.llm.provider import ToolCall

    app, console = build_app(
        Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "6*7"}),)),
        Completion(text="That is 42."),
    )
    _, out = await run_cli(app, console, "what is 6 times 7\n/quit\n")
    assert "calculate" in out, "the tool must be shown to the user"
    assert "ok" in out
    assert "6*7 = 42" in out


async def test_a_failed_tool_is_shown_as_a_failure_not_a_success() -> None:
    from prithivi.llm.provider import ToolCall

    app, console = build_app(
        Completion(tool_calls=(ToolCall("c1", "calculate", {"expression": "2 +"}),)),
        Completion(text="That expression is malformed."),
    )
    _, out = await run_cli(app, console, "add two to nothing\n/quit\n")
    assert "fail" in out
    assert "ok " not in out.split("calculate")[-1][:40]


async def test_end_of_input_exits_without_a_slash_command() -> None:
    """A closed stdin must end the loop, not spin on it."""
    app, console = build_app()
    code, _out = await run_cli(app, console, "")
    assert code == 0


async def test_blank_lines_are_ignored() -> None:
    app, console = build_app(Completion(text="ok"))
    code, out = await run_cli(app, console, "\n\n   \n/quit\n")
    assert code == 0


@pytest.mark.parametrize("command", ["/help", "/state", "/tools", "/doctor", "/policy", "/audit"])
async def test_every_documented_command_runs(command: str) -> None:
    """Each command must at least run and not raise."""
    app, console = build_app()
    code, out = await run_cli(app, console, f"{command}\n/quit\n")
    assert code == 0
    assert out


async def test_the_audit_command_can_verify_the_chain() -> None:
    from prithivi.core.schema import Model
    from prithivi.llm.provider import ToolCall
    from prithivi.security.audit import AuditLog
    from prithivi.tools.base import ToolResult, tool

    class Args(Model):
        """Tool arguments model."""

        target: str = "x"

    @tool(name="noop", description="Test tool.", input_model=Args, permission_level=0)
    async def _noop(ctx, target: str = "x") -> ToolResult:
        return ToolResult.ok({"target": target})

    with AuditLog(":memory:") as audit:
        app, console = build_app(
            Completion(tool_calls=(ToolCall("c1", "noop", {"target": "x"}),)),
            Completion(text="done"),
            audit=audit,
        )
        registry: ToolRegistry = app.agent.registry
        registry.register(_noop, replace=True)
        _, out = await run_cli(app, console, "run noop\n/audit verify\n/quit\n")
        assert "verif" in out.lower() or "ok" in out.lower()
        assert audit.verify()[0] is True


async def test_grant_and_revoke_change_the_permission_policy() -> None:
    app, console = build_app()
    permissions = app.permissions

    await run_cli(app, console, "/grant calculate\n/quit\n")
    assert any(p.scope_pattern.startswith("calculate") for p in permissions.list_policies())

    await run_cli(app, console, "/revoke calculate\n/quit\n")
    assert not any(p.scope_pattern.startswith("calculate") for p in permissions.list_policies())


async def test_clear_forgets_the_conversation() -> None:
    app, console = build_app(Completion(text="one"), Completion(text="two"))
    await run_cli(app, console, "hello\n/clear\nhello again\n/quit\n")
    assert len(app.agent.session.history) == 2, "only the second exchange should remain"


async def test_an_unavailable_llm_is_reported_and_exits_non_interactively() -> None:

    class Unavailable(ScriptedProvider):
        async def available(self):
            return False, "PRITHIVI_LLM_API_KEY is not set"

    app, console = build_app()
    app.agent.provider = Unavailable([])
    cli = CLI(app, stream=io.StringIO(""), interactive=False)
    code = await cli.run()
    assert code == 1
    assert "not set" in console.file.getvalue()


async def test_a_confirmed_turn_uses_the_cli_as_the_confirming_surface() -> None:
    """The terminal prompt is the second factor for a real L2+ action."""
    from prithivi.core.schema import Model
    from prithivi.llm.provider import ToolCall
    from prithivi.security.audit import AuditLog
    from prithivi.tools.base import SideEffect, ToolResult, tool

    class Args(Model):
        """Tool arguments model."""

        target: str

    @tool(
        name="erase", description="Test-only.", input_model=Args,
        permission_level=3, side_effects=SideEffect.IRREVERSIBLE,
        requires_confirmation=True,
    )
    async def _erase(ctx, target: str) -> ToolResult:
        return ToolResult.ok({"erased": target})

    with AuditLog(":memory:") as audit:
        app, console = build_app(
            Completion(tool_calls=(ToolCall("c1", "erase", {"target": "/tmp/a"}),)),
            Completion(text="Understood."),
            audit=audit,
        )
        app.agent.registry.register(_erase)
        # Wire the terminal itself as the second factor, as `main.py` does.
        app.agent.confirmations = ConfirmationBroker(app.confirm)

        # A non-interactive terminal must decline rather than auto-approve.
        code, out = await run_cli(app, console, "erase /tmp/a\n/quit\n")

    assert code == 0
    assert "Declining" in out
    assert "not an interactive terminal" in out


async def test_the_turn_is_recorded_once_per_exchange() -> None:
    """Regression guard: the CLI and the agent used to both record the user turn."""
    app, console = build_app(Completion(text="a"), Completion(text="b"))
    await run_cli(app, console, "first\nsecond\n/quit\n")

    history = app.agent.session.history
    assert [e["role"] for e in history] == ["user", "assistant", "user", "assistant"]
    assert [e["content"] for e in history[:2]] == ["first", "a"]
    assert app.agent.session.turn_count == 2


async def test_the_cli_turns_over_onto_the_agent() -> None:
    """The CLI is a source, and the audit trail must record which one."""
    from prithivi.security.audit import AuditLog

    with AuditLog(":memory:") as audit:
        app, console = build_app(Completion(text="hi"), audit=audit)
        await run_cli(app, console, "hello\n/quit\n")

        starts = [r for r in audit.all() if r.action == "turn.start"]
        assert starts
        assert starts[0].detail["source"] == TurnSource.CLI.value


async def test_a_tool_granted_by_policy_runs_without_a_prompt() -> None:
    from prithivi.security.permissions import Effect

    permissions = PermissionEngine(default_level=2, policies=[
        Policy("calculate", effect=Effect.ALLOW, source="test"),
    ])
    app, console = build_app(
        Completion(text="ok"),
        permissions=permissions,
    )
    # Nothing to assert about the run itself; the point is that /tools and a turn
    # both complete without the policy causing an error.
    code, out = await run_cli(app, console, "hello\n/tools\n/quit\n")
    assert code == 0
    assert "calculate" in out


async def test_the_help_text_documents_the_real_commands() -> None:
    for command in ("/quit", "/state", "/tools", "/audit", "/doctor", "/grant", "/revoke", "/research"):
        assert command in HELP


async def test_the_research_command_answers_with_sources() -> None:
    from prithivi.rag.citations import CitationSet, Claim, Source
    from prithivi.rag.web_research import ResearchResult

    sources = (
        Source(id=1, url="https://docs.python.org/3.14/", title="Python 3.14 released", host="docs.python.org"),
        Source(id=2, url="https://blog.example.net/", title="3.14 ships", host="blog.example.net"),
    )
    citations = CitationSet(
        sources=sources,
        claims=(Claim(text="Python 3.14 was released on 7 October 2025.", source_ids=(1, 2), verification="VERIFIED"),),
    )

    async def research(self, question: str) -> ResearchResult:
        assert question == "when was python 3.14 released?"
        return ResearchResult(
            question=question,
            answer="Python 3.14 was released on 7 October 2025.",
            citations=citations,
        )

    app, console = build_app(researcher=type("Researcher", (), {"research": research})())
    code, out = await run_cli(app, console, "/research when was python 3.14 released?\n/quit\n")

    assert code == 0
    assert "7 October 2025" in out
    assert "VERIFIED" in out
    assert "sources" in out.lower()
    assert "docs.python.org" in out


async def test_the_research_command_reports_when_research_is_unavailable() -> None:
    app, console = build_app()
    code, out = await run_cli(app, console, "/research anything\n/quit\n")
    assert code == 0
    assert "not available" in out.lower()