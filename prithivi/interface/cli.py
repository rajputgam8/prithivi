"""Terminal interface — the always-available HUD.

Works with zero GUI dependencies, which matters because the target device is a
phone in Termux: a browser UI would need a port forward and a screen nobody is
looking at.

Commands:

    /help /quit /state /tools /audit /doctor /research /memory /grant /revoke /clear
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.agent import Agent, TurnCancelled
from ..core.events import EventBus
from ..core.router import Router
from ..core.state import AgentState, TurnSource
from ..security.permissions import Effect, PermissionEngine, Policy

__all__ = ["CLI", "Application"]

BANNER = r"""
  ____      _     _  ___ _           _
 |  _ \ _   _| |_  |_   | |__   __ _| |_
 | |_) | | | | __| | | | | '_ \ / _` | __|
 |  _ <| |_| | |_| | | | | |_) | (_| | |_
 |_| \_\\__,_|\__|_| |_| |_.__/ \__,_|\__|
        a local-first voice assistant
"""

HELP = """Type a message and press enter. Slash commands:

  /help        this list
  /quit        exit (or Ctrl-D)
  /state       show agent state and session counters
  /tools       list tools with permission levels
  /audit       recent audit entries (add 'verify' to check the hash chain)
  /doctor      check LLM reachability and device capabilities
  /research    answer a question from the live web, with sources: /research <question>
  /grant       allow a tool without asking: /grant open_application
  /revoke      remove a grant: /revoke open_application
  /policy      show permission levels and policies
  /clear       forget this conversation's history

Everything else is sent to Prithivi. "open instagram", "what time is it",
"calculate (125*0.18)+3400-500" all work.
"""


@dataclass
class Application:
    """The wiring shared by the CLI and other entry points."""

    agent: Agent
    router: Router
    permissions: PermissionEngine
    console: Console
    bus: EventBus
    audit: Any = None
    settings: Any = None
    researcher: Any = None
    quit_requested: bool = False

    # -- confirmations -------------------------------------------------
    async def confirm(self, request: Any) -> Any:
        """Ask the user in the terminal. A non-TTY auto-declines, never auto-approves."""
        from ..security.confirmation import ConfirmationOutcome

        if not self.console.is_terminal:
            self.console.print(
                f"[yellow]Confirmation required but this is not an interactive terminal. "
                f"Declining: {request.consequence}[/]"
            )
            return ConfirmationOutcome.decline("no interactive terminal available")

        typed = getattr(request, "typed_phrase", "")
        body = f"[bold red]Confirm needed[/] [dim]({request.level_name})[/]\n\n{request.consequence}"
        if typed:
            # L4: a y/n is not enough, so the user must repeat the phrase. Shown
            # once, in full, and read back exactly as typed.
            body += f"\n\n[bold yellow]Type this phrase to confirm:[/] [cyan]{typed}[/]"
        body += f"\n\n[dim]{request.args_preview[:200]}[/]"

        self.console.print(Panel(body, title=f"{request.tool}", border_style="yellow"))

        try:
            if typed:
                answer = (await asyncio.to_thread(input, "[bold yellow]Phrase:[/] ")).strip()
                if answer.lower() != typed.lower():
                    self.console.print("[red]Phrase did not match. Declining.[/]")
                    return ConfirmationOutcome.decline("typed confirmation phrase did not match")
                return ConfirmationOutcome.approve(factor="typed", typed_phrase=answer)

            answer = await asyncio.to_thread(input, "[bold yellow]Proceed?[/] [green]y[/]/[red]n[/]: ")
        except (EOFError, KeyboardInterrupt):
            return ConfirmationOutcome.decline("input interrupted")

        from ..security.confirmation import classify_voice_confirmation

        # Reuse the voice classifier so a typed "yes" and a spoken "yes" are
        # judged by exactly the same rules.
        return classify_voice_confirmation(answer, request, confidence=1.0)


class CLI:
    """Interactive REPL.

    Args:
        app: the wired application.
        stream: input stream; injected for tests.
        interactive: False disables the REPL loop and prompt rendering, for use
            as a library.
    """

    def __init__(self, app: Application, *, stream: Any = None, interactive: bool = True) -> None:
        self.app = app
        self.console = app.console
        self._stream = stream
        self.interactive = interactive
        self._turn = 0

    # -- main loop -----------------------------------------------------
    async def run(self) -> int:
        console = self.console
        console.print(BANNER, style="bold cyan")
        ok, reason = await self.app.agent.provider.available()
        if not ok:
            console.print(f"[red]LLM unavailable:[/] {reason}")
            console.print("[dim]Set PRITHIVI_LLM_API_KEY in .env, or run `prithivi demo` for an offline demo.[/]")
            # Unconditional, not just when interactive: a run that cannot reach
            # the model cannot serve the user, and reporting exit status 0 would
            # tell a supervisor script or a package build that it worked.
            return 1
        else:
            console.print(f"[green]Ready.[/] [dim]{reason}[/]")
        console.print("[dim]/help for commands. Ctrl-D to exit.[/]\n")

        if self.interactive:
            self._attach_bus()

        while not self.app.quit_requested:
            try:
                line = await self._read_line()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/]")
                break
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if await self._command(line):
                    break
                continue
            await self._turn_loop(line)

        return 0

    async def _turn_loop(self, utterance: str) -> None:
        """Run one turn, with barge-in support via Ctrl-C."""
        self._turn += 1
        cancel = asyncio.Event()
        try:
            # Same parity as one-shot and voice: recognised commands ("open
            # youtube", "what time is it", arithmetic) are served by the
            # deterministic router with zero model calls, so the terminal is not
            # the one surface where common commands are mysteriously slow.
            result = await self.app.agent.dispatch_deterministic(
                utterance, source=TurnSource.CLI, cancel=cancel
            )
            if result is None:
                result = await self.app.agent.run(
                    utterance,
                    source=TurnSource.CLI,
                    history=self._recent_history(),
                    cancel=cancel,
                )
        except TurnCancelled:
            self.console.print("[yellow]cancelled[/]")
            return
        except KeyboardInterrupt:
            self.console.print("\n[yellow]interrupted[/]")
            return

        self._print_result(result)

    def _print_result(self, result: Any) -> None:
        colour = "green" if not result.blocked and result.succeeded else "yellow"
        self.console.print(f"[{colour}]{result.text}[/]")

        if result.tool_results:
            table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
            table.add_column("tool", style="cyan")
            table.add_column("result")
            for name, tool_result in result.tool_results:
                mark = "[green]ok[/]" if tool_result.success else "[red]fail[/]"
                table.add_row(name, f"{mark} {tool_result.spoken()}")
            self.console.print(table)

        chips = result.narration.chips
        if chips:
            self.console.print(f"[dim]{' | '.join(chips)}[/]")
        meta = f"[dim]{result.iterations} iteration(s), {result.elapsed_ms} ms, trace {result.trace_id}[/]"
        self.console.print(meta)

    def _recent_history(self, turns: int = 6) -> list[tuple[str, str]]:
        """Prior turns, excluding the one in flight.

        The agent appends to the transcript at the *end* of a turn, so reading the
        whole history here cannot include the utterance about to be sent. This
        used to also skip the last entry, because the CLI pre-recorded the user
        message itself — which meant every user turn was stored twice and the
        assistant's reply was dropped from the next turn's context.
        """
        return [
            (entry["role"], entry["content"])
            for entry in self.app.agent.session.history[-turns:]
            if entry["role"] in {"user", "assistant"}
        ]

    # -- input ---------------------------------------------------------
    async def _read_line(self) -> str | None:
        if self._stream is not None:
            data = self._stream.readline()
            if data == "":
                return None
            return data.rstrip("\n")
        if not self.interactive:
            return None
        try:
            line = await asyncio.to_thread(input, "[bold cyan]you ›[/] ")
        except (EOFError, KeyboardInterrupt):
            return None
        return line

    def _attach_bus(self) -> None:
        """Mirror agent state to the terminal."""
        labels = {
            AgentState.THINKING: "[dim]thinking…[/]",
            AgentState.EXECUTING: "[dim]working…[/]",
            AgentState.SEARCHING: "[dim]searching…[/]",
            AgentState.SPEAKING: "[dim]speaking…[/]",
        }

        def on_state(event: Any) -> None:
            payload = event.payload or {}
            text = labels.get(AgentState(payload.get("state", "")), "")
            if text and payload.get("tool"):
                text = f"[dim]{payload['tool']}…[/]"
            if text:
                self.console.print(text, end="\r")

        self.app.bus.subscribe("agent.state", on_state)

    # -- commands ------------------------------------------------------
    async def _command(self, line: str) -> bool:
        """Handle a slash command. Returns True when the REPL should stop."""
        name, _, rest = line[1:].partition(" ")
        name = name.strip().lower()
        rest = rest.strip()

        if name in {"quit", "exit", "q"}:
            return True
        if name in {"help", "h", "?"}:
            self.console.print(HELP)
        elif name == "state":
            self._cmd_state()
        elif name == "tools":
            self._cmd_tools()
        elif name == "audit":
            self._cmd_audit(rest)
        elif name == "doctor":
            await self._cmd_doctor()
        elif name == "research":
            await self._cmd_research(rest)
        elif name == "grant":
            self._cmd_grant(rest)
        elif name == "revoke":
            self._cmd_revoke(rest)
        elif name == "policy":
            self._cmd_policy()
        elif name == "clear":
            self.app.agent.session.history.clear()
            self.console.print("[green]conversation history cleared[/]")
        else:
            self.console.print(f"[red]unknown command[/] /{name} — try /help")
        return False

    def _cmd_state(self) -> None:
        session = self.app.agent.session
        table = Table(show_header=False, box=None)
        table.add_column("field", style="cyan")
        table.add_column("value")
        table.add_row("state", session.state.value)
        table.add_row("session", session.session_id)
        table.add_row("turns", str(session.turn_count))
        table.add_row("tools", ", ".join(self.app.agent.registry.names) or "none")
        self.console.print(table)

    def _cmd_tools(self) -> None:
        table = Table(title="tools")
        table.add_column("name", style="cyan")
        table.add_column("level")
        table.add_column("effects")
        table.add_column("confirm")
        for entry in self.app.agent.registry.summary():
            table.add_row(
                entry["name"],
                f"L{entry['level']} {entry['level_name']}",
                entry["side_effects"],
                "yes" if entry["confirm"] else "no",
            )
        self.console.print(table)

    def _cmd_audit(self, rest: str) -> None:
        if self.app.audit is None:
            self.console.print("[yellow]no audit log attached[/]")
            return
        if rest.startswith("verify"):
            ok, bad = self.app.audit.verify()
            if ok:
                self.console.print(f"[green]audit chain intact[/] ({len(self.app.audit)} records)")
            else:
                self.console.print(f"[red]audit chain BROKEN at record {bad}[/]")
            return
        table = Table(title="recent audit")
        table.add_column("ts", style="dim")
        table.add_column("action", style="cyan")
        table.add_column("outcome")
        table.add_column("target", style="dim")
        for record in self.app.audit.tail(20):
            colour = {"success": "green", "denied": "yellow"}.get(record.outcome, "red")
            table.add_row(record.ts, record.action, f"[{colour}]{record.outcome}[/]", record.target or "")
        self.console.print(table)

    async def _cmd_doctor(self) -> None:
        table = Table(title="doctor")
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail", style="dim")
        table.add_row("LLM", *self._status(await self.app.agent.provider.available()))
        for name, available, reason in self.app.agent.registry.readiness():
            status, detail = self._status((available, "" if available else reason))
            table.add_row(f"tool:{name}", status, detail)
        self.console.print(table)

        from ..platform.detect import describe_environment, detect
        self.console.print(f"[dim]{describe_environment()}[/]")
        self.console.print(f"[dim]{detect().describe()}[/]")
        for problem in self.app.settings.validate() if self.app.settings else []:
            self.console.print(f"[yellow]config:[/] {problem}")

    @staticmethod
    def _status(result: tuple[bool, str]) -> tuple[str, str]:
        """A ``(available, detail)`` pair as a coloured status plus detail cell."""
        ok, reason = result
        return ("[green]ok[/]" if ok else "[red]unavailable[/]", reason or "")

    async def _cmd_research(self, question: str) -> None:
        """Answer from the live web, showing sources and how well they agree.

        Bypasses the chat turn on purpose: research is bounded, cited, and worth
        seeing raw. The answer still goes through the same permission-free
        read-only path as the tool, so nothing here can act on the user's behalf.
        """
        if not question:
            self.console.print("[red]usage:[/] /research <question>")
            return
        if self.app.researcher is None:
            self.console.print("[yellow]web research is not available in this build[/]")
            return

        self.console.print(f"[dim]researching:[/] {question}")
        try:
            result = await self.app.researcher.research(question)
        except Exception as exc:  # noqa: BLE001 - a research failure is a message, not a crash
            self.console.print(f"[red]research failed:[/] {exc}")
            return

        colour = "yellow" if result.degraded else "green"
        self.console.print(f"[bold {colour}]{result.answer}[/]")

        for claim in result.citations.claims[:8]:
            mark = {"VERIFIED": "green", "UNVERIFIED": "yellow", "CONFLICTED": "red"}.get(claim.verification, "dim")
            self.console.print(
                f"  [{mark}]{claim.verification}[/] {claim.text[:160]} "
                f"{result.citations.marker_for(claim.source_ids)}"
            )
        for conflict in result.citations.conflicts:
            self.console.print(f"  [red]conflict[/] {conflict.describe()}")
        if result.citations.sources:
            self.console.print(Panel(result.citations.render_sources(), title="sources", border_style="dim"))
        for note in result.notes:
            self.console.print(f"[yellow]note:[/] {note}")
        self.console.print(
            f"[dim]{result.source_count} source(s), {result.duration_ms} ms, "
            f"provenance={result.provenance.value}[/]"
        )

    def _cmd_grant(self, rest: str) -> None:
        if not rest:
            self.console.print("[red]usage:[/] /grant <tool>")
            return
        self.app.permissions.add_policy(Policy(f"{rest}:*", Effect.ALLOW))
        self.console.print(f"[green]granted[/] {rest} (no confirmation needed until revoked)")

    def _cmd_revoke(self, rest: str) -> None:
        if not rest:
            self.console.print("[red]usage:[/] /revoke <tool>")
            return
        removed = self.app.permissions.remove_policy(f"{rest}:*")
        self.console.print(
            f"[green]revoked[/] {rest} ({removed} policy/policies removed)"
            if removed
            else f"[yellow]no grant found for[/] {rest}"
        )

    def _cmd_policy(self) -> None:
        table = Table(title="permission policies")
        table.add_column("scope", style="cyan")
        table.add_column("effect")
        table.add_column("uses")
        table.add_column("source", style="dim")
        for policy in self.app.permissions.list_policies():
            uses = "unlimited" if policy.max_uses is None else str(policy.remaining_uses or 0)
            table.add_row(policy.scope_pattern, policy.effect, uses, policy.source)
        self.console.print(table)
        self.console.print(f"[dim]default level: L{self.app.permissions.default_level}[/]")
