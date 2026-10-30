#!/usr/bin/env python
"""Prithivi entry point.

One place where every component is constructed, so ``interface/cli.py`` and the
future voice interface cannot drift apart and disagree about what is enabled.

    python main.py                  # interactive terminal
    python main.py "what time is it" # one-shot
    python main.py --demo           # offline, no API key needed
    python main.py doctor           # environment report
    python main.py tools            # list tools and permission levels
    python main.py verify           # audit-chain check

Run it with Termux's own interpreter:
``/data/data/com.termux/files/usr/bin/python main.py``. The PRoot copy of Python
cannot load the native wheels this project uses.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from prithivi.core.agent import Agent
from prithivi.core.config import Settings
from prithivi.core.events import EventBus
from prithivi.core.narrator import Narrator
from prithivi.core.router import Router
from prithivi.core.state import TurnSource
from prithivi.db.connection import Database
from prithivi.db.repositories.sessions import SessionRepository
from prithivi.db.repositories.settings import SettingsRepository
from prithivi.interface.cli import CLI, Application
from prithivi.llm.openai_compat import build_provider_from_settings
from prithivi.llm.scripted import ScriptedProvider
from prithivi.memory.long_term import MemoryStore
from prithivi.observability.logging import configure_logging, get_logger
from prithivi.platform.detect import describe_environment, in_proot
from prithivi.rag.embeddings import HashEmbedder
from prithivi.rag.knowledge_base import KnowledgeBase
from prithivi.security.audit import AuditLog
from prithivi.security.confirmation import (
    ConfirmationBroker,
    ConfirmationOutcome,
    classify_voice_confirmation,
)
from prithivi.security.permissions import Effect, PermissionEngine, Policy
from prithivi.tools import build_tool_specs
from prithivi.tools.base import ToolContext
from prithivi.tools.registry import ToolRegistry

log = get_logger("prithivi.main")

ConfirmHandler = Callable[[Any], Any]


@dataclass
class Runtime:
    """A fully wired Prithivi instance."""

    settings: Settings
    bus: EventBus
    provider: Any
    registry: ToolRegistry
    permissions: PermissionEngine
    confirmations: ConfirmationBroker
    audit: AuditLog
    agent: Agent
    router: Router
    console: Console
    _researcher: Any = None
    db: Database | None = None
    memory: MemoryStore | None = None
    knowledge: KnowledgeBase | None = None
    sessions: Any = None
    settings_repo: SettingsRepository | None = None

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        console: Console,
        confirm_handler: ConfirmHandler | None = None,
        provider: Any = None,
        audit: AuditLog | None = None,
    ) -> Runtime:
        bus = EventBus()
        settings.ensure_dirs()

        audit = audit if audit is not None else AuditLog(settings.audit_db_path)
        db = None
        memory = None
        knowledge = None
        sessions = None
        settings_repo = None
        try:
            db = Database(settings.db_path)
            db.migrate()
            sessions = SessionRepository(db)
            settings_repo = SettingsRepository(db)
            memory_enabled = bool(getattr(settings, "memory_enabled", True))
            stored = settings_repo.get("memory.enabled")
            if isinstance(stored, bool):
                memory_enabled = stored
            memory = MemoryStore(db, embedder=HashEmbedder(), enabled=memory_enabled)
            knowledge = KnowledgeBase(
                db,
                embedder=HashEmbedder(),
                chunk_size=int(getattr(settings, "chunk_size", 512)),
                chunk_overlap=int(getattr(settings, "chunk_overlap", 64)),
                top_k=int(getattr(settings, "retrieval_top_k", 8)),
            )
        except Exception:  # Phase 7 services must never block startup
            log.exception("could not initialise Phase 7 memory/knowledge services")

        registry = ToolRegistry(audit=audit)
        # The web tools are not registered at all when egress is off: offering
        # the model a tool that is guaranteed to refuse is worse than not having it.
        registry.register_all(
            build_tool_specs(
                include_web=settings.egress_allow_network,
                include_memory=memory is not None and bool(getattr(memory, "enabled", False)),
                include_knowledge=knowledge is not None,
            )
        )

        permissions = PermissionEngine(default_level=settings.permission_default_level)
        # Reading the clock and pure arithmetic cannot damage anything; asking
        # every time would only train the user to reflexively approve prompts.
        permissions.add_policy(
            Policy("get_time:*", effect=Effect.ALLOW, reason="read-only", source="builtin")
        )
        permissions.add_policy(
            Policy("calculate:*", effect=Effect.ALLOW, reason="pure function", source="builtin")
        )

        confirmations = ConfirmationBroker(confirm_handler)
        provider = provider if provider is not None else build_provider_from_settings(settings)
        router = Router(provider=provider, app_names=tuple(registry.names))
        agent = Agent(
            provider=provider,
            registry=registry,
            permissions=permissions,
            confirmations=confirmations,
            audit=audit,
            settings=settings,
            bus=bus,
            narrator=Narrator(),
            router=router,
            memory=memory,
            knowledge=knowledge,
            sessions=sessions,
        )
        return cls(
            settings=settings,
            bus=bus,
            provider=provider,
            registry=registry,
            permissions=permissions,
            confirmations=confirmations,
            audit=audit,
            agent=agent,
            router=router,
            console=console,
            db=db,
            memory=memory,
            knowledge=knowledge,
            sessions=sessions,
            settings_repo=settings_repo,
        )

    def application(self) -> Application:
        return Application(
            agent=self.agent,
            router=self.router,
            permissions=self.permissions,
            console=self.console,
            bus=self.bus,
            audit=self.audit,
            settings=self.settings,
            researcher=self.researcher,
            memory=self.memory,
            knowledge=self.knowledge,
            sessions=self.sessions,
            db=self.db,
            settings_repo=self.settings_repo,
        )

    @property
    def researcher(self) -> Any:
        """The web researcher, built on first use.

        Built lazily because it opens an HTTP client, and ``prithivi doctor``
        should not create one it never uses.
        """
        if self._researcher is None:
            from prithivi.rag.web_research import build_researcher

            self._researcher = build_researcher(self.settings, llm=self.provider)
        return self._researcher

    def tool_context(self, *, source: TurnSource = TurnSource.CLI) -> ToolContext:
        return ToolContext(
            settings=self.settings,
            audit=self.audit,
            bus=self.bus,
            confirmation=self.confirmations,
            session_id=self.agent.session.session_id,
            source=source.value,
            scratch=self.agent.scratch,
            memory=self.memory,
            knowledge=self.knowledge,
        )


def make_demo_provider() -> ScriptedProvider:
    """A fixed script, so ``--demo`` proves the plumbing without a network."""
    from prithivi.llm.provider import Completion, ToolCall

    return ScriptedProvider(
        [
            Completion(tool_calls=(ToolCall("demo-1", "get_time", {"timezone": "local"}),)),
            Completion(text="It is currently 14:32 in your local time zone."),
        ]
    )


def stdin_confirm(console: Console) -> ConfirmHandler:
    """Build a handler that asks for consent on stdin.

    Auto-*declines* when there is no terminal. Never auto-approves: a batch or
    piped invocation must not silently gain consent.
    """

    def handler(request: Any) -> Any:
        if not sys.stdin.isatty():
            console.print(
                f"[yellow]Confirmation needed but this is not a terminal; "
                f"declining:[/] {request.consequence}"
            )
            return ConfirmationOutcome.decline("no interactive terminal")
        console.print(
            f"[yellow]Confirm[/] [{request.level_name}] {request.consequence}\n"
            f"[dim]{request.args_preview[:200]}[/]"
        )
        try:
            answer = input("proceed? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            return ConfirmationOutcome.decline("input interrupted")
        return classify_voice_confirmation(answer, request, confidence=1.0)

    return handler


COMMANDS = ("doctor", "verify", "tools", "voice", "web")


def build_parser() -> argparse.ArgumentParser:
    # Deliberately no ``subparsers``: they cannot coexist with a
    # ``nargs="*"`` utterance, because argparse would try to match the first word
    # of the utterance against the command names ("what time is it" is not a
    # valid subcommand). Commands are dispatched in ``async_main`` instead.
    parser = argparse.ArgumentParser(
        prog="prithivi",
        description="Prithivi — a local-first voice assistant.",
        epilog=f"commands: {', '.join(COMMANDS)}  (e.g. `prithivi doctor`)",
    )
    parser.add_argument("words", nargs="*", help="a command name, or something to say once")
    parser.add_argument("--demo", action="store_true", help="offline mode, no API key required")
    parser.add_argument("--env-file", default=None, help="path to .env (default: ./data/.env)")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-format", default=None, choices=["text", "json"])
    return parser


def load_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, str] = {}
    if args.log_level:
        overrides["PRITHIVI_LOG_LEVEL"] = args.log_level
    if args.log_format:
        overrides["PRITHIVI_LOG_FORMAT"] = args.log_format
    return Settings.from_env(args.env_file, overrides=overrides or None)


async def run_one_shot(runtime: Runtime, utterance: str) -> int:
    """Answer a single command, preferring the deterministic router.

    The router path is not a shortcut around security: it runs the same
    permission check, the same confirmation flow, and the same audited executor.
    """
    console = runtime.console
    console.print(f"[bold cyan]you ›[/] {utterance}")

    outcome = runtime.router.route(utterance)
    if outcome.needs_clarification:
        console.print(f"[yellow]prithivi ›[/] {outcome.question}")
        return 1

    # The router's fast path, the permission engine, the confirmation flow and the
    # audited executor all live in one place, shared with the voice loop, so a
    # typed "open youtube" and a spoken one behave identically.
    handled = await runtime.agent.dispatch_deterministic(utterance, source=TurnSource.CLI)
    if handled is not None:
        colour = "green" if not handled.blocked else "yellow"
        console.print(f"[bold {colour}]prithivi ›[/] {handled.text}")
        for name, tool_result in handled.tool_results:
            if not tool_result.success and tool_result.error is not None:
                if tool_result.error.remedy:
                    console.print(f"[dim]{tool_result.error.remedy}[/]")
            elif handled.text != tool_result.spoken():
                console.print(f"  [green]{name}[/]: {tool_result.spoken()}")
        acted_ok = all(r.success for _, r in handled.tool_results) and not handled.blocked
        return 0 if acted_ok else 1

    turn = await runtime.agent.run(utterance, source=TurnSource.CLI)
    colour = "green" if turn.succeeded and not turn.blocked else "yellow"
    console.print(f"[bold {colour}]prithivi ›[/] {turn.text}")
    for name, tool_result in turn.tool_results:
        mark = "green" if tool_result.success else "red"
        console.print(f"  [{mark}]{name}[/]: {tool_result.spoken()}")
    if turn.narration.chips:
        console.print(f"[dim]{' | '.join(turn.narration.chips)}[/]")
    return 0 if turn.succeeded and not turn.blocked else 1


async def run_voice(settings: Settings, console: Console, *, demo: bool = False) -> int:
    """Run the Phase 2 voice loop: listen → agent → speak.

    Voice mode builds its own runtime because the confirmation second factor is
    the microphone, not the keyboard: the broker is wired to speak the
    consequence and hear the answer.
    """
    from prithivi.voice import build_stt, build_tts, make_confirm_handler
    from prithivi.voice.conversation import VoiceConversation
    from prithivi.voice.wake_word import WakeWord

    stt = build_stt(settings)
    tts = build_tts(settings)

    ok, reason = await stt.available()
    if not ok:
        console.print(f"[red]voice input unavailable:[/] {reason}")
        console.print(
            "[dim]Install the Termux:API app from F-Droid and run `pkg install termux-api`.[/]"
        )
        return 1

    confirm_handler = make_confirm_handler(stt, tts, console=console)
    if demo:
        runtime = Runtime.build(
            Settings.for_testing(log_level=settings.log_level),
            console=console,
            confirm_handler=confirm_handler,
            provider=make_demo_provider(),
            audit=AuditLog(":memory:"),
        )
    else:
        runtime = Runtime.build(settings, console=console, confirm_handler=confirm_handler)

    conversation = VoiceConversation(
        agent=runtime.agent,
        stt=stt,
        tts=tts,
        bus=runtime.bus,
        console=console,
        wake_word=WakeWord(getattr(settings, "wake_word", "hey prithivi")),
        stream_replies=bool(getattr(settings, "llm_stream", True)),
        tts_slow_after_s=float(getattr(settings, "tts_slow_after_s", 1.5)),
        tts_coalesce_chars=int(getattr(settings, "tts_coalesce_chars", 220)),
    )
    console.print(
        "[dim]Voice mode. Say 'hey prithivi …' or just speak; say 'goodbye' to stop.[/]"
    )
    return await conversation.run()


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    # A single leading word that names a known command is treated as that
    # command; anything else is an utterance, verbatim.
    command: str | None = None
    words = list(args.words)
    if words and words[0] in COMMANDS:
        command, words = words[0], words[1:]

    settings = load_settings(args)
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    for problem in settings.validate():
        console.print(f"[yellow]config:[/] {problem}")

    if command == "verify":
        audit = AuditLog(settings.audit_db_path)
        ok, bad = audit.verify()
        if ok:
            console.print(f"[green]audit chain intact[/] ({len(audit)} records)")
            return 0
        console.print(f"[red]audit chain BROKEN at record {bad}[/]")
        return 1

    if command == "voice":
        return await run_voice(settings, console, demo=args.demo)

    if command == "web":
        from prithivi.api.server import serve

        if args.demo:
            runtime = Runtime.build(
                Settings.for_testing(log_level=settings.log_level),
                console=console,
                confirm_handler=stdin_confirm(console),
                provider=make_demo_provider(),
                audit=AuditLog(":memory:"),
            )
        else:
            runtime = Runtime.build(settings, console=console, confirm_handler=stdin_confirm(console))
        return await serve(settings, runtime)

    if args.demo:
        # In-memory audit: a demo must not pollute the real tamper-evident log.
        runtime = Runtime.build(
            Settings.for_testing(log_level=settings.log_level),
            console=console,
            confirm_handler=stdin_confirm(console),
            provider=make_demo_provider(),
            audit=AuditLog(":memory:"),
        )
    else:
        runtime = Runtime.build(settings, console=console, confirm_handler=stdin_confirm(console))

    cli = CLI(runtime.application(), interactive=sys.stdin.isatty())

    if command == "tools":
        await cli._command("/tools")
        return 0
    if command == "doctor":
        await cli._command("/doctor")
        return 0
    if words:
        return await run_one_shot(runtime, " ".join(words))

    console.print(f"[dim]{describe_environment()}[/]")
    if in_proot():
        console.print(
            "[yellow]Tip:[/] running under PRoot blocks app and browser control. "
            "Run [bold]~/prithivi[/] from native Termux to open YouTube, Instagram, and other apps."
        )
    return await cli.run()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)
        return 130
    except Exception as exc:
        log.exception("fatal error")
        print(f"prithivi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())