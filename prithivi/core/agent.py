"""The agent loop.

A bounded ReAct cycle (ARCHITECTURE.md S3.1, S8):

    hydrate → decide → [authorise → confirm → execute] × N → narrate

Invariants:

* **The model never authorises anything.** It emits tool *requests*; every one
  passes through the permission engine, and an L2+ action additionally waits for a
  human second factor. A denial is terminal — the loop will not try to reach the
  same outcome another way.
* **Bounded.** At most ``max_iterations`` model calls and a wall-clock budget.
  A model stuck in a loop gets cut off and told so.
* **Honest.** Narration is rebuilt from actual :class:`ToolResult` objects, so a
  failed action cannot be reported as a success.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.events import EventBus
from ..core.narrator import Narration, Narrator, Provenance, ProvenanceKind
from ..core.state import AgentState, Modality, SessionState, TurnSource
from ..llm.prompts import SYSTEM_PROMPT
from ..llm.provider import Completion, DeltaCallback, LLMError, LLMProvider, Message
from ..security.audit import AuditLog
from ..security.confirmation import ConfirmationBroker
from ..security.permissions import PermissionDecision, PermissionEngine
from ..tools.base import ToolContext, ToolErrorCode, ToolResult
from ..tools.registry import ToolRegistry
from .router import IntentDecision, Router

__all__ = ["Agent", "TurnCancelled", "TurnResult"]

log = logging.getLogger("prithivi.agent")


class TurnCancelled(Exception):
    """The turn was interrupted (barge-in, stop button, timeout)."""


@dataclass
class TurnResult:
    """Everything one user turn produced."""

    text: str
    narration: Narration
    state: AgentState
    tool_results: list[tuple[str, ToolResult]] = field(default_factory=list)
    iterations: int = 0
    elapsed_ms: int = 0
    trace_id: str = ""
    blocked: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "state": self.state.value,
            "iterations": self.iterations,
            "elapsed_ms": self.elapsed_ms,
            "trace_id": self.trace_id,
            "blocked": self.blocked,
            "error": self.error,
            "provenance": [p.chip() for p in self.narration.provenance],
            "tools": [
                {"tool": n, "success": r.success, "summary": r.summary()} for n, r in self.tool_results
            ],
        }


class Agent:
    """Runs one turn at a time.

    Args:
        provider: the model.
        registry: the allowlisted tools.
        permissions: the enforcement point.
        confirmations: broker used for L2+ second factors.
        audit: append-only log.
        settings: runtime configuration.
        bus: event bus for UI updates.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        permissions: PermissionEngine,
        confirmations: ConfirmationBroker | None = None,
        audit: AuditLog | None = None,
        settings: Any = None,
        bus: EventBus | None = None,
        narrator: Narrator | None = None,
        router: Router | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.permissions = permissions
        self.confirmations = confirmations or ConfirmationBroker()
        self.audit = audit
        self.settings = settings
        self.bus = bus or EventBus()
        self.narrator = narrator or Narrator()
        # The fast-path router the deterministic shortcut uses. It never needs the
        # model: only its regex table is exercised here, so it is built without a
        # provider (the LLM path stays the responsibility of ``run``).
        self.router = router or Router(provider=None, app_names=registry.names)
        self.system_prompt = system_prompt
        # One scratch dict for the whole agent, shared by every ToolContext this
        # agent builds. Tools cache their expensive, reusable machinery here — the
        # platform adapter (which probes with several subprocesses) and the HTTP
        # clients (which own connection pools). A fresh dict per tool call, as it
        # used to be, meant re-probing the device and re-opening TLS on every turn.
        self._scratch: dict[str, Any] = {}
        self._session = SessionState(session_id=f"s-{uuid.uuid4().hex[:10]}")

    # -- session -------------------------------------------------------
    @property
    def session(self) -> SessionState:
        return self._session

    @property
    def scratch(self) -> dict[str, Any]:
        """Process-lifetime cache shared by this agent's tool calls.

        Exposed so other entry points (``Runtime.tool_context``) can hand the same
        dict to a tool, rather than starting a cold cache on every invocation.
        """
        return self._scratch

    def reset(self) -> None:
        self._session = SessionState(session_id=f"s-{uuid.uuid4().hex[:10]}")
        self.confirmations.clear()

    # -- the turn ------------------------------------------------------
    async def run(
        self,
        utterance: str,
        *,
        source: TurnSource = TurnSource.CLI,
        modality: Modality = Modality.TEXT,
        history: Sequence[tuple[str, str]] = (),
        context_blocks: Sequence[str] = (),
        cancel: asyncio.Event | None = None,
        on_text: DeltaCallback | None = None,
    ) -> TurnResult:
        """Handle one user turn end to end.

        Args:
            on_text: optional streaming sink. When given, the model's answer is
                forwarded to it in pieces as it is generated, so a voice turn can
                start speaking before the reply is complete. The returned
                :class:`TurnResult` is unaffected — it always holds the full text,
                so callers that do not stream see no difference.
        """
        trace_id = f"t-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        max_iterations = int(getattr(self.settings, "agent_max_iterations", 6) or 6)
        budget_s = float(getattr(self.settings, "agent_turn_budget_s", 45.0) or 45.0)
        deadline = started + budget_s

        self._session = self._session.evolve(turn_count=self._session.turn_count + 1, modality=modality)
        self._set_state(AgentState.THINKING, source)

        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        for block in context_blocks:
            messages.append(Message(role="user", content=block))
        for role, content in history:
            messages.append(Message(role="user" if role == "user" else "assistant", content=content))
        messages.append(Message(role="user", content=utterance))

        tool_results: list[tuple[str, ToolResult]] = []
        tool_names: list[str] = []
        provenance: list[Provenance] = []
        iterations = 0
        final_text = ""
        blocked = False
        error: str | None = None

        self._audit("turn.start", "info", detail={
            "utterance": utterance[:400], "source": source.value, "trace": trace_id,
        })

        while iterations < max_iterations:
            if time.monotonic() > deadline:
                error = f"stopped after {budget_s:g}s"
                self._audit("turn.timeout", "error", detail={"trace": trace_id})
                break
            if cancel is not None and cancel.is_set():
                self._audit("turn.cancelled", "info", detail={"trace": trace_id})
                self._set_state(AgentState.WAITING_USER, source)
                raise TurnCancelled("cancelled by user")

            iterations += 1
            try:
                completion = await self.provider.complete(
                    messages,
                    tools=self.registry.schemas(),
                    temperature=_routing_temperature(self.provider, source),
                    on_delta=on_text,
                )
            except LLMError as exc:
                error = str(exc)
                self._audit("turn.llm_error", "error", detail={"trace": trace_id, "error": error})
                self._set_state(AgentState.ERROR, source)
                break

            messages.append(_assistant_message(completion))

            if not completion.wants_tools:
                final_text = completion.text
                break

            for call in completion.tool_calls:
                if time.monotonic() > deadline:
                    error = f"stopped after {budget_s:g}s"
                    break
                name, result = await self._handle_tool_call(call, messages, source, trace_id)
                tool_names.append(name)
                tool_results.append((name, result))
                if result.success and result.evidence:
                    provenance.append(Provenance(kind=ProvenanceKind.TOOL_OBSERVATION, source_id=name))
                if result.error and result.error.code in {
                    ToolErrorCode.DENIED, ToolErrorCode.CONFIRMATION_REQUIRED,
                }:
                    # Denial is terminal for this action: do not retry, do not
                    # rephrase, do not look for another route.
                    blocked = True
            if error:
                break

        # A failed model call with nothing else to show must still say *why* it
        # failed. Swallowing the reason behind a generic "I could not find
        # anything to do" turns a one-line config fix (a bad model name, a 404)
        # into an unexplained dead end.
        if error is not None and not final_text and not tool_results:
            final_text = f"I could not reach the model: {error}"

        # Falling out of the loop by exhausting the iteration budget means the
        # model never produced an answer. That is an incomplete turn, not a
        # successful one, and must not be reported as success.
        if error is None and not final_text and iterations >= max_iterations:
            error = (
                f"gave up after {max_iterations} tool iterations without reaching an answer"
            )
            self._audit("turn.iteration_limit", "error", detail={"trace": trace_id})

        narration = self.narrator.render(
            text=final_text,
            tool_results=[r for _, r in tool_results],
            tool_names=tool_names,
            provenance=provenance,
            blocked=blocked,
        )
        state = AgentState.ERROR if error else AgentState.IDLE
        self._set_state(state, source)

        result = TurnResult(
            text=narration.text,
            narration=narration,
            state=state,
            tool_results=tool_results,
            iterations=iterations,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            blocked=blocked,
            error=error,
        )
        self._session.record_turn("user", utterance)
        self._session.record_turn("assistant", narration.text)
        self._audit("turn.end", "error" if error else "success", detail=result.to_dict() | {"trace": trace_id})
        return result

    # -- deterministic shortcut ----------------------------------------
    async def dispatch_deterministic(
        self,
        utterance: str,
        *,
        source: TurnSource = TurnSource.ROUTER,
        cancel: asyncio.Event | None = None,
    ) -> TurnResult | None:
        """Answer a recognised command without touching the model.

        The fast-path router (ARCHITECTURE.md S8) resolves a handful of
        high-frequency, unambiguous commands — open an app, open a URL, the time,
        arithmetic — with a regex table. When it does, this runs the same
        permission check, confirmation flow, audited executor and honest narration
        that a model-driven turn would, but with **zero model calls**. That is the
        whole point: it removes the dominant latency from the common commands, and
        because no model sees the text, the result cannot be prompt-injected.

        Returns:
            A completed :class:`TurnResult` for a deterministic tool action, or
            ``None`` when the request needs the agent loop (conversation,
            ambiguity, or a tool the fast path does not own). Callers fall back to
            :meth:`run` on ``None``.
        """
        outcome = self.router.route(utterance)
        decision = outcome.decision
        if outcome.needs_clarification or not decision.via_fast_path:
            return None
        if not decision.is_tool_call or decision.tool not in self.registry.names:
            return None
        if cancel is not None and cancel.is_set():
            raise TurnCancelled("cancelled by user")

        trace_id = f"t-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        self._session = self._session.evolve(turn_count=self._session.turn_count + 1)
        self._set_state(AgentState.THINKING, source)
        self._audit("turn.start", "info", detail={
            "utterance": utterance[:400], "source": source.value,
            "trace": trace_id, "route": decision.kind, "via": "fast_path",
        })

        name = decision.tool
        assert name is not None
        spec = self.registry.get(name)
        assert spec is not None
        args = dict(decision.arguments)

        result = await self._authorise_and_execute(spec, name, args, source=source, trace_id=trace_id)

        blocked = (
            result.error is not None
            and result.error.code in {ToolErrorCode.DENIED, ToolErrorCode.CONFIRMATION_REQUIRED}
        )
        # The result is authoritative — it is what actually happened — so the text
        # is built from it, never from a model that might claim more than was done.
        text = _deterministic_text(decision, result)
        provenance: list[Provenance] = []
        if result.success and result.evidence:
            provenance.append(Provenance(ProvenanceKind.TOOL_OBSERVATION, source_id=name))
        narration = Narration(
            text=text,
            provenance=provenance,
            blocked=blocked,
            details={"tool_calls": [{
                "tool": name, "success": result.success,
                "summary": result.summary(), "result": result.to_dict(),
            }]},
        )
        state = AgentState.IDLE
        self._set_state(state, source)

        turn = TurnResult(
            text=narration.text,
            narration=narration,
            state=state,
            tool_results=[(name, result)],
            iterations=0,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            blocked=blocked,
        )
        self._session.record_turn("user", utterance)
        self._session.record_turn("assistant", narration.text)
        self._audit("turn.end", "success" if result.success else "error", detail=turn.to_dict() | {"trace": trace_id})
        return turn

    # -- one tool call -------------------------------------------------
    async def _handle_tool_call(
        self,
        call: Any,
        messages: list[Message],
        source: TurnSource,
        trace_id: str,
    ) -> tuple[str, ToolResult]:
        """Authorise, confirm if needed, execute, and feed the observation back."""
        name = call.name
        args = call.parsed
        spec = self.registry.get(name)

        if spec is None:
            result = ToolResult.fail(
                ToolErrorCode.UNKNOWN_TOOL,
                f"There is no tool called {name!r}. Available: {', '.join(self.registry.names)}",
                remedy="Call one of the available tools, or answer without using a tool.",
                data={"available": list(self.registry.names)},
            )
            await self.registry.execute(name, args, None, ToolContext(source=source.value), source=source)
            messages.append(_tool_message(call.id, name, result))
            return name, result

        result = await self._authorise_and_execute(spec, name, args, source=source, trace_id=trace_id)
        messages.append(_tool_message(call.id, name, result))
        return name, result

    async def _authorise_and_execute(
        self,
        spec: Any,
        name: str,
        args: dict[str, Any],
        *,
        source: TurnSource,
        trace_id: str,
    ) -> ToolResult:
        """The single authorisation-and-execution path for one tool call.

        Shared by the model-driven loop and the deterministic shortcut so both are
        governed by exactly the same permission engine, confirmation flow and
        audit trail. The model never authorises anything; a denial is terminal.
        """
        decision = self.permissions.check(
            scope=spec.scope(str(args.get("application") or args.get("url") or "")),
            level=spec.permission_level,
            tool_requires_confirmation=spec.requires_confirmation,
            source=source.value,
            consequence=spec.consequence(args),
        )

        self._audit("permission.check", "info", detail={
            "trace": trace_id, "tool": name, "effect": decision.effect,
            "reason": decision.reason, "level": decision.level, "source": source.value,
        })

        if decision.requires_confirmation:
            outcome = await self.confirmations.request(decision, tool=name, args=args)
            self._audit(
                "confirmation", "success" if outcome.approved else "denied",
                detail={"trace": trace_id, "tool": name, "factor": outcome.factor, "reason": outcome.reason},
            )
            if not outcome.approved:
                result = ToolResult.fail(
                    ToolErrorCode.DENIED,
                    f"You did not confirm: {outcome.reason}",
                    remedy="Ask again if you want me to.",
                )
                await self.registry.execute(name, args, _deny(decision), ToolContext(source=source.value), source=source)
                return result
            decision = self.permissions.check(
                scope=spec.scope(), level=spec.permission_level,
                tool_requires_confirmation=spec.requires_confirmation,
                source=source.value, consequence=spec.consequence(args),
                satisfaction=outcome.factor,
            )

        self._set_state(AgentState.EXECUTING, source, tool=name)
        ctx = ToolContext(
            settings=self.settings, audit=self.audit, bus=self.bus,
            confirmation=self.confirmations, session_id=self._session.session_id,
            source=source.value, scratch=self._scratch,
        )
        result = await self.registry.execute(name, args, decision, ctx, source=source)
        self.bus.publish("tool.result", {"tool": name, "result": result.to_dict()}, source=source.value)
        return result

    # -- plumbing ------------------------------------------------------
    def _set_state(self, state: AgentState, source: TurnSource, *, tool: str | None = None) -> None:
        self._session = self._session.set_state(state)
        self.bus.publish(
            "agent.state",
            {"state": state.value, "tool": tool, "session": self._session.session_id},
            source=source.value,
        )

    def _audit(self, action: str, outcome: str, *, detail: dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(action=action, outcome=outcome, actor="agent", channel="core", detail=detail)
        except Exception:
            log.exception("audit append failed for %s", action)


def _deny(decision: PermissionDecision) -> PermissionDecision:
    return PermissionDecision(
        allowed=False, requires_confirmation=False, level=decision.level,
        effect="deny", reason="user_declined_confirmation", scope=decision.scope,
    )


def _deterministic_text(decision: IntentDecision, result: ToolResult) -> str:
    """Turn a fast-path decision's actual result into something worth saying.

    With no model in the loop, a bare ``ToolResult.spoken()`` for an app launch
    would be a provenance chip ("ok [activity_started] in 30ms"), which is not
    speech. For the intents we recognise, a short, truthful sentence is better. A
    failure always speaks the tool's own message — it is specific and honest.
    """
    if not result.success:
        assert result.error is not None
        return result.error.message

    data = result.data or {}
    if decision.kind == "open_app":
        app = data.get("application") or decision.arguments.get("application")
        if app:
            return f"Opening {app}."
    if decision.kind == "open_url":
        url = data.get("url") or decision.arguments.get("url")
        if url:
            return f"Opening {url}."
    return result.spoken()


def _assistant_message(completion: Completion) -> Message:
    return Message(role="assistant", content=completion.text, tool_calls=completion.tool_calls)


def _tool_message(call_id: str, name: str, result: ToolResult) -> Message:
    """Serialise an observation back to the model.

    The full result is included — including the failure code and remedy — so the
    model can correct itself instead of repeating the same broken call.

    A result flagged ``untrusted_source`` is fenced before it goes into the
    prompt. Phase 6 made this reachable: ``web_fetch`` returns whatever a stranger
    wrote, and unwrapped that text would be a prompt-injection channel with a
    tool call already in flight (SECURITY.md T1).
    """
    import json

    payload = json.dumps(result.to_dict(), default=str)
    if result.untrusted_source:
        from ..llm.prompts import wrap_untrusted

        payload = wrap_untrusted(payload, source=f"{name} result from {result.untrusted_source}")
    return Message(role="tool", name=name, tool_call_id=call_id, content=payload)


def _routing_temperature(provider: LLMProvider, source: TurnSource) -> float:
    """Deterministic intent routing; freer prose.

    A tool *selection* should be reproducible so the same request does not
    randomly open a different app. Final wording benefits from variety.
    """
    return 0.0 if source in {TurnSource.ROUTER, TurnSource.GESTURE} else 0.3
