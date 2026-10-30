"""Tool contracts.

Every capability Prithivi has is a tool: opening Instagram, playing music,
searching the web, sending a message. Tools are **dumb executors plus permission
metadata** (ARCHITECTURE.md S9) — they do not reason, and they never call the
agent back.

Three rules give the system its safety properties:

1. **No code generation.** There is no ``eval``/``exec`` and no shell-passthrough
   tool. ``calculate`` uses a hand-written recursive-descent parser instead.
2. **No fake success.** A tool returns :class:`ToolResult` with ``evidence``.
   The narrator is forbidden from claiming success without it.
3. **Degrade honestly.** If a backend is missing (no browser, no TTS engine),
   the result is ``BACKEND_UNAVAILABLE`` with a reason — never a silent stub.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..core.schema import Model, ValidationError
from ..security.permissions import PermissionDecision

__all__ = [
    "BACKEND_UNAVAILABLE",
    "Evidence",
    "Executor",
    "SideEffect",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolResult",
    "ToolSpec",
    "fail",
    "ok",
    "tool",
    "unavailable",
]

TIn = TypeVar("TIn", bound=Model)
TOut = TypeVar("TOut", bound=Model)


class SideEffect(str):
    NONE = "none"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    EXTERNAL = "external"  # visible to a third party


class ToolErrorCode(str):
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    VALIDATION = "VALIDATION_ERROR"
    DENIED = "DENIED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    CONFIRMATION_DECLINED = "CONFIRMATION_DECLINED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    PERMISSION = "PERMISSION"
    NETWORK = "NETWORK"
    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ToolError:
    """A typed failure the agent can reason about.

    ``remedy`` exists so the narrator can suggest the next step rather than
    simply apologising.
    """

    code: str
    message: str
    retryable: bool = False
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable, "remedy": self.remedy}


@dataclass(frozen=True)
class Evidence:
    """Proof the action actually happened.

    The narrator may only claim success when this is present (ARCHITECTURE.md
    rule 2). ``kind`` examples: ``pid``, ``http_status``, ``path_exists``,
    ``media_state``, ``row_count``.
    """

    kind: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.detail}


@dataclass
class ToolResult:
    """The only thing a tool returns. No exceptions escape an executor."""

    success: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    evidence: Evidence | None = None
    duration_ms: int = 0
    untrusted_source: str | None = None
    """Set when ``data`` contains text from outside the user's machine — a fetched
    page, a search snippet, a document. The agent fences it as untrusted data
    before it reaches the model (SECURITY.md T1); leaving it unset would put
    attacker-controlled instructions straight into a system prompt."""

    @classmethod
    def ok(
        cls,
        data: dict[str, Any] | None = None,
        *,
        evidence: Evidence | None = None,
        untrusted_source: str | None = None,
    ) -> ToolResult:
        """Success. Callers should pass ``evidence`` whenever it is obtainable."""
        return cls(success=True, data=data or {}, evidence=evidence, untrusted_source=untrusted_source)

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        remedy: str = "",
        data: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(success=False, data=data, error=ToolError(code=code, message=message, retryable=retryable, remedy=remedy))

    @classmethod
    def unavailable(cls, backend: str, *, remedy: str = "") -> ToolResult:
        return cls(
            success=False,
            error=ToolError(
                code=ToolErrorCode.BACKEND_UNAVAILABLE,
                message=f"{backend} is not available in this environment",
                remedy=remedy,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "duration_ms": self.duration_ms,
            "untrusted_source": self.untrusted_source,
        }

    def summary(self) -> str:
        """One-line form for logs and transcripts."""
        if self.success:
            kind = f" [{self.evidence.kind}]" if self.evidence else ""
            return f"ok{kind} in {self.duration_ms}ms"
        assert self.error is not None
        return f"error {self.error.code}: {self.error.message}"

    #: ``data`` keys, in preference order, that a caller can show verbatim. Used
    #: by :meth:`spoken` so the fast path can answer without a second model call.
    DISPLAY_KEYS: tuple[str, ...] = (
        "result", "value", "answer", "text", "content", "summary",
        "time", "iso", "path", "title",
    )

    def spoken(self) -> str:
        """The payload as a phrase fit to say out loud.

        Falls back to :meth:`summary` when there is nothing meaningful to render,
        so this never returns an empty string.
        """
        if not self.success:
            assert self.error is not None
            return self.error.message

        data = self.data or {}
        for key in self.DISPLAY_KEYS:
            if key in data and data[key] not in (None, "", {}):
                value = data[key]
                if key == "result" and "expression" in data:
                    return f"{data['expression']} = {value}"
                return str(value)
        return self.summary()


def ok(data: dict[str, Any] | None = None, *, evidence: Evidence | None = None) -> ToolResult:
    return ToolResult.ok(data, evidence=evidence)


def fail(code: str, message: str, **kwargs: Any) -> ToolResult:
    return ToolResult.fail(code, message, **kwargs)


def unavailable(backend: str, *, remedy: str = "") -> ToolResult:
    return ToolResult.unavailable(backend, remedy=remedy)


BACKEND_UNAVAILABLE = ToolErrorCode.BACKEND_UNAVAILABLE


@dataclass
class ToolContext:
    """What an executor is given.

    Tools receive services, never the agent. Passing ``ToolContext`` as a
    keyword-only parameter is how :func:`tool` finds it.

    Attributes:
        settings: runtime configuration.
        audit: append-only log; the registry writes tool.call automatically, so
            a tool only needs this for extra detail.
        bus: event bus for progress events.
        confirmation: available if a tool needs extra mid-flight consent.
        session_id: for correlation.
    """

    settings: Any = None
    audit: Any = None
    bus: Any = None
    confirmation: Any = None
    session_id: str = "default"
    source: str = "agent"
    scratch: dict[str, Any] = field(default_factory=dict)
    memory: Any = None
    """Long-term memory service (:class:`prithivi.memory.MemoryStore`), when enabled."""
    knowledge: Any = None
    """The user's document knowledge base (:class:`prithivi.rag.KnowledgeBase`)."""

    def progress(self, summary: str) -> None:
        """Emit a ``tool.activity`` event for the UI."""
        if self.bus is not None:
            self.bus.publish("tool.activity", {"summary": summary}, source=self.source)


Executor = Callable[..., Awaitable[ToolResult]]


@dataclass
class ToolSpec(Generic[TIn]):
    """Everything the system knows about one tool.

    Attributes:
        name: snake_case, globally unique, ``[a-z0-9_]{1,64}``.
        description: written for the model. Must state when **not** to use it.
        input_model: a :class:`~prithivi.core.schema.Model` subclass; its
            JSON Schema is what the model sees and what validates the arguments.
        permission_level: L0..L4 (SECURITY.md S1).
        side_effects: ``none`` / ``reversible`` / ``irreversible`` / ``external``.
        requires_confirmation: forces a confirm even below L2.
        idempotent: safe to retry.
        timeout_s: enforced with ``asyncio.wait_for``.
        available: optional probe; when it returns a reason, calls short-circuit
            to ``BACKEND_UNAVAILABLE`` without running the executor.
    """

    name: str
    description: str
    input_model: type[Model]
    executor: Executor
    version: str = "1"
    output_model: type[Model] | None = None
    permission_level: int = 0
    side_effects: str = SideEffect.NONE
    requires_confirmation: bool = False
    idempotent: bool = False
    timeout_s: float = 20.0
    availability_probe: Callable[[ToolContext], str | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import re
        if not re.fullmatch(r"[a-z0-9_]{1,64}", self.name):
            raise ValueError(f"tool name {self.name!r} must match [a-z0-9_]{{1,64}}")
        if not 0 <= self.permission_level <= 4:
            raise ValueError(f"{self.name}: permission_level must be 0..4")
        if not self.description.strip():
            raise ValueError(f"{self.name}: description is required (the model reads it)")
        try:
            if not issubclass(self.input_model, Model):
                raise ValueError(f"{self.name}: input_model must be a prithivi.core.schema.Model subclass")
        except TypeError as exc:
            if "issubclass()" in str(exc):
                raise ValueError(f"{self.name}: input_model must be a prithivi.core.schema.Model subclass") from exc
            raise

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.input_model.json_schema()},
        }

    def scope(self, detail: str = "") -> str:
        """The string policies glob against."""
        return f"{self.name}:{detail}" if detail else self.name

    def consequence(self, args: Any) -> str:
        """Human-readable summary of what running this would do.

        Shown *before* confirmation (SECURITY.md S3). Tools may override this
        when the default is not informative enough.
        """
        try:
            rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
        except AttributeError:
            rendered = repr(args)
        return f"Run {self.name}({rendered}) [{self.side_effects}]"


def tool(
    *,
    name: str,
    description: str,
    input_model: type[Model],
    permission_level: int = 0,
    side_effects: str = SideEffect.NONE,
    requires_confirmation: bool = False,
    idempotent: bool = False,
    timeout_s: float = 20.0,
    output_model: type[Model] | None = None,
    availability_probe: Callable[[ToolContext], str | None] | None = None,
) -> Callable[[Executor], ToolSpec]:
    """Decorator that turns an async function into a :class:`ToolSpec`.

    The wrapped function receives validated ``args`` as keyword arguments plus
    ``ctx: ToolContext``. It must return a :class:`ToolResult`.

    Example:
        @tool(name="get_time", description="...", input_model=GetTimeArgs,
              permission_level=0, idempotent=True)
        async def _get_time(ctx: ToolContext) -> ToolResult: ...
    """

    def decorate(fn: Executor) -> ToolSpec:
        if not inspect.iscoroutinefunction(fn):
            raise ValueError(f"{name}: tool executors must be async def")
        return ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            executor=fn,
            permission_level=permission_level,
            side_effects=side_effects,
            requires_confirmation=requires_confirmation,
            idempotent=idempotent,
            timeout_s=timeout_s,
            output_model=output_model,
            availability_probe=availability_probe,
            metadata={"qualname": fn.__qualname__, "module": fn.__module__},
        )

    return decorate


async def run_executor(
    spec: ToolSpec,
    args: Model,
    ctx: ToolContext,
) -> ToolResult:
    """Validate availability, enforce the timeout, and normalise exceptions.

    Guarantees the caller gets a :class:`ToolResult` and never a traceback, which
    is what lets the agent continue instead of dying on one bad tool.
    """
    started = time.monotonic()

    if spec.availability_probe is not None:
        try:
            reason = spec.availability_probe(ctx)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult.fail(ToolErrorCode.INTERNAL, f"availability probe failed: {exc}")
            result.duration_ms = int((time.monotonic() - started) * 1000)
            return result
        if reason:
            result = ToolResult.unavailable(spec.name + f" ({reason})", remedy="Run `prithivi doctor` to see every environment gap.")
            result.duration_ms = int((time.monotonic() - started) * 1000)
            return result

    payload = args.to_dict()
    try:
        result = await asyncio.wait_for(spec.executor(ctx=ctx, **payload), timeout=spec.timeout_s)
    except TimeoutError:
        result = ToolResult.fail(
            ToolErrorCode.TIMEOUT,
            f"{spec.name} did not finish within {spec.timeout_s:g}s",
            retryable=True,
            remedy="Raise the timeout or try a smaller task.",
        )
    except asyncio.CancelledError:
        raise
    except TypeError as exc:
        # A mismatch between input_model and the executor's signature is a bug
        # in the tool, not bad user input. Say so precisely.
        result = ToolResult.fail(ToolErrorCode.INTERNAL, f"{spec.name} has an invalid signature: {exc}")
    except Exception as exc:  # noqa: BLE001
        result = ToolResult.fail(
            ToolErrorCode.INTERNAL, f"{spec.name} raised {type(exc).__name__}: {exc}",
            retryable=False, remedy="This is a bug in Prithivi; the audit log has the trace.",
        )

    if not isinstance(result, ToolResult):
        result = ToolResult.fail(
            ToolErrorCode.INTERNAL,
            f"{spec.name} returned {type(result).__name__}, expected ToolResult",
        )
    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


def validate_args(spec: ToolSpec, arguments: object) -> tuple[Model | None, ToolResult | None]:
    """Validate raw arguments, returning either the model or a failure result."""
    try:
        return spec.input_model.from_dict(arguments), None
    except ValidationError as exc:
        return None, ToolResult.fail(
            ToolErrorCode.VALIDATION,
            str(exc),
            remedy=f"Call {spec.name} again with arguments matching its schema.",
            data={"schema": spec.input_model.json_schema(), "fields": exc.to_dict()["fields"]},
        )


def decision_blocks(decision: PermissionDecision | None) -> ToolResult | None:
    """Return a failure result if ``decision`` forbids execution.

    SECURITY.md invariant 1: no executor runs without a decision. A missing
    decision is treated exactly like a denial — never as an implicit allow.
    """
    if decision is None:
        return ToolResult.fail(
            ToolErrorCode.DENIED,
            "tool executed without a permission decision; refusing",
            remedy="This is a bug in Prithivi's enforcement chain.",
        )
    if decision.denied:
        return ToolResult.fail(ToolErrorCode.DENIED, f"Denied: {decision.reason}", remedy="Ask the user what they would like instead.")
    if decision.requires_confirmation:
        return ToolResult.fail(
            ToolErrorCode.CONFIRMATION_REQUIRED,
            decision.consequence or f"{spec_consequence(decision)} needs confirmation",
            data={"level": decision.level, "consequence": decision.consequence},
        )
    return None


def spec_consequence(decision: PermissionDecision) -> str:
    return decision.scope or "this action"
