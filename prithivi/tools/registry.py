"""Allowlist tool registry and the single execution path.

``ToolRegistry.execute`` is the only way to run a tool, and it requires a
:class:`~prithivi.security.permissions.PermissionDecision` as a keyword-only
argument. Passing ``None`` is treated as a denial, so the enforcement chain
cannot be skipped by omitting a flag (SECURITY.md invariant 1).

Registration is explicit. There is no filesystem scanning for tools, because a
directory that can execute code is a privilege escalation waiting to happen.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from ..core.state import TurnSource
from ..llm.provider import ToolSchema
from ..security.audit import AuditLog
from ..security.permissions import PermissionDecision, PermissionLevel
from .base import (
    ToolContext,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
    decision_blocks,
    run_executor,
    validate_args,
)

__all__ = ["ToolRegistry"]

log = logging.getLogger("prithivi.tools")


class ToolRegistry:
    """Holds the allowlist of tools and executes them with authorisation."""

    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._audit = audit

    # -- registration --------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def register_all(self, specs: Iterable[ToolSpec], *, replace: bool = False) -> list[ToolSpec]:
        return [self.register(s, replace=replace) for s in specs]

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # -- lookup --------------------------------------------------------
    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def schemas(self) -> list[ToolSchema]:
        """Model-facing tool definitions, sorted for deterministic prompts."""
        return [
            ToolSchema(
                name=s.name,
                description=s.description,
                parameters=s.input_model.json_schema(),
            )
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    def summary(self) -> list[dict[str, Any]]:
        """Human-readable inventory for ``prithivi tools``."""
        return [
            {
                "name": s.name,
                "level": s.permission_level,
                "level_name": PermissionLevel.name_of(s.permission_level),
                "side_effects": s.side_effects,
                "confirm": s.requires_confirmation,
                "timeout_s": s.timeout_s,
                "description": s.description.splitlines()[0][:100],
            }
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    # -- execution -----------------------------------------------------
    async def execute(
        self,
        name: str,
        arguments: object,
        decision: PermissionDecision | None,
        ctx: ToolContext,
        *,
        source: TurnSource | str = TurnSource.AGENT,
        extra_detail: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Authorise, validate, run, and audit one tool call.

        Args:
            name: tool name as produced by the router/LLM.
            arguments: raw JSON mapping.
            decision: the permission verdict. **Required** — ``None`` is a denial.
            ctx: services handed to the executor.

        Returns:
            A :class:`ToolResult`. Never raises for tool-level problems; an
            unknown name is reported as ``UNKNOWN_TOOL`` so the agent can correct
            itself (SECURITY.md matrix).
        """
        source_value = source.value if isinstance(source, TurnSource) else str(source)

        spec = self._tools.get(name)
        if spec is None:
            result = ToolResult.fail(
                ToolErrorCode.UNKNOWN_TOOL,
                f"No tool named {name!r}. Available tools: {', '.join(self.names) or 'none'}",
                remedy="Call one of the listed tools, or answer without a tool.",
                data={"available": list(self.names)},
            )
            self._log_call(source_value, name, arguments, result, None, extra_detail)
            return result

        # 1. permission, before any argument handling: an unauthorised call must
        #    not even be parsed into a tool.
        blocked = decision_blocks(decision)
        if blocked is not None:
            blocked.duration_ms = 0
            self._log_call(source_value, name, arguments, blocked, spec, extra_detail, decision=decision)
            return blocked

        # 2. validate arguments against the tool's own schema.
        args, invalid = validate_args(spec, arguments)
        if invalid is not None:
            self._log_call(source_value, name, arguments, invalid, spec, extra_detail, decision=decision)
            return invalid

        assert args is not None
        if ctx is None:
            ctx = ToolContext()
        ctx.progress(f"Running {name}")
        result = await run_executor(spec, args, ctx)

        detail = dict(extra_detail or {})
        detail["side_effects"] = spec.side_effects
        self._log_call(source_value, name, args.to_dict(), result, spec, detail, decision=decision)
        return result

    # -- audit ---------------------------------------------------------
    def _log_call(
        self,
        source: str,
        name: str,
        arguments: object,
        result: ToolResult,
        spec: ToolSpec | None,
        extra: dict[str, Any] | None,
        *,
        decision: PermissionDecision | None = None,
    ) -> None:
        if self._audit is None:
            return
        outcome = {
            ToolErrorCode.DENIED: "denied",
            ToolErrorCode.CONFIRMATION_REQUIRED: "denied",
            ToolErrorCode.TIMEOUT: "timeout",
            ToolErrorCode.UNKNOWN_TOOL: "error",
        }.get(result.error.code if result.error else "", "success" if result.success else "error")
        detail: dict[str, Any] = {
            "args": arguments,
            "result": result.to_dict(),
            "source": source,
        }
        if decision is not None:
            detail["decision"] = {"effect": decision.effect, "reason": decision.reason, "level": decision.level}
        if extra:
            detail.update(extra)
        try:
            self._audit.append(
                action="tool.execute",
                outcome=outcome,
                actor="agent" if source in {"agent", "router"} else source,
                channel=source,
                target=name,
                detail=detail,
            )
        except Exception:
            log.exception("failed to audit tool call %s", name)

    # -- diagnostics ---------------------------------------------------
    def readiness(self) -> list[tuple[str, bool, str]]:
        """Probe every tool's availability for ``prithivi doctor``.

        A probe returns ``None`` when the tool is usable, or a human-readable
        reason (or a sequence of reasons) when it is not.

        Returns:
            ``(name, available, reason)`` per tool, in name order, where
            ``reason`` is always a ``str`` — a probe that returns a tuple of
            notes is joined here rather than leaking a non-renderable object to
            the UI.
        """
        ctx = ToolContext()
        out: list[tuple[str, bool, str]] = []
        for spec in sorted(self._tools.values(), key=lambda s: s.name):
            if spec.availability_probe is None:
                out.append((spec.name, True, ""))
                continue
            try:
                result = spec.availability_probe(ctx)
            except Exception as exc:  # noqa: BLE001
                out.append((spec.name, False, f"probe raised {type(exc).__name__}: {exc}"))
                continue
            if result is None:
                out.append((spec.name, True, ""))
            elif isinstance(result, str):
                out.append((spec.name, False, result))
            elif isinstance(result, Sequence):
                out.append((spec.name, False, "; ".join(str(item) for item in result)))
            else:
                out.append((spec.name, False, str(result)))
        return out

    def apply(self, other: ToolRegistry) -> None:
        """Merge another registry's tools into this one."""
        for spec in other:
            self.register(spec, replace=True)
