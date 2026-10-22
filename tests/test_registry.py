"""Tool registry and tool contracts.

TESTING.md §4: registry rejection, schema generation, and the tool contract in
general. The registry is the single execution path, so most of what matters here
is about what it *refuses*: unknown names, duplicate registrations, unauthorised
calls, and bad arguments.
"""

from __future__ import annotations

import asyncio

import pytest

from prithivi.core.schema import Model, ValidationError
from prithivi.security.audit import AuditLog
from prithivi.security.permissions import (
    Effect,
    PermissionDecision,
    PermissionEngine,
    PermissionLevel,
)
from prithivi.tools.base import (
    Evidence,
    SideEffect,
    ToolContext,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
    decision_blocks,
    run_executor,
    tool,
    validate_args,
)
from prithivi.tools.basic import build_registry
from prithivi.tools.registry import ToolRegistry


class Args(Model):
    """Tool arguments model."""

    target: str
    count: int = 1


def allow(level: int = 0) -> PermissionDecision:
    return PermissionDecision(
        allowed=True, requires_confirmation=False, level=level, effect=Effect.ALLOW,
        reason="test", scope="t",
    )


async def _echo(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
    return ToolResult.ok({"target": target, "count": count})


def make_tool(**overrides) -> ToolSpec:
    """A valid L0 tool, with any field overridable per test."""
    defaults: dict = dict(
        name="sample",
        description="A test tool.",
        input_model=Args,
        executor=_echo,
        permission_level=0,
        side_effects=SideEffect.NONE,
        idempotent=True,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


# --- registration -----------------------------------------------------------


def test_registration_is_explicit_and_duplicate_names_are_refused() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool())
    # …unless replacement is asked for on purpose.
    registry.register(make_tool(), replace=True)
    assert len(registry) == 1


def test_tools_can_be_unregistered_and_looked_up() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    assert "sample" in registry
    assert registry.get("sample") is not None
    assert registry.get("absent") is None
    assert registry.unregister("sample") is True
    assert registry.unregister("sample") is False
    assert "sample" not in registry


def test_names_are_sorted_for_deterministic_prompts() -> None:
    registry = ToolRegistry()
    registry.register_all([
        make_tool(name="zebra"), make_tool(name="alpha"), make_tool(name="middle"),
    ])
    assert registry.names == ("alpha", "middle", "zebra")


def test_registries_can_be_merged() -> None:
    first = ToolRegistry()
    first.register(make_tool(name="a"))
    second = ToolRegistry()
    second.register(make_tool(name="b"))

    first.apply(second)
    assert first.names == ("a", "b")


# --- schema generation ------------------------------------------------------


@pytest.mark.parametrize("name", ["UPPER", "with-dash", "with space", "", "x" * 65, "ünïcode"])
def test_tool_names_are_constrained(name: str) -> None:
    """A name the model cannot reproduce verbatim is a name that will misfire."""
    with pytest.raises(ValueError, match=r"\[a-z0-9_\]\{1,64\}"):
        make_tool(name=name)


def test_the_permission_level_must_be_in_range() -> None:
    for level in (-1, 5):
        with pytest.raises(ValueError, match="0..4"):
            make_tool(permission_level=level)


def test_a_tool_must_describe_itself() -> None:
    """The description is what the model reads; a blank one is a silent tool."""
    with pytest.raises(ValueError, match="description is required"):
        make_tool(description="   ")


def test_the_input_model_must_be_a_prithivi_model() -> None:
    class NotAModel:
        pass

    with pytest.raises(ValueError, match="input_model"):
        ToolSpec(
            name="x", description="d", input_model=NotAModel,  # type: ignore[arg-type]
            executor=lambda **kw: None,  # type: ignore[arg-type,return-value]
        )


def test_generated_schemas_are_well_formed() -> None:
    """Every registered tool must expose valid JSON Schema the model can read."""
    for schema in build_registry():
        payload = schema.input_model.json_schema()
        assert payload["type"] == "object"
        assert payload["properties"], f"{schema.name} has no parameters"
        for name, field in payload["properties"].items():
            assert "type" in field, f"{schema.name}.{name} has no type"
        assert payload.get("required") is not None


def test_the_registry_sorts_its_schemas() -> None:
    names = [s.name for s in ToolRegistry().schemas()]
    assert names == sorted(names)
    assert ToolRegistry().schemas() == []


def test_a_registry_summary_reports_level_and_side_effects() -> None:
    registry = ToolRegistry()
    registry.register_all(build_registry())
    by_name = {row["name"]: row for row in registry.summary()}
    assert by_name["get_time"]["level"] == PermissionLevel.INFORMATION
    assert by_name["open_url"]["side_effects"] == SideEffect.REVERSIBLE
    assert by_name["open_url"]["level_name"] == PermissionLevel.name_of(1)
    assert by_name["calculate"]["confirm"] is False


def test_a_tool_scope_names_the_tool_and_the_detail() -> None:
    spec = make_tool()
    assert spec.scope() == "sample"
    assert spec.scope("chrome") == "sample:chrome"


def test_the_consequence_is_shown_before_confirmation() -> None:
    consequence = make_tool(name="delete", side_effects=SideEffect.IRREVERSIBLE).consequence(
        {"path": "/tmp/x"}
    )
    assert "delete" in consequence
    assert "/tmp/x" in consequence
    assert "irreversible" in consequence


# --- the tool decorator -----------------------------------------------------


def test_the_decorator_produces_a_spec() -> None:
    @tool(name="decorated", description="d", input_model=Args, permission_level=1)
    async def _executor(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
        return ToolResult.ok({"target": target})

    assert _executor.name == "decorated"
    assert _executor.permission_level == 1
    assert _executor.metadata["module"].startswith("tests")


def test_an_executor_must_be_async() -> None:
    """A sync executor would return a coroutine-or-value ambiguity to callers."""
    with pytest.raises(ValueError, match="must be async"):
        tool(name="sync", description="d", input_model=Args)(
            lambda **kw: None  # type: ignore[arg-type,return-value]
        )


# --- argument validation ----------------------------------------------------


def test_arguments_are_validated_against_the_tool_schema() -> None:
    spec = make_tool()
    args, invalid = validate_args(spec, {"target": "x", "count": 3})
    assert invalid is None
    assert args is not None and args.to_dict() == {"target": "x", "count": 3}


def test_a_missing_required_argument_is_reported_not_raised() -> None:
    spec = make_tool()
    args, invalid = validate_args(spec, {})
    assert args is None
    assert invalid is not None
    assert invalid.error.code == ToolErrorCode.VALIDATION
    assert "target" in invalid.error.message
    # The model needs the schema back so it can retry correctly.
    assert invalid.data["schema"]["type"] == "object"


def test_a_wrongly_typed_argument_is_reported() -> None:
    spec = make_tool()
    _, invalid = validate_args(spec, {"target": "x", "count": "lots"})
    assert invalid is not None
    assert invalid.error.code == ToolErrorCode.VALIDATION


def test_the_schema_model_raises_on_bad_data() -> None:
    with pytest.raises(ValidationError):
        Args.from_dict({"count": 1})


# --- the enforcement gate ---------------------------------------------------


@pytest.mark.security
def test_no_decision_means_no_execution() -> None:
    """SECURITY.md invariant 1: a missing decision is a denial, never an allow."""
    result = decision_blocks(None)
    assert result is not None
    assert result.error.code == ToolErrorCode.DENIED


def test_a_denial_blocks() -> None:
    denied = PermissionDecision(
        allowed=False, requires_confirmation=False, level=3, effect=Effect.DENY, reason="no"
    )
    assert decision_blocks(denied) is not None


def test_an_unconfirmed_confirmation_requirement_blocks() -> None:
    pending = PermissionDecision(
        allowed=False, requires_confirmation=True, level=2, effect=Effect.CONFIRM,
        reason="ask", consequence="Delete 3 files",
    )
    result = decision_blocks(pending)
    assert result is not None
    assert result.error.code == ToolErrorCode.CONFIRMATION_REQUIRED
    assert "Delete 3 files" in result.error.message


def test_a_granted_decision_does_not_block() -> None:
    assert decision_blocks(allow()) is None


@pytest.mark.security
async def test_a_denied_call_never_reaches_the_executor() -> None:
    ran: list[str] = []

    async def executor(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
        ran.append(target)
        return ToolResult.ok({})

    registry = ToolRegistry()
    registry.register(make_tool(executor=executor))
    denied = PermissionDecision(
        allowed=False, requires_confirmation=False, level=3, effect=Effect.DENY, reason="no"
    )
    result = await registry.execute("sample", {"target": "x"}, denied, ToolContext())
    assert result.success is False
    assert ran == []


@pytest.mark.security
async def test_permission_is_checked_before_arguments_are_parsed() -> None:
    """An unauthorised call must not even be decoded into a tool.

    Otherwise a malformed payload from a blocked scope would still exercise the
    validator, and the error returned would be about the arguments rather than
    about the refusal.
    """
    registry = ToolRegistry()
    registry.register(make_tool())
    denied = PermissionDecision(
        allowed=False, requires_confirmation=False, level=3, effect=Effect.DENY, reason="no"
    )
    result = await registry.execute("sample", {"totally": "wrong"}, denied, ToolContext())
    assert result.error.code == ToolErrorCode.DENIED


async def test_an_unknown_tool_names_the_alternatives() -> None:
    """The model can only self-correct if it is told what it may call."""
    registry = ToolRegistry()
    registry.register_all(build_registry())
    result = await registry.execute("send_money", {}, allow(), ToolContext())
    assert result.success is False
    assert result.error.code == ToolErrorCode.UNKNOWN_TOOL
    assert "send_money" in result.error.message
    assert "calculate" in result.error.message
    assert set(result.data["available"]) == set(registry.names)


async def test_invalid_arguments_never_reach_the_executor() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    result = await registry.execute("sample", {}, allow(), ToolContext())
    assert result.error.code == ToolErrorCode.VALIDATION


async def test_a_valid_call_runs_and_returns_its_result() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    result = await registry.execute(
        "sample", {"target": "abc", "count": 2}, allow(), ToolContext()
    )
    assert result.success is True
    assert result.data == {"target": "abc", "count": 2}


# --- executor normalisation -------------------------------------------------


async def test_a_timeout_is_a_typed_result_not_a_hang() -> None:
    async def slow(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult.ok({})

    result = await run_executor(
        make_tool(executor=slow, timeout_s=0.05), Args(target="x"), ToolContext()
    )
    assert result.success is False
    assert result.error.code == ToolErrorCode.TIMEOUT
    assert result.error.retryable is True


async def test_an_exception_becomes_a_typed_result() -> None:
    async def boom(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
        raise RuntimeError("kaboom")

    result = await run_executor(make_tool(executor=boom), Args(target="x"), ToolContext())
    assert result.success is False
    assert result.error.code == ToolErrorCode.INTERNAL
    assert "RuntimeError" in result.error.message


async def test_a_non_toolresult_return_is_rejected() -> None:
    async def wrong(ctx: ToolContext, target: str, count: int = 1):
        return {"target": target}

    result = await run_executor(make_tool(executor=wrong), Args(target="x"), ToolContext())
    assert result.success is False
    assert result.error.code == ToolErrorCode.INTERNAL
    assert "expected ToolResult" in result.error.message


async def test_an_unavailable_backend_short_circuits_before_the_executor() -> None:
    """Rule 3: degrade honestly, never run a stub."""
    ran: list[str] = []

    async def executor(ctx: ToolContext, target: str, count: int = 1) -> ToolResult:
        ran.append(target)
        return ToolResult.ok({})

    spec = make_tool(
        executor=executor, availability_probe=lambda ctx: "no browser installed",
    )
    result = await run_executor(spec, Args(target="x"), ToolContext())
    assert result.success is False
    assert result.error.code == ToolErrorCode.BACKEND_UNAVAILABLE
    assert "no browser installed" in result.error.message
    assert ran == []


async def test_a_raising_probe_is_reported_not_ignored() -> None:
    def probe(ctx: ToolContext) -> str | None:
        raise OSError("device busy")

    result = await run_executor(
        make_tool(availability_probe=probe), Args(target="x"), ToolContext()
    )
    assert result.error.code == ToolErrorCode.INTERNAL
    assert "device busy" in result.error.message


async def test_a_progress_event_is_published_while_a_tool_runs() -> None:
    """The UI needs to know which tool is running, and from which surface."""
    published: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload, source=None):
            published.append((topic, payload))

    registry = ToolRegistry()
    registry.register(make_tool())
    await registry.execute(
        "sample", {"target": "x"}, allow(), ToolContext(bus=Bus(), source="voice")
    )
    assert ("tool.activity", {"summary": "Running sample"}) in published


async def test_a_progress_event_is_not_published_for_a_blocked_call() -> None:
    """Nothing runs, so nothing may claim to be running."""
    published: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload, source=None):
            published.append((topic, payload))

    registry = ToolRegistry()
    registry.register(make_tool())
    denied = PermissionDecision(
        allowed=False, requires_confirmation=False, level=3, effect=Effect.DENY, reason="no"
    )
    await registry.execute("sample", {"target": "x"}, denied, ToolContext(bus=Bus()))
    assert published == []


# --- readiness --------------------------------------------------------------


def test_readiness_reports_availability_for_doctor() -> None:
    registry = ToolRegistry()
    registry.register_all(build_registry())
    rows = registry.readiness()
    assert [name for name, _, _ in rows] == sorted(registry.names)
    for _, available, reason in rows:
        assert isinstance(available, bool)
        assert isinstance(reason, str)


def test_a_probe_may_return_several_notes() -> None:
    """A tuple of reasons must be joined, not leaked as a raw object."""
    registry = ToolRegistry()
    registry.register(make_tool(availability_probe=lambda ctx: ("a", "b")))
    _, available, reason = registry.readiness()[0]
    assert available is False
    assert reason == "a; b"


def test_a_raising_probe_is_reported_as_unavailable() -> None:
    def probe(ctx: ToolContext) -> str | None:
        raise RuntimeError("nope")

    registry = ToolRegistry()
    registry.register(make_tool(availability_probe=probe))
    _, available, reason = registry.readiness()[0]
    assert available is False
    assert "RuntimeError" in reason


def test_a_tool_with_no_probe_is_always_available() -> None:
    registry = ToolRegistry()
    registry.register(make_tool(availability_probe=None))
    assert registry.readiness() == [("sample", True, "")]


# --- result shape -----------------------------------------------------------


def test_a_successful_result_carries_its_evidence() -> None:
    """Rule 2: the narrator may only claim success with evidence."""
    result = ToolResult.ok({"n": 1}, evidence=Evidence("pid", {"pid": 42}))
    assert result.to_dict()["evidence"] == {"kind": "pid", "pid": 42}
    assert "pid" in result.summary()


def test_a_result_without_evidence_is_still_describable() -> None:
    assert ToolResult.ok({"n": 1}).summary().startswith("ok")


def test_spoken_prefers_the_human_readable_field() -> None:
    assert ToolResult.ok({"result": 42, "as_float": 42.0}).spoken() == "42"
    assert ToolResult.ok({"expression": "2+2", "result": 4}).spoken() == "2+2 = 4"
    assert ToolResult.ok({"time": "12:00"}).spoken() == "12:00"
    assert ToolResult.ok({"answer": "42"}).spoken() == "42"


def test_spoken_never_returns_nothing() -> None:
    assert ToolResult.ok({}).spoken() != ""
    assert ToolResult.ok({"result": None}).spoken() != ""
    failed = ToolResult.fail(ToolErrorCode.INTERNAL, "it broke")
    assert failed.spoken() == "it broke"


def test_an_unavailable_result_suggests_a_remedy() -> None:
    result = ToolResult.unavailable("Playwright", remedy="pip install playwright")
    assert result.error.code == ToolErrorCode.BACKEND_UNAVAILABLE
    assert "Playwright" in result.error.message
    assert result.error.remedy == "pip install playwright"


# --- auditing ---------------------------------------------------------------


async def test_every_execution_outcome_is_audited() -> None:
    with AuditLog(":memory:") as audit:
        registry = ToolRegistry(audit=audit)
        registry.register_all(build_registry())

        await registry.execute("calculate", {"expression": "1+1"}, allow(), ToolContext())
        await registry.execute("calculate", {"expression": "bad("}, allow(), ToolContext())
        await registry.execute("nonexistent", {}, allow(), ToolContext())
        denied = PermissionDecision(
            allowed=False, requires_confirmation=False, level=3, effect=Effect.DENY, reason="no"
        )
        await registry.execute("open_url", {"url": "https://x.example"}, denied, ToolContext())

        outcomes = [r.outcome for r in audit.all()]
        assert outcomes.count("success") == 1
        assert outcomes.count("error") == 2
        assert outcomes.count("denied") == 1
        assert audit.verify() == (True, None)


async def test_the_audit_records_the_decision_that_was_used() -> None:
    with AuditLog(":memory:") as audit:
        registry = ToolRegistry(audit=audit)
        registry.register(make_tool())
        await registry.execute("sample", {"target": "x"}, allow(level=1), ToolContext())

        detail = audit.all()[0].detail
        assert detail["decision"]["effect"] == Effect.ALLOW
        assert detail["decision"]["level"] == 1


async def test_the_audit_records_the_originating_surface() -> None:
    """A gesture-originated call must be distinguishable in the audit trail."""
    with AuditLog(":memory:") as audit:
        registry = ToolRegistry(audit=audit)
        registry.register(make_tool())
        await registry.execute(
            "sample", {"target": "x"}, allow(), ToolContext(), source="gesture"
        )
        assert audit.all()[0].detail["source"] == "gesture"
        assert audit.all()[0].channel == "gesture"


async def test_a_failing_audit_never_breaks_execution() -> None:
    class BrokenAudit:
        def append(self, **kwargs):
            raise RuntimeError("disk full")

    registry = ToolRegistry(audit=BrokenAudit())
    registry.register(make_tool())
    result = await registry.execute("sample", {"target": "x"}, allow(), ToolContext())
    assert result.success is True, "a broken audit log must not fail the action"


async def test_execution_without_an_audit_log_is_fine() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    assert (await registry.execute("sample", {"target": "x"}, allow(), ToolContext())).success


def test_the_registry_is_an_allowlist_not_a_scanner() -> None:
    """Nothing appears unless it was explicitly registered."""
    registry = ToolRegistry()
    registry.register_all(build_registry())
    assert set(registry.names) == {"get_time", "calculate", "open_url", "open_application"}


def test_every_shipped_tool_is_well_declared() -> None:
    """Descriptions must state when *not* to use the tool (ARCHITECTURE.md S9)."""
    for spec in build_registry():
        assert spec.description.strip()
        assert "\n" not in spec.description.strip() or len(spec.description) > 80
        assert 0 <= spec.permission_level <= 4
        assert spec.timeout_s > 0


def test_the_engine_default_and_the_tool_levels_agree() -> None:
    """Phase 1 ships no L2+ tool, so a default baseline of 2 runs everything."""
    engine = PermissionEngine(default_level=2)
    for spec in build_registry():
        decision = engine.check(
            scope=spec.name, level=spec.permission_level,
            tool_requires_confirmation=spec.requires_confirmation,
        )
        assert decision.granted, f"{spec.name} would prompt on every call"