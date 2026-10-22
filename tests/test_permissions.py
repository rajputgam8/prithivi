"""Permission engine: the L0-L4 matrix and its precedence rules.

SECURITY.md S1/S2. The engine is the only thing standing between "the model
asked" and "the tool ran", so these tests assert the *order* of the rules as much
as the individual verdicts: a grant must never be able to override a denial, and
a gesture must never be able to authorise itself.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prithivi.security.permissions import (
    DecisionReason,
    Effect,
    PermissionDecision,
    PermissionEngine,
    PermissionLevel,
    Policy,
)

NOW = dt.datetime(2026, 10, 5, 12, 0, tzinfo=dt.UTC)


class Clock:
    """Injectable clock, so expiry tests never sleep.

    Callable, because that is the contract
    :class:`~prithivi.security.confirmation.ConfirmationBroker` uses and the one
    an injected clock is most likely to be written against.
    """

    def __init__(self, now: dt.datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


class MethodClock:
    """The other supported shape: an object exposing ``now()``."""

    def __init__(self, now: dt.datetime = NOW) -> None:
        self._now = now

    def now(self) -> dt.datetime:
        return self._now


def test_both_clock_shapes_are_honoured() -> None:
    """Regression guard for a real defect.

    ``_now`` used to accept only an object with a callable ``now()``. A plain
    callable — the obvious thing to inject, and the form the confirmation broker
    takes — was silently ignored and the engine fell back to the wall clock, so
    expiry assertions were being made against real time and could pass for the
    wrong reason. Both shapes must work.
    """
    expired = NOW - dt.timedelta(seconds=1)

    as_callable = PermissionEngine(default_level=2, clock=Clock(), policies=[
        Policy("send:*", effect=Effect.ALLOW, expires_at=expired),
    ])
    assert as_callable.check(scope="send:msg", level=3).requires_confirmation

    as_object = PermissionEngine(default_level=2, clock=MethodClock(), policies=[
        Policy("send:*", effect=Effect.ALLOW, expires_at=expired),
    ])
    assert as_object.check(scope="send:msg", level=3).requires_confirmation


def test_a_bogus_clock_falls_back_to_the_wall_clock() -> None:
    engine = PermissionEngine(default_level=2, clock=object())
    assert engine.check(scope="get_time", level=0).granted


# --- the plain level matrix, no policies in play ---------------------------


@pytest.mark.security
@pytest.mark.parametrize("level", [0, 1])
def test_below_the_baseline_runs_without_asking(level: int) -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="get_time", level=level)
    assert decision.granted
    assert decision.reason == DecisionReason.BELOW_DEFAULT


@pytest.mark.security
@pytest.mark.parametrize("level", [2, 3])
def test_at_or_above_the_baseline_asks(level: int) -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=level)
    assert not decision.granted
    assert decision.requires_confirmation
    assert not decision.denied
    assert decision.reason == DecisionReason.ALWAYS_CONFIRM_AT_LEVEL


def test_the_baseline_does_not_raise_a_tools_own_level() -> None:
    """A high ``default_level`` must not reclassify a harmless L0 tool as L2.

    ``check`` gates on the tool's declared level; if it gated on
    ``max(level, default_level)`` then simply raising the baseline would start
    prompting for ``get_time``.
    """
    engine = PermissionEngine(default_level=4)
    decision = engine.check(scope="get_time", level=0)
    assert decision.granted
    assert decision.level == PermissionLevel.INFORMATION


def test_a_tool_may_declare_confirmation_below_the_baseline() -> None:
    engine = PermissionEngine(default_level=3)
    decision = engine.check(
        scope="notify", level=0, tool_requires_confirmation=True
    )
    assert decision.requires_confirmation
    assert decision.reason == DecisionReason.TOOL_DECLARES_CONFIRM


def test_a_satisfied_second_factor_unlocks_the_gate() -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2, satisfaction="voice")
    assert decision.granted
    assert decision.reason == DecisionReason.CONFIRMATION_REQUIRED


def test_rule_is_not_a_second_factor_for_the_level_gate() -> None:
    """"rule" satisfies a *policy* but is not a human factor.

    Only ``voice`` and ``ui`` may satisfy the level gate. Letting "rule" through
    here would let any caller label its own pre-check a confirmation.
    """
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2, satisfaction="rule")
    assert decision.requires_confirmation


# --- L4 ---------------------------------------------------------------------


@pytest.mark.security
def test_l4_demands_a_typed_phrase() -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="format_disk", level=4)
    assert decision.requires_confirmation
    assert decision.typed_confirmation_required is True
    assert decision.reason == DecisionReason.L4_TYPED


@pytest.mark.security
def test_l4_is_allowed_only_by_voice_or_ui() -> None:
    engine = PermissionEngine(default_level=2)
    assert engine.check(scope="format_disk", level=4, satisfaction="voice").granted
    assert engine.check(scope="format_disk", level=4, satisfaction="ui").granted
    # "rule" and an untrusted surface must not reach an L4 action.
    assert engine.check(scope="format_disk", level=4, satisfaction="rule").requires_confirmation
    assert engine.check(
        scope="format_disk", level=4, source="gesture", satisfaction="api"
    ).requires_confirmation


def test_a_typed_factor_satisfies_the_l4_gate() -> None:
    """Regression guard for a two-part defect.

    ``ConfirmationOutcome.approve("typed", …)`` reports its factor as
    ``"typed"``, so the agent's post-confirmation re-check passes
    ``satisfaction="typed"``. While the engine accepted only ``voice``/``ui``,
    an L4 action the user had *correctly* confirmed was still refused — and the
    broker's own ``VALID_FACTORS`` listed "typed" as legitimate.
    """
    from prithivi.security.confirmation import ConfirmationOutcome

    assert "typed" in ConfirmationOutcome.VALID_FACTORS
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="format_disk", level=4, satisfaction="typed")
    assert decision.granted


@pytest.mark.security
def test_a_deny_policy_beats_an_allow_policy_and_the_l4_gate() -> None:
    """DENY is checked first, so nothing downstream can re-open it."""
    engine = PermissionEngine(default_level=2, policies=[
        Policy("format_disk", effect=Effect.ALLOW),
        Policy("format_disk:*", effect=Effect.DENY),
    ])
    decision = engine.check(scope="format_disk:/dev/sda", level=4, satisfaction="voice")
    assert decision.denied
    assert decision.reason == DecisionReason.POLICY_DENY


# --- untrusted surfaces -----------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("source", ["gesture", "api", "automation"])
def test_an_untrusted_surface_can_ask_but_never_grant(source: str) -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="open_application:whatsapp", level=3, source=source)
    assert not decision.granted
    assert decision.requires_confirmation
    expected = (
        DecisionReason.GESTURE_SECOND_FACTOR
        if source == "gesture"
        else DecisionReason.UNTRUSTED_SOURCE
    )
    assert decision.reason == expected


@pytest.mark.security
def test_a_gesture_cannot_authorise_itself_with_its_own_satisfaction() -> None:
    """``satisfaction`` is set by a *trusted* surface, not by the caller itself.

    A gesture-originated turn reaching ``check`` with ``source="gesture"`` must
    not be able to pass ``satisfaction="ui"`` and grant itself — that is exactly
    the escalation invariant in SECURITY.md.
    """
    engine = PermissionEngine(default_level=2)
    decision = engine.check(
        scope="delete_everything", level=3, source="gesture", satisfaction="ui"
    )
    assert not decision.granted


def test_an_explicit_allow_policy_does_cover_an_untrusted_source() -> None:
    """A pre-existing narrow user grant *can* authorise a gesture.

    This is the documented exception: the grant came from the user earlier, so
    the gesture is not conferring new authority.
    """
    engine = PermissionEngine(default_level=2, policies=[
        Policy("open_application:whatsapp", effect=Effect.ALLOW),
    ])
    decision = engine.check(
        scope="open_application:whatsapp", level=3, source="gesture"
    )
    assert decision.granted


def test_an_exhausted_grant_stops_covering_an_untrusted_source() -> None:
    engine = PermissionEngine(default_level=2, policies=[
        Policy("open_application:whatsapp", effect=Effect.ALLOW, max_uses=1),
    ])
    first = engine.check(scope="open_application:whatsapp", level=3, source="gesture")
    second = engine.check(scope="open_application:whatsapp", level=3, source="gesture")
    assert first.granted
    assert second.requires_confirmation
    assert second.reason == DecisionReason.SCOPE_EXHAUSTED


# --- consumable grants ------------------------------------------------------


def test_a_narrow_grant_runs_out() -> None:
    engine = PermissionEngine(default_level=2, policies=[
        Policy("post:*", effect=Effect.ALLOW, max_uses=2),
    ])
    assert engine.check(scope="post:x", level=2).granted
    assert engine.check(scope="post:y", level=2).granted
    third = engine.check(scope="post:z", level=2)
    assert third.requires_confirmation
    assert third.reason == DecisionReason.SCOPE_EXHAUSTED


def test_an_exhausted_grant_does_not_leak_to_other_scopes() -> None:
    engine = PermissionEngine(default_level=2, policies=[
        Policy("delete:*", effect=Effect.ALLOW, max_uses=1),
    ])
    assert engine.check(scope="delete:a", level=3).granted
    # The wildcard is spent, so a *different* delete must go back to asking.
    assert engine.check(scope="delete:b", level=3).requires_confirmation


# --- scope patterns and precedence -----------------------------------------


@pytest.mark.security
def test_a_wildcard_deny_beats_an_exact_allow() -> None:
    """DENY is terminal, so specificity never rescues an action from it.

    ``_first_match`` is evaluated per-effect and the deny is checked first, so a
    broad ``calc*`` block cannot be undone by naming the tool exactly. This is
    the deliberately asymmetric direction: over-blocking is recoverable, and the
    inverse would let a narrow grant punch a hole in a broad restriction.
    """
    engine = PermissionEngine(default_level=2, policies=[
        Policy("calc*", effect=Effect.DENY),
        Policy("calculate", effect=Effect.ALLOW),
    ])
    decision = engine.check(scope="calculate", level=2)
    assert decision.denied
    assert decision.reason == DecisionReason.POLICY_DENY


def test_an_exact_scope_beats_a_wildcard_within_the_same_effect() -> None:
    engine = PermissionEngine(default_level=2, policies=[
        Policy("post:*", effect=Effect.ALLOW),
        Policy("post:public", effect=Effect.DENY),
    ])
    # The exact deny wins over the wildcard allow.
    assert engine.check(scope="post:public", level=2).denied
    # The wildcard still allows the others.
    assert engine.check(scope="post:private", level=2).granted


def test_wildcards_match_within_a_segment() -> None:
    engine = PermissionEngine(default_level=2, policies=[
        Policy("open_application:*", effect=Effect.DENY),
    ])
    assert engine.check(scope="open_application:chrome", level=1).denied
    # A different tool is unaffected.
    assert not engine.check(scope="calculate", level=0).denied


def test_an_expired_policy_is_ignored() -> None:
    clock = Clock()
    engine = PermissionEngine(default_level=2, clock=clock, policies=[
        Policy("send:*", effect=Effect.ALLOW, expires_at=NOW + dt.timedelta(seconds=30)),
    ])
    assert engine.check(scope="send:msg", level=3).granted
    clock.advance(seconds=31)
    decision = engine.check(scope="send:msg", level=3)
    assert decision.requires_confirmation


def test_an_expired_deny_policy_stops_denying() -> None:
    """A deny that lapses must fall back to the normal gate, not to allow-all."""
    clock = Clock()
    engine = PermissionEngine(default_level=2, clock=clock, policies=[
        Policy("post:*", effect=Effect.DENY, expires_at=NOW + dt.timedelta(seconds=10)),
    ])
    assert engine.check(scope="post:x", level=3).denied
    clock.advance(seconds=11)
    decision = engine.check(scope="post:x", level=3)
    assert not decision.denied
    assert decision.requires_confirmation


def test_policies_can_be_listed_and_removed() -> None:
    engine = PermissionEngine(default_level=2)
    engine.add_policy(Policy("a:*", effect=Effect.ALLOW))
    engine.add_policy(Policy("a:*", effect=Effect.ALLOW))
    assert len(engine.list_policies()) == 2
    assert engine.remove_policy("a:*") == 2
    assert engine.list_policies() == ()
    assert engine.remove_policy("a:*") == 0


def test_policy_is_live_and_consume_are_self_consistent() -> None:
    unlimited = Policy("x")
    assert unlimited.is_live() is True
    assert unlimited.consume() is True

    limited = Policy("y", max_uses=1)
    assert limited.is_live() is True
    assert limited.consume() is True
    assert limited.remaining_uses == 0
    assert limited.is_live() is False
    assert limited.consume() is False


# --- decision shape ---------------------------------------------------------


def test_decision_describe_reads_as_a_sentence() -> None:
    allowed = PermissionDecision(
        allowed=True, requires_confirmation=False, level=0, effect=Effect.ALLOW,
        reason=DecisionReason.BELOW_DEFAULT,
    )
    ask = PermissionDecision(
        allowed=False, requires_confirmation=True, level=3, effect=Effect.CONFIRM,
        reason=DecisionReason.ALWAYS_CONFIRM_AT_LEVEL, consequence="Delete 3 files",
    )
    denied = PermissionDecision(
        allowed=False, requires_confirmation=False, level=4, effect=Effect.DENY,
        reason=DecisionReason.POLICY_DENY,
    )
    assert "Allowed" in allowed.describe() and "information" in allowed.describe()
    assert "Delete 3 files" in ask.describe()
    assert "sensitive" in ask.describe()
    assert denied.describe().startswith("Denied")


def test_a_granted_decision_is_never_simultaneously_blocking() -> None:
    engine = PermissionEngine(default_level=2)
    for level in range(5):
        decision = engine.check(scope="s", level=level)
        assert not (decision.granted and decision.requires_confirmation)