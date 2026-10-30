"""Permission engine — the enforcement point (SECURITY.md S1/S2).

The governing rule: **the model is never an authority.** The LLM proposes a tool
call; this module disposes of it. Every executor is reached only through a
:class:`PermissionDecision`, which the registry requires as a positional
argument so it cannot be bypassed with a truthy value.

Levels (L0-L4) and their defaults come from SECURITY.md S1. A user's stored
policy may tighten or, explicitly, loosen a tool, but the *effective* decision is
always the most restrictive of the tool declaration and the matching policy.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

__all__ = [
    "DecisionReason",
    "Effect",
    "PermissionDecision",
    "PermissionEngine",
    "PermissionLevel",
    "Policy",
]


class PermissionLevel(int):
    """L0..L4, ordered so ``max()`` means "most restrictive"."""

    INFORMATION = 0
    LOW_RISK = 1
    USER_DATA = 2
    SENSITIVE = 3
    CRITICAL = 4

    NAMES: ClassVar[dict[int, str]] = {
        0: "information",
        1: "low-risk",
        2: "user-data",
        3: "sensitive",
        4: "critical",
    }

    @classmethod
    def name_of(cls, level: int) -> str:
        return cls.NAMES.get(level, "unknown")


class Effect(str):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class DecisionReason(str):
    """Why a decision was reached. Logged verbatim; surfaced to the narrator."""

    BELOW_DEFAULT = "below_default_level"
    NO_POLICY = "no_policy_match"
    POLICY_ALLOW = "policy_allow"
    POLICY_CONFIRM = "policy_confirm"
    POLICY_DENY = "policy_deny"
    TOOL_DECLARES_CONFIRM = "tool_requires_confirmation"
    GESTURE_SECOND_FACTOR = "gesture_needs_second_factor"
    UNTRUSTED_SOURCE = "untrusted_source_requires_confirmation"
    ALWAYS_CONFIRM_AT_LEVEL = "level_requires_confirmation"
    L4_TYPED = "l4_requires_typed_confirmation"
    SCOPE_EXHAUSTED = "narrow_grant_exhausted"
    CONFIRMATION_REQUIRED = "confirmation_required"


#: Surfaces that may never, on their own, complete an L2+ action. Mirrors
#: :attr:`prithivi.core.state.TurnSource.is_untrusted_for_l3`; duplicated as plain
#: strings so this module stays importable on its own.
_UNTRUSTED_SOURCES = frozenset({"gesture", "api", "automation"})

#: Factors that may satisfy an L4 gate. "typed" is the strongest: the user
#: repeated a phrase minted for this exact action.
_L4_FACTORS = frozenset({"voice", "ui", "typed"})


@dataclass(frozen=True)
class PermissionDecision:
    """The verdict. ``registry.execute`` will not run without one."""

    allowed: bool
    requires_confirmation: bool
    level: int
    effect: str
    reason: str
    scope: str = ""
    expires_at: dt.datetime | None = None
    typed_confirmation_required: bool = False
    consequence: str = ""

    @property
    def denied(self) -> bool:
        return self.effect == Effect.DENY

    @property
    def granted(self) -> bool:
        """True when the executor may run now (confirmation already satisfied)."""
        return self.allowed and not self.requires_confirmation

    def describe(self) -> str:
        level_name = PermissionLevel.name_of(self.level)
        if self.denied:
            return f"Denied ({level_name}): {self.consequence or self.scope or self.reason}"
        if self.requires_confirmation:
            return f"Needs your confirmation ({level_name}): {self.consequence or self.scope}"
        return f"Allowed ({level_name})"


@dataclass
class Policy:
    """A user grant or restriction.

    ``scope_pattern`` is a glob over ``tool:name`` or ``tool:*`` / ``*:name``.
    ``max_uses`` implements narrow, consumable trust for automations.
    """

    scope_pattern: str
    effect: str = Effect.CONFIRM
    max_uses: int | None = None
    remaining_uses: int | None = None
    requires_factor: str | None = None
    reason: str = "user_policy"
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    expires_at: dt.datetime | None = None
    source: str = "user"

    def matches(self, scope: str) -> bool:
        return fnmatch.fnmatchcase(scope, self.scope_pattern)

    def is_live(self, now: dt.datetime | None = None) -> bool:
        now = now or dt.datetime.now(dt.UTC)
        if self.expires_at is not None and self.expires_at <= now:
            return False
        if self.max_uses is not None:
            remaining = self.remaining_uses if self.remaining_uses is not None else self.max_uses
            return remaining > 0
        return True

    def consume(self) -> bool:
        """Spend one use. Returns False when the grant is exhausted."""
        if self.max_uses is None:
            return True
        remaining = self.remaining_uses if self.remaining_uses is not None else self.max_uses
        if remaining <= 0:
            return False
        self.remaining_uses = remaining - 1
        return True


class PermissionEngine:
    """Decides ``ALLOW | CONFIRM_REQUIRED | DENY`` for a proposed action.

    Args:
        default_level: baseline gate applied when no policy matches.
        policies: initial user policies.
        clock: injected time source — a ``Callable[[], datetime]`` or an object
            with a ``now()`` method. Supplying one is mandatory for any test of
            policy expiry or TTLs, since those paths read it.
    """

    def __init__(
        self,
        *,
        default_level: int = 2,
        policies: Iterable[Policy] = (),
        clock: object = None,
    ) -> None:
        self.default_level = default_level
        self.policies: list[Policy] = list(policies)
        self._clock = clock

    # -- policy management -------------------------------------------
    def add_policy(self, policy: Policy) -> Policy:
        self.policies.append(policy)
        return policy

    def remove_policy(self, scope_pattern: str) -> int:
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.scope_pattern != scope_pattern]
        return before - len(self.policies)

    def list_policies(self) -> Sequence[Policy]:
        return tuple(self.policies)

    def _now(self) -> dt.datetime:
        """Read the injected clock, falling back to the wall clock.

        Accepts both shapes on purpose: a bare callable (the same contract
        :class:`~prithivi.security.confirmation.ConfirmationBroker` uses) and an
        object exposing a ``now()`` method. The previous implementation honoured
        *only* the second, so passing a plain callable — the obvious thing to do,
        and the form the rest of the codebase uses — silently fell back to real
        time and every expiry test would have passed against the wrong clock.
        """
        clock = self._clock
        if clock is None:
            return dt.datetime.now(dt.UTC)
        if callable(clock):
            return clock()  # type: ignore[no-any-return]
        now = getattr(clock, "now", None)
        if callable(now):
            return now()  # type: ignore[no-any-return]
        return dt.datetime.now(dt.UTC)

    # -- the decision --------------------------------------------------
    def check(
        self,
        *,
        scope: str,
        level: int,
        tool_requires_confirmation: bool = False,
        source: str = "agent",
        consequence: str = "",
        satisfaction: str | None = None,
    ) -> PermissionDecision:
        """Evaluate one proposed action.

        Args:
            scope: ``tool_name`` or ``tool_name:detail``; policies glob against it.
            level: the tool's declared permission level.
            tool_requires_confirmation: set by a tool that always needs consent.
            source: originating :class:`~prithivi.core.state.TurnSource` value.
            satisfaction: set to ``"voice"``/``"ui"``/``"rule"`` when the caller
                already obtained a second factor for this exact action.
        """
        now = self._now()
        # ``default_level`` is the *baseline below which nothing is gated*; it does
        # not raise a tool's own level. Gating uses the declared level so that an
        # L1 tool stays L1 even when the baseline is L2.
        gate = level

        # 1. DENY is terminal and checked first: nothing can override a denial.
        deny = self._first_match(scope, Effect.DENY, now)
        if deny is not None:
            return PermissionDecision(
                allowed=False, requires_confirmation=False, level=gate, effect=Effect.DENY,
                reason=DecisionReason.POLICY_DENY, scope=scope, consequence=consequence,
            )

        # 2. L4 always demands a typed phrase from a trusted surface.
        #
        #    "typed" is accepted here because it is strictly stronger than
        #    "voice"/"ui": it required the user to repeat a phrase minted for
        #    this exact action. Excluding it meant an L4 action could be
        #    correctly confirmed and then still refused by the re-check, because
        #    the confirmation outcome reports its factor as "typed".
        if gate >= PermissionLevel.CRITICAL:
            if satisfaction in _L4_FACTORS:
                return PermissionDecision(
                    allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                    reason=DecisionReason.POLICY_ALLOW, scope=scope, consequence=consequence,
                )
            return PermissionDecision(
                allowed=False, requires_confirmation=True, level=gate, effect=Effect.CONFIRM,
                reason=DecisionReason.L4_TYPED, scope=scope, consequence=consequence,
                typed_confirmation_required=True,
                expires_at=now + dt.timedelta(seconds=180),
            )

        # 3. A gesture (or any untrusted surface) can never authorise alone
        #    (SECURITY.md invariant 3). It may *ask*, never grant.
        #
        #    Deliberately NOT gated on ``satisfaction is None``: a claimed second
        #    factor arriving on the same untrusted turn is part of the attack, not
        #    a mitigation. Letting it through would mean any caller could pair
        #    ``source="gesture"`` with ``satisfaction="ui"`` and be granted an
        #    L2+ action at step 6. The only thing that may pre-authorise an
        #    untrusted surface is an explicit user ALLOW policy below.
        if source in _UNTRUSTED_SOURCES:
            live_allow, spent_allow = self._allow_candidates(scope, now)
            if live_allow is None:
                reason = (
                    DecisionReason.SCOPE_EXHAUSTED if spent_allow is not None
                    else DecisionReason.GESTURE_SECOND_FACTOR if source == "gesture"
                    else DecisionReason.UNTRUSTED_SOURCE
                )
                return PermissionDecision(
                    allowed=False, requires_confirmation=True, level=gate, effect=Effect.CONFIRM,
                    reason=reason,
                    scope=scope, consequence=consequence,
                    expires_at=now + dt.timedelta(seconds=120),
                )
            live_allow.consume()
            return PermissionDecision(
                allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                reason=DecisionReason.POLICY_ALLOW, scope=scope, consequence=consequence,
            )

        # 4. Explicit allow policy, if it is still live.
        allow, spent = self._allow_candidates(scope, now)
        if allow is not None:
            allow.consume()
            return PermissionDecision(
                allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                reason=DecisionReason.POLICY_ALLOW, scope=scope, consequence=consequence,
            )
        if spent is not None:
            # The grant matched this scope but has no uses left. Saying so is the
            # difference between "I need permission" and "your one-shot grant is
            # spent" — and the audit trail is the only place the second fact can
            # be recovered from.
            return PermissionDecision(
                allowed=False, requires_confirmation=True, level=gate, effect=Effect.CONFIRM,
                reason=DecisionReason.SCOPE_EXHAUSTED, scope=scope, consequence=consequence,
                expires_at=now + dt.timedelta(seconds=120),
            )

        # 5. A live confirm policy means "ask", unless the action is below the gate.
        confirm = self._first_match(scope, Effect.CONFIRM, now)
        if confirm is not None:
            if level < self.default_level and not tool_requires_confirmation:
                return PermissionDecision(
                    allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                    reason=DecisionReason.BELOW_DEFAULT, scope=scope, consequence=consequence,
                )
            if satisfaction in {"voice", "ui", "rule"}:
                return PermissionDecision(
                    allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                    reason=DecisionReason.POLICY_CONFIRM, scope=scope, consequence=consequence,
                )
            return PermissionDecision(
                allowed=False, requires_confirmation=True, level=gate, effect=Effect.CONFIRM,
                reason=DecisionReason.POLICY_CONFIRM, scope=scope, consequence=consequence,
                expires_at=confirm.expires_at or now + dt.timedelta(seconds=120),
            )

        # 6. No policy: fall back to the level gate. Anything below the baseline
        #    runs freely; L2+ asks; L0-L1 with a low baseline just runs.
        if level < self.default_level and not tool_requires_confirmation:
            return PermissionDecision(
                allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                reason=DecisionReason.BELOW_DEFAULT, scope=scope, consequence=consequence,
            )
        if tool_requires_confirmation or gate >= PermissionLevel.USER_DATA:
            if satisfaction in {"voice", "ui"}:
                return PermissionDecision(
                    allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
                    reason=DecisionReason.CONFIRMATION_REQUIRED, scope=scope, consequence=consequence,
                )
            return PermissionDecision(
                allowed=False, requires_confirmation=True, level=gate, effect=Effect.CONFIRM,
                reason=DecisionReason.TOOL_DECLARES_CONFIRM if tool_requires_confirmation
                else DecisionReason.ALWAYS_CONFIRM_AT_LEVEL,
                scope=scope, consequence=consequence,
                expires_at=now + dt.timedelta(seconds=60 if gate <= 2 else 120),
            )
        return PermissionDecision(
            allowed=True, requires_confirmation=False, level=gate, effect=Effect.ALLOW,
            reason=DecisionReason.NO_POLICY, scope=scope, consequence=consequence,
        )

    def _allow_candidates(self, scope: str, now: dt.datetime) -> tuple[Policy | None, Policy | None]:
        """Return ``(live_allow_policy, spent_allow_policy)`` for ``scope``.

        Both halves matter. A single live/non-live lookup cannot tell "you have
        no grant for this" apart from "your grant is used up", and the two need
        different reasons — :attr:`DecisionReason.SCOPE_EXHAUSTED` only ever
        means the second. Since ``_first_match`` filters on
        :meth:`Policy.is_live`, a spent policy is invisible to it, so the spent
        case is found by a second, unfiltered lookup.
        """
        live = self._first_match(scope, Effect.ALLOW, now)
        if live is not None:
            return live, None
        spent = self._first_match(scope, Effect.ALLOW, now, include_exhausted=True)
        return None, spent

    def _first_match(
        self, scope: str, effect: str, now: dt.datetime, *, include_exhausted: bool = False
    ) -> Policy | None:
        """Most specific live policy wins: exact scope beats a wildcard."""
        candidates = [
            p for p in self.policies
            if p.effect == effect and p.matches(scope)
            and (include_exhausted or p.is_live(now))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.scope_pattern.count("*") == 0, len(p.scope_pattern)))
