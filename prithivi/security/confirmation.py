"""Confirmation broker (SECURITY.md S3).

An L2+ action is not refused — it is *pending* until you approve it, and the
approval is bound to the exact action. Three properties make this safe:

1. **Bound to the action.** The token commits to ``sha256(tool + canonical args)``.
   Approving "delete 3 files" cannot be replayed against different files.
2. **Single use.** Consumed on first successful use; a second attempt re-asks.
3. **Expiring.** Default 60 s for L2, 120 s for L3, 180 s for L4. An expired
   pending action is a decline, and is audited as one.

There is deliberately no programmatic ``auto_approve`` flag on this class. A
caller that holds a decision must still pass a human-originated factor
(``voice``/``ui``) or a pre-existing narrow :class:`Policy`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import secrets as pysecrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .permissions import PermissionDecision, PermissionLevel
from .secrets import redactor

__all__ = [
    "ConfirmationBroker",
    "ConfirmationOutcome",
    "ConfirmationRequest",
    "action_fingerprint",
    "new_typed_phrase",
]

ConfirmHandler = Callable[["ConfirmationRequest"], Awaitable["ConfirmationOutcome"]]


def action_fingerprint(tool: str, args: object) -> str:
    """Stable digest binding a confirmation to one exact tool call."""
    payload = tool + "\x00" + json.dumps(redactor.scrub_obj(args), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Two short, unambiguous words for the L4 typed phrase. Kept tiny and distinct
#: so a speech recogniser cannot plausibly produce one by accident, and so the
#: user can retype it on a phone keyboard without hunting through a long string.
_TYPED_PHRASE_WORDS = (
    "amber", "basalt", "cobalt", "dune", "ember", "fjord", "glacier", "harbour",
    "ivory", "juniper", "kelp", "lantern", "meadow", "nimbus", "onyx", "prairie",
    "quartz", "ridge", "summit", "tundra", "umber", "violet", "willow", "zephyr",
)


def new_typed_phrase() -> str:
    """Mint the two-word phrase an L4 action must be confirmed with."""
    return f"{pysecrets.choice(_TYPED_PHRASE_WORDS)} {pysecrets.choice(_TYPED_PHRASE_WORDS)}"


@dataclass
class ConfirmationRequest:
    """What the user is being asked to approve, and what it will actually do."""

    token: str
    tool: str
    fingerprint: str
    level: int
    consequence: str
    args_preview: str
    created_at: dt.datetime
    expires_at: dt.datetime
    used: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def level_name(self) -> str:
        return PermissionLevel.name_of(self.level)

    @property
    def typed_phrase(self) -> str:
        """The phrase this request must be confirmed with, if it needs one.

        Empty for every level below L4. Exposed as a property rather than read
        out of ``metadata`` at the call site so a UI cannot accidentally prompt
        for an empty phrase.
        """
        return str(self.metadata.get("typed_phrase", ""))

    def is_expired(self, now: dt.datetime | None = None) -> bool:
        now = now or dt.datetime.now(dt.UTC)
        return self.expires_at <= now

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "tool": self.tool,
            "level": self.level,
            "level_name": self.level_name,
            "consequence": self.consequence,
            "args_preview": self.args_preview,
            "expires_at": self.expires_at.isoformat(),
            "used": self.used,
            "typed_phrase": self.typed_phrase,
        }


@dataclass(frozen=True)
class ConfirmationOutcome:
    """The user's answer.

    Attributes:
        approved: whether the action may proceed.
        factor: ``voice`` | ``ui`` | ``rule`` | ``typed``. Anything else is
            treated as *not* a second factor.
        typed_phrase: for L4, the short phrase the user must repeat.
        reason: shown to the agent when not approved.
    """

    approved: bool
    factor: str = "ui"
    typed_phrase: str | None = None
    reason: str = ""

    VALID_FACTORS = frozenset({"voice", "ui", "rule", "typed"})

    @classmethod
    def approve(cls, factor: str = "ui", typed_phrase: str | None = None) -> ConfirmationOutcome:
        return cls(approved=True, factor=factor, typed_phrase=typed_phrase)

    @classmethod
    def decline(cls, reason: str = "declined by user") -> ConfirmationOutcome:
        return cls(approved=False, factor="none", reason=reason)

    @property
    def is_valid_factor(self) -> bool:
        return self.factor in self.VALID_FACTORS


class ConfirmationBroker:
    """Issues, validates, and consumes approval tokens.

    Args:
        handler: called to ask the user (voice prompt, UI modal). Injected so
            that tests and the CLI can supply different interfaces.
        clock: injectable time source.
    """

    def __init__(
        self,
        handler: ConfirmHandler | None = None,
        *,
        clock: Callable[[], dt.datetime] | None = None,
        default_timeout_s: int = 120,
    ) -> None:
        self._handler = handler
        self._clock = clock or (lambda: dt.datetime.now(dt.UTC))
        self.default_timeout_s = default_timeout_s
        self._pending: dict[str, ConfirmationRequest] = {}

    def set_handler(self, handler: ConfirmHandler | None) -> None:
        """Attach the interface that asks the user.

        The web server and the voice loop both construct the broker before their
        own UI exists, so the handler is swappable at startup. Swapping it never
        grants an approval: an absent handler still declines.
        """
        self._handler = handler

    # -- lifecycle -----------------------------------------------------
    def _ttl_for(self, level: int) -> int:
        """SECURITY.md S3 timeouts: L2 60 s, L3 120 s, L4 180 s."""
        return {0: 30, 1: 30, 2: 60, 3: 120, 4: 180}.get(level, self.default_timeout_s)

    def issue(self, *, tool: str, args: object, level: int, consequence: str, **metadata: object) -> ConfirmationRequest:
        """Create a pending request without asking anyone yet.

        An L4 request always gets a freshly minted typed phrase. It has to be
        minted *here* rather than at the call site: :meth:`request` only has the
        decision to hand, and a phrase generated outside this method would leave
        L4 actions permanently unapprovable — the check in :meth:`request` would
        always compare against an empty expectation.
        """
        now = self._clock()
        if level >= PermissionLevel.CRITICAL and "typed_phrase" not in metadata:
            metadata["typed_phrase"] = new_typed_phrase()
        request = ConfirmationRequest(
            token=pysecrets.token_urlsafe(24),
            tool=tool,
            fingerprint=action_fingerprint(tool, args),
            level=level,
            consequence=consequence,
            args_preview=redactor.scrub(json.dumps(args, sort_keys=True, default=str))[:400],
            created_at=now,
            expires_at=now + dt.timedelta(seconds=self._ttl_for(level)),
            metadata=dict(metadata),
        )
        self._pending[request.token] = request
        return request

    @property
    def pending(self) -> tuple[ConfirmationRequest, ...]:
        now = self._clock()
        return tuple(r for r in self._pending.values() if not r.used and not r.is_expired(now))

    def get(self, token: str) -> ConfirmationRequest | None:
        return self._pending.get(token)

    # -- the ask -------------------------------------------------------
    async def request(self, decision: PermissionDecision, *, tool: str, args: object) -> ConfirmationOutcome:
        """Ask the user to approve ``tool(args)``.

        Returns a decline (never raises) on expiry, refusal, or the absence of a
        handler — so a caller cannot accidentally treat "nobody answered" as
        approval.
        """
        if decision.granted:
            return ConfirmationOutcome.approve(factor="rule")

        request = self.issue(
            tool=tool, args=args, level=decision.level,
            consequence=decision.consequence or f"Run {tool}",
        )

        if self._handler is None:
            self._pending.pop(request.token, None)
            return ConfirmationOutcome.decline("no confirmation interface is attached")

        try:
            outcome = await self._handler(request)
        except Exception as exc:  # noqa: BLE001 - a broken UI must not mean "approved"
            self._pending.pop(request.token, None)
            return ConfirmationOutcome.decline(f"confirmation interface failed: {exc}")

        # L4 additionally requires the typed phrase to match.
        if decision.typed_confirmation_required:
            expected = request.typed_phrase
            if not outcome.approved:
                self._pending.pop(request.token, None)
                return outcome
            if not expected or not outcome.typed_phrase or outcome.typed_phrase.strip().lower() != expected.lower():
                self._pending.pop(request.token, None)
                return ConfirmationOutcome.decline("typed confirmation phrase did not match")
            outcome = ConfirmationOutcome.approve(factor="typed", typed_phrase=outcome.typed_phrase)

        if not outcome.approved:
            self._pending.pop(request.token, None)
            return outcome

        if not outcome.is_valid_factor:
            self._pending.pop(request.token, None)
            return ConfirmationOutcome.decline(f"{outcome.factor!r} is not a valid confirmation factor")

        if request.is_expired(self._clock()):
            self._pending.pop(request.token, None)
            return ConfirmationOutcome.decline("confirmation expired before you answered")

        return outcome

    # -- redemption ----------------------------------------------------
    def redeem(self, token: str, *, tool: str, args: object) -> PermissionDecision:
        """Consume a token and return the ALLOW decision it authorises.

        Rejects (as a denied decision, never an exception) an unknown, used, or
        expired token, or one issued for a *different* action.
        """
        request = self._pending.get(token)
        now = self._clock()

        def deny(reason: str) -> PermissionDecision:
            self._pending.pop(token, None)
            return PermissionDecision(
                allowed=False, requires_confirmation=False, level=3, effect="deny",
                reason=reason, scope=tool,
            )

        if request is None:
            return deny("unknown or already-consumed confirmation token")
        if request.used:
            return deny("confirmation token already used (single use)")
        if request.is_expired(now):
            return deny("confirmation token expired")
        if request.fingerprint != action_fingerprint(tool, args):
            return deny("confirmation was issued for a different action")

        request.used = True
        self._pending.pop(token, None)
        return PermissionDecision(
            allowed=True, requires_confirmation=False, level=request.level, effect="allow",
            reason="confirmed", scope=f"{request.tool}:{token[:8]}",
        )

    def cancel(self, token: str) -> bool:
        return self._pending.pop(token, None) is not None

    def clear(self) -> None:
        self._pending.clear()


# -- voice confirmation grammar ------------------------------------------
# SECURITY.md S3: a generic "yes" after a *different* question must not count.
# Matching is scoped to the pending tool plus an explicit approval verb.

_APPROVE_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "confirm", "confirmed", "do it", "go ahead",
    "proceed", "approved", "authorize", "authorise", "sure", "ok", "okay",
    "affirmative", "correct", "that's right", "thats right",
})
_DECLINE_WORDS = frozenset({
    "no", "nope", "nah", "don't", "dont", "cancel", "stop", "deny", "denied",
    "never mind", "nevermind", "not now", "wait", "hold on", "negative",
})


def classify_voice_confirmation(
    utterance: str,
    request: ConfirmationRequest,
    *,
    confidence: float = 1.0,
    min_confidence: float = 0.55,
) -> ConfirmationOutcome:
    """Turn a spoken reply into a decision for this specific pending action.

    Low-confidence transcription is treated as *no* confirmation, which is the
    safe direction: it re-asks rather than silently approving.
    """
    if confidence < min_confidence:
        return ConfirmationOutcome.decline("I did not hear that clearly enough to confirm")

    text = utterance.strip().lower().strip(".!?, ")
    if not text:
        return ConfirmationOutcome.decline("empty reply")

    # The tool name or a salient argument must appear: "yes" alone is not enough
    # unless the reply is unambiguous *and* names the action.
    names_action = request.tool.lower() in text or any(
        word for word in text.split() if len(word) > 4
    )

    if text in _DECLINE_WORDS or any(text.startswith(w) for w in _DECLINE_WORDS if " " in w):
        return ConfirmationOutcome.decline("you declined")

    if text in _APPROVE_WORDS:
        if not names_action:
            return ConfirmationOutcome.decline("ambiguous 'yes' — say the action name to confirm")
        return ConfirmationOutcome.approve(factor="voice")

    if any(word in text for word in _APPROVE_WORDS) and names_action:
        return ConfirmationOutcome.approve(factor="voice")
    if any(word in text for word in _DECLINE_WORDS):
        return ConfirmationOutcome.decline("you declined")
    return ConfirmationOutcome.decline(f"'{utterance[:40]}' is not a clear yes or no")
