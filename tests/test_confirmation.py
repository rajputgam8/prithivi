"""Confirmation broker: an approval must be bound to one exact action.

SECURITY.md S3. The properties under test are the ones that make an approval
safe to honour — binding, single use, expiry, and a grammar that does not accept
a stray "yes" meant for a different question.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prithivi.security.confirmation import (
    ConfirmationBroker,
    ConfirmationOutcome,
    classify_voice_confirmation,
    new_typed_phrase,
)
from prithivi.security.permissions import PermissionEngine, PermissionLevel

NOW = dt.datetime(2026, 10, 5, 12, 0, tzinfo=dt.UTC)


class Clock:
    def __init__(self, now: dt.datetime = NOW) -> None:
        self._now = now

    def __call__(self) -> dt.datetime:
        return self._now

    def advance(self, **kwargs: float) -> None:
        self._now = self._now + dt.timedelta(**kwargs)


def asking(outcome: ConfirmationOutcome):
    """A handler that returns a fixed outcome, recording what it was asked."""
    seen: list[object] = []

    async def handler(request):
        seen.append(request)
        return outcome

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


# --- the fingerprint --------------------------------------------------------


def test_the_fingerprint_ignores_argument_order() -> None:
    assert _fingerprint("send", {"a": 1, "b": 2}) == _fingerprint("send", {"b": 2, "a": 1})


def test_the_fingerprint_separates_different_arguments() -> None:
    """The whole point of binding: "delete 3 files" != "delete 4 files"."""
    assert _fingerprint("delete", {"n": 3}) != _fingerprint("delete", {"n": 4})
    assert _fingerprint("delete", {"n": 3}) != _fingerprint("erase", {"n": 3})


def test_the_fingerprint_does_not_store_the_secret() -> None:
    """A confirmed action's args are hashed, not stored in the clear.

    The fingerprint is logged and shown in the audit trail, so a token pasted
    into the arguments must not survive into it.
    """
    token = "sk-abcdefghijklmnopqrstuvwxyz1234"
    digest = _fingerprint("post", {"body": token})
    assert token not in digest


def _fingerprint(tool: str, args: object) -> str:
    from prithivi.security.confirmation import action_fingerprint

    return action_fingerprint(tool, args)


# --- issuing and TTLs -------------------------------------------------------


@pytest.mark.parametrize(
    "level,ttl", [(0, 30), (1, 30), (2, 60), (3, 120), (4, 180)]
)
def test_the_timeout_grows_with_the_level(level: int, ttl: int) -> None:
    clock = Clock()
    broker = ConfirmationBroker(clock=clock)
    request = broker.issue(tool="t", args={}, level=level, consequence="c")
    assert (request.expires_at - request.created_at).total_seconds() == ttl


def test_only_l4_mints_a_typed_phrase() -> None:
    broker = ConfirmationBroker()
    assert broker.issue(tool="t", args={}, level=4, consequence="c").typed_phrase
    for level in (0, 1, 2, 3):
        assert broker.issue(tool="t", args={}, level=level, consequence="c").typed_phrase == ""


def test_each_l4_phrase_is_fresh() -> None:
    """A reused phrase would let an old transcript authorise a new action."""
    broker = ConfirmationBroker()
    phrases = {
        broker.issue(tool="t", args={}, level=4, consequence="c").typed_phrase
        for _ in range(40)
    }
    assert len(phrases) > 1


def test_the_typed_phrase_is_two_speakable_words() -> None:
    phrase = new_typed_phrase()
    words = phrase.split()
    assert len(words) == 2
    # Letters only: no digits or punctuation for a speech recogniser to mangle.
    assert all(word.isalpha() and word.islower() for word in words)


# --- asking -----------------------------------------------------------------


async def test_an_already_granted_action_needs_no_question() -> None:
    engine = PermissionEngine(default_level=2)
    granted = engine.check(scope="get_time", level=0)
    assert granted.granted

    broker = ConfirmationBroker(asking(ConfirmationOutcome.decline()))
    outcome = await broker.request(granted, tool="get_time", args={})
    assert outcome.approved
    assert outcome.factor == "rule"


async def test_no_handler_declines_and_never_approves() -> None:
    """Silence must not read as consent."""
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2)

    outcome = await ConfirmationBroker().request(decision, tool="write_file", args={})
    assert outcome.approved is False
    assert "no confirmation interface" in outcome.reason


async def test_a_broken_handler_declines_rather_than_raising() -> None:
    """A crashing UI is not consent, and must not kill the agent loop."""

    async def handler(request):
        raise RuntimeError("the modal crashed")

    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2)
    outcome = await ConfirmationBroker(handler).request(decision, tool="write_file", args={})
    assert outcome.approved is False
    assert "confirmation interface failed" in outcome.reason


async def test_an_invented_factor_is_not_a_confirmation() -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2)
    handler = asking(ConfirmationOutcome(approved=True, factor="vibes"))
    outcome = await ConfirmationBroker(handler).request(decision, tool="write_file", args={})
    assert outcome.approved is False
    assert "not a valid confirmation factor" in outcome.reason


async def test_a_l4_action_needs_the_exact_phrase() -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="format_disk", level=4)
    assert decision.typed_confirmation_required

    broker: ConfirmationBroker
    seen: list[str] = []

    async def handler(request):
        seen.append(request.typed_phrase)
        return ConfirmationOutcome.approve("ui", typed_phrase="wrong words entirely")

    broker = ConfirmationBroker(handler)
    outcome = await broker.request(decision, tool="format_disk", args={})
    assert outcome.approved is False
    assert "did not match" in outcome.reason
    assert seen and seen[0]  # the phrase really was issued and shown


async def test_a_l4_action_runs_when_the_phrase_is_repeated() -> None:
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="format_disk", level=4)

    async def handler(request):
        # Case and surrounding whitespace must not matter to a person typing.
        return ConfirmationOutcome.approve(
            "ui", typed_phrase=f"  {request.typed_phrase.upper()}  "
        )

    outcome = await ConfirmationBroker(handler).request(decision, tool="format_disk", args={})
    assert outcome.approved is True
    assert outcome.factor == "typed"


async def test_an_answer_arriving_too_late_is_a_decline() -> None:
    clock = Clock()
    engine = PermissionEngine(default_level=2)
    decision = engine.check(scope="write_file", level=2)

    async def handler(request):
        clock.advance(seconds=10_000)  # the user wandered off
        return ConfirmationOutcome.approve("ui")

    outcome = await ConfirmationBroker(handler, clock=clock).request(
        decision, tool="write_file", args={}
    )
    assert outcome.approved is False
    assert "expired" in outcome.reason


# --- redemption -------------------------------------------------------------


def test_a_token_authorises_exactly_one_use() -> None:
    broker = ConfirmationBroker()
    request = broker.issue(tool="write_file", args={"path": "/tmp/x"}, level=2, consequence="c")

    first = broker.redeem(request.token, tool="write_file", args={"path": "/tmp/x"})
    assert first.granted

    second = broker.redeem(request.token, tool="write_file", args={"path": "/tmp/x"})
    assert second.denied
    assert "already" in second.reason or "consumed" in second.reason


@pytest.mark.security
def test_an_approval_cannot_be_replayed_on_different_arguments() -> None:
    """The token commits to the exact call, so swapping arguments must fail."""
    broker = ConfirmationBroker()
    request = broker.issue(tool="write_file", args={"path": "/tmp/x"}, level=2, consequence="c")

    decision = broker.redeem(request.token, tool="write_file", args={"path": "/etc/passwd"})
    assert decision.denied
    assert "different action" in decision.reason


def test_an_unknown_token_is_denied_rather_than_raising() -> None:
    decision = ConfirmationBroker().redeem("made-up", tool="write_file", args={})
    assert decision.denied


def test_an_expired_token_is_denied() -> None:
    clock = Clock()
    broker = ConfirmationBroker(clock=clock)
    request = broker.issue(tool="write_file", args={"p": 1}, level=2, consequence="c")
    clock.advance(seconds=61)
    decision = broker.redeem(request.token, tool="write_file", args={"p": 1})
    assert decision.denied
    assert "expired" in decision.reason


def test_pending_hides_spent_and_expired_requests() -> None:
    clock = Clock()
    broker = ConfirmationBroker(clock=clock)
    spent = broker.issue(tool="a", args={}, level=2, consequence="c")
    broker.redeem(spent.token, tool="a", args={})
    broker.issue(tool="b", args={}, level=2, consequence="c")

    assert [r.tool for r in broker.pending] == ["b"]

    clock.advance(seconds=10_000)
    assert broker.pending == ()


def test_a_token_can_be_cancelled_or_cleared() -> None:
    broker = ConfirmationBroker()
    request = broker.issue(tool="a", args={}, level=2, consequence="c")
    assert broker.cancel(request.token) is True
    assert broker.cancel(request.token) is False

    broker.issue(tool="b", args={}, level=2, consequence="c")
    broker.clear()
    assert broker.pending == ()


def test_a_request_can_be_fetched_by_token() -> None:
    broker = ConfirmationBroker()
    request = broker.issue(tool="a", args={}, level=2, consequence="c")
    assert broker.get(request.token) is request
    assert broker.get("nope") is None


def test_the_args_preview_is_scrubbed_and_bounded() -> None:
    broker = ConfirmationBroker()
    request = broker.issue(
        tool="post",
        args={"body": "sk-abcdefghijklmnopqrstuvwxyz1234", "pad": "x" * 5000},
        level=2,
        consequence="c",
    )
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in request.args_preview
    assert len(request.args_preview) <= 400


def test_the_serialised_request_carries_everything_a_ui_needs() -> None:
    broker = ConfirmationBroker()
    payload = broker.issue(
        tool="post", args={"body": "hi"}, level=4, consequence="Publish this"
    ).to_dict()
    assert payload["tool"] == "post"
    assert payload["level_name"] == PermissionLevel.name_of(4)
    assert payload["consequence"] == "Publish this"
    assert payload["typed_phrase"]


# --- the voice/typed grammar ------------------------------------------------


def _request(tool: str = "delete_everything", level: int = 3):
    return ConfirmationBroker().issue(tool=tool, args={}, level=level, consequence="c")


def test_a_bare_yes_is_not_enough() -> None:
    """"yes" answering a *different* question must not authorise this one."""
    outcome = classify_voice_confirmation("yes", _request())
    assert outcome.approved is False
    assert "ambiguous" in outcome.reason


def test_yes_naming_the_action_is_accepted() -> None:
    outcome = classify_voice_confirmation("yes, delete everything", _request())
    assert outcome.approved is True
    assert outcome.factor == "voice"


def test_the_tool_name_itself_is_enough() -> None:
    outcome = classify_voice_confirmation("yes delete_everything", _request())
    assert outcome.approved is True


@pytest.mark.parametrize("reply", ["no", "nope", "cancel", "stop", "never mind"])
def test_decline_words_are_honoured(reply: str) -> None:
    outcome = classify_voice_confirmation(reply, _request())
    assert outcome.approved is False
    assert "declined" in outcome.reason or "clear yes or no" in outcome.reason


def test_an_unclear_reply_is_not_approval() -> None:
    outcome = classify_voice_confirmation("what were you saying", _request())
    assert outcome.approved is False


def test_low_confidence_is_treated_as_no() -> None:
    """Mis-hearing must re-ask, never approve."""
    outcome = classify_voice_confirmation(
        "yes, delete everything", _request(), confidence=0.2
    )
    assert outcome.approved is False
    assert "clearly enough" in outcome.reason


def test_an_empty_reply_is_not_approval() -> None:
    assert classify_voice_confirmation("   ", _request()).approved is False


def test_punctuation_and_case_do_not_matter() -> None:
    assert classify_voice_confirmation("YES, DELETE EVERYTHING!", _request()).approved is True


def test_only_valid_factors_count() -> None:
    assert ConfirmationOutcome.approve("voice").is_valid_factor
    assert ConfirmationOutcome.approve("ui").is_valid_factor
    assert ConfirmationOutcome.approve("rule").is_valid_factor
    assert ConfirmationOutcome.approve("typed").is_valid_factor
    assert not ConfirmationOutcome.approve("telepathy").is_valid_factor
    assert not ConfirmationOutcome.decline().is_valid_factor


def test_a_decline_carries_no_valid_factor() -> None:
    outcome = ConfirmationOutcome.decline("because")
    assert outcome.approved is False
    assert outcome.reason == "because"
    assert outcome.factor == "none"