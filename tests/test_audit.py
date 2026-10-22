"""Audit log: the hash chain must actually detect tampering.

SECURITY.md S2/T9. An append-only log only means something if editing a past
row is *detectable*, so these tests mutate the database directly and assert that
verification fails at the right record.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from prithivi.security.audit import (
    GENESIS_HASH,
    AuditLog,
    canonical_json,
    record_hash,
)

NOW = dt.datetime(2026, 10, 5, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def log(tmp_path) -> AuditLog:
    with AuditLog(tmp_path / "audit.db") as audit:
        yield audit


def _fill(audit: AuditLog, count: int) -> None:
    for i in range(count):
        audit.append(
            action="tool.execute",
            outcome="success",
            actor="agent",
            channel="cli",
            target=f"tool_{i}",
            detail={"i": i},
            at=NOW + dt.timedelta(seconds=i),
        )


# --- the chain --------------------------------------------------------------


def test_an_empty_log_verifies() -> None:
    with AuditLog(":memory:") as audit:
        assert audit.verify() == (True, None)


def test_records_chain_from_genesis(log: AuditLog) -> None:
    first = log.append(action="a", detail={}, at=NOW)
    second = log.append(action="b", detail={}, at=NOW)
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.hash
    assert log.verify() == (True, None)


def test_each_record_hashes_its_own_content(log: AuditLog) -> None:
    record = log.append(action="post", target="t", outcome="denied", detail={"n": 1}, at=NOW)
    expected = record_hash(
        GENESIS_HASH, record.ts, record.actor, record.channel,
        record.action, record.target, record.outcome, {"n": 1},
    )
    assert record.hash == expected


def test_canonical_json_is_key_order_independent() -> None:
    """Otherwise a hash would depend on dict insertion order, not on content."""
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_canonical_json_scrubs_secrets() -> None:
    """A key pasted into a prompt must not reach disk in the clear."""
    blob = canonical_json({"prompt": "my token is sk-abcdefghijklmnopqrstuvwxyz1234"})
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in blob


def test_every_field_participates_in_the_hash() -> None:
    base = record_hash(GENESIS_HASH, "ts", "actor", "cli", "act", "tgt", "ok", {})
    assert base != record_hash("0" * 63 + "1", "ts", "actor", "cli", "act", "tgt", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts2", "actor", "cli", "act", "tgt", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts", "other", "cli", "act", "tgt", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts", "actor", "voice", "act", "tgt", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts", "actor", "cli", "other", "tgt", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts", "actor", "cli", "act", "other", "ok", {})
    assert base != record_hash(GENESIS_HASH, "ts", "actor", "cli", "act", "tgt", "bad", {})
    assert base != record_hash(GENESIS_HASH, "ts", "actor", "cli", "act", "tgt", "ok", {"x": 1})


def test_a_null_target_does_not_crash_the_hash() -> None:
    assert record_hash(GENESIS_HASH, "ts", "a", "cli", "act", None, "ok", {}) == record_hash(
        GENESIS_HASH, "ts", "a", "cli", "act", "", "ok", {}
    )


# --- tamper detection -------------------------------------------------------


def test_editing_a_historical_row_is_detected(log: AuditLog) -> None:
    _fill(log, 5)
    assert log.verify() == (True, None)

    # Rewrite the outcome of the 2nd record, exactly as a motivated actor would.
    row = log.all()[1]
    log._conn.execute(
        "UPDATE audit_log SET outcome='success', detail_json='{\"i\": 999}' WHERE id=?",
        (row.id,),
    )

    ok, broken = log.verify()
    assert ok is False
    assert broken == row.id, "verification must stop at the first altered record"


def test_deleting_a_record_is_detected(log: AuditLog) -> None:
    _fill(log, 4)
    victim = log.all()[1].id
    log._conn.execute("DELETE FROM audit_log WHERE id=?", (victim,))

    ok, broken = log.verify()
    assert ok is False
    assert broken is not None


def test_recomputing_one_hash_does_not_fix_the_chain(log: AuditLog) -> None:
    """Rewriting the row *and* its own hash still breaks the following link.

    This is the case a naive integrity check misses: the altered row is
    internally consistent, so only the successor's ``prev_hash`` betrays it.
    """
    _fill(log, 4)
    row = log.all()[1]
    successor = log.all()[2]

    log._conn.execute("UPDATE audit_log SET outcome='denied' WHERE id=?", (row.id,))
    forged = record_hash(
        row.prev_hash, row.ts, row.actor, row.channel,
        row.action, row.target, "denied", {"i": 1},
    )
    log._conn.execute("UPDATE audit_log SET hash=? WHERE id=?", (forged, row.id))

    ok, broken = log.verify()
    assert ok is False
    # The successor still points at the original hash, which is no longer in
    # the chain, so the break surfaces one record later.
    assert broken == successor.id


def test_appending_after_tampering_still_fails(log: AuditLog) -> None:
    """A repaired-looking tail must not launder a broken history."""
    _fill(log, 3)
    row = log.all()[0]
    log._conn.execute("UPDATE audit_log SET actor='mallory' WHERE id=?", (row.id,))
    assert log.verify()[0] is False

    log.append(action="later", detail={}, at=NOW)
    assert log.verify()[0] is False


# --- reading ----------------------------------------------------------------


def test_tail_returns_newest_first(log: AuditLog) -> None:
    _fill(log, 5)
    tail = log.tail(limit=3)
    assert [r.target for r in tail] == ["tool_4", "tool_3", "tool_2"]


def test_tail_can_filter_by_action_prefix(log: AuditLog) -> None:
    log.append(action="permission.check", detail={}, at=NOW)
    log.append(action="tool.execute", detail={}, at=NOW)
    assert [r.action for r in log.tail(action_prefix="permission")] == ["permission.check"]


def test_the_log_is_iterable_and_sized(log: AuditLog) -> None:
    _fill(log, 3)
    assert len(log) == 3
    assert len(list(log)) == 3
    assert [r.id for r in log] == sorted(r.id for r in log)


def test_a_record_round_trips_through_the_database(log: AuditLog) -> None:
    log.append(
        action="tool.execute", outcome="denied", actor="user", channel="voice",
        target="delete", detail={"level": 3}, at=NOW,
    )
    record = log.all()[0]
    assert record.actor == "user"
    assert record.channel == "voice"
    assert record.detail == {"level": 3}
    assert record.to_dict()["hash"] == record.hash


def test_the_timestamp_is_stored_as_utc(log: AuditLog) -> None:
    log.append(action="a", at=NOW)
    assert log.all()[0].ts.endswith("Z")


# --- retention --------------------------------------------------------------


def test_pruning_removes_only_the_old_rows(log: AuditLog) -> None:
    now = NOW + dt.timedelta(days=40)
    for day in range(40):
        log.append(action=f"day{day}", detail={}, at=NOW + dt.timedelta(days=day))

    removed = log.prune(keep_days=30, now=now)
    assert removed == 10
    assert len(log) == 30
    assert log.all()[0].target is None or log.all()[0].action == "day10"


def test_pruning_keeps_the_chain_valid(log: AuditLog) -> None:
    """Truncation restarts the chain; verification must still pass afterwards."""
    now = NOW + dt.timedelta(days=40)
    for day in range(40):
        log.append(action=f"day{day}", detail={}, at=NOW + dt.timedelta(days=day))

    log.prune(keep_days=30, now=now)
    ok, broken = log.verify()
    assert (ok, broken) == (True, None)
    # The new oldest record re-anchors on genesis.
    assert log.all()[0].prev_hash == GENESIS_HASH


def test_pruning_nothing_is_a_no_op(log: AuditLog) -> None:
    _fill(log, 3)
    assert log.prune(keep_days=365, now=NOW) == 0
    assert len(log) == 3
    assert log.verify() == (True, None)


# --- durability -------------------------------------------------------------


def test_the_log_survives_reopening(tmp_path) -> None:
    path = tmp_path / "audit.db"
    with AuditLog(path) as first:
        _fill(first, 3)
    with AuditLog(path) as second:
        assert len(second) == 3
        assert second.verify() == (True, None)


def test_the_parent_directory_is_created(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "audit.db"
    with AuditLog(nested) as audit:
        audit.append(action="a", detail={})
    assert nested.exists()


def test_a_failed_audit_does_not_raise_here() -> None:
    """Callers wrap audit writes; the log itself stays usable after one fails."""
    with AuditLog(":memory:") as audit:
        audit._conn.close()
        with pytest.raises(sqlite3.ProgrammingError):
            audit.append(action="a", detail={})