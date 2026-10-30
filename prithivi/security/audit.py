"""Append-only, hash-chained audit log (SECURITY.md S2/T9).

Every routed intent, tool call, permission decision, and confirmation is written
here. Tampering is detectable: each record's ``hash`` is
``sha256(prev_hash + canonical_json(record))``, so altering any historical row
breaks verification for every row after it.

Storage is stdlib :mod:`sqlite3` in WAL mode — no compiled extension, which
matters because the target device is aarch64-Android (S19).

Rules enforced here:

* there is no UPDATE or DELETE code path for this table;
* :meth:`AuditLog.verify` is the only way to trust the log;
* payloads pass through the :mod:`~prithivi.security.secrets` redactor on the
  way in, so a key pasted into a prompt is never persisted in the clear.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .secrets import redactor

__all__ = ["GENESIS_HASH", "AuditLog", "AuditRecord", "canonical_json", "record_hash"]

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    actor       TEXT    NOT NULL,
    channel     TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    target      TEXT,
    outcome     TEXT    NOT NULL,
    detail_json TEXT    NOT NULL DEFAULT '{}',
    prev_hash   TEXT    NOT NULL,
    hash        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
"""


def canonical_json(value: Any) -> str:
    """Stable JSON so a record's hash is reproducible across processes."""
    return json.dumps(redactor.scrub_obj(value), sort_keys=True, separators=(",", ":"), default=str)


def record_hash(prev_hash: str, ts: str, actor: str, channel: str, action: str,
                target: str | None, outcome: str, detail: Any) -> str:
    """The chain function. Changing any input changes the digest."""
    payload = "|".join([
        prev_hash, ts, actor, channel, action, target or "", outcome, canonical_json(detail),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    id: int
    ts: str
    actor: str
    channel: str
    action: str
    target: str | None
    outcome: str
    detail: dict[str, Any]
    prev_hash: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "ts": self.ts, "actor": self.actor, "channel": self.channel,
            "action": self.action, "target": self.target, "outcome": self.outcome,
            "detail": self.detail, "hash": self.hash,
        }


class AuditLog:
    """Thread-safe append-only log with a verifiable hash chain.

    Args:
        path: SQLite file. ``":memory:"`` is supported for tests.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # -- writing -------------------------------------------------------
    def append(
        self,
        *,
        action: str,
        outcome: str = "success",
        actor: str = "agent",
        channel: str = "cli",
        target: str | None = None,
        detail: dict[str, Any] | None = None,
        at: dt.datetime | None = None,
    ) -> AuditRecord:
        """Append one record and return it, including the computed chain hash."""
        detail = detail or {}
        ts = (at or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
        with self._lock:
            row = self._conn.execute(
                "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = row["hash"] if row else GENESIS_HASH
            digest = record_hash(prev_hash, ts, actor, channel, action, target, outcome, detail)
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, actor, channel, action, target, outcome,"
                " detail_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, actor, channel, action, target, outcome, canonical_json(detail), prev_hash, digest),
            )
            return AuditRecord(
                id=int(cur.lastrowid or 0), ts=ts, actor=actor, channel=channel, action=action,
                target=target, outcome=outcome, detail=detail, prev_hash=prev_hash, hash=digest,
            )

    # -- reading -------------------------------------------------------
    def tail(self, limit: int = 50, *, action_prefix: str | None = None) -> list[AuditRecord]:
        """Most recent records first."""
        sql = "SELECT * FROM audit_log"
        params: list[Any] = []
        if action_prefix:
            sql += " WHERE action LIKE ?"
            params.append(f"{action_prefix}%")
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self._conn.execute(sql, params)]

    def all(self) -> list[AuditRecord]:
        """Every record in insertion order."""
        return [self._row(r) for r in self._conn.execute("SELECT * FROM audit_log ORDER BY id ASC")]

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self.all())

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])

    @staticmethod
    def _row(row: sqlite3.Row) -> AuditRecord:
        return AuditRecord(
            id=int(row["id"]), ts=row["ts"], actor=row["actor"], channel=row["channel"],
            action=row["action"], target=row["target"], outcome=row["outcome"],
            detail=json.loads(row["detail_json"]), prev_hash=row["prev_hash"], hash=row["hash"],
        )

    # -- integrity -----------------------------------------------------
    def verify(self) -> tuple[bool, int | None]:
        """Recompute the whole chain.

        Returns:
            ``(ok, first_bad_id)``. ``first_bad_id`` is ``None`` when intact.
        """
        prev = GENESIS_HASH
        for record in self.all():
            expected = record_hash(
                prev, record.ts, record.actor, record.channel, record.action,
                record.target, record.outcome, record.detail,
            )
            if record.prev_hash != prev or record.hash != expected:
                return False, record.id
            prev = record.hash
        return True, None

    def prune(self, keep_days: int, *, now: dt.datetime | None = None) -> int:
        """Delete records older than ``keep_days``.

        SECURITY.md: the chain may only be truncated *from the end*, so this
        removes the oldest rows and restarts the chain. The number pruned is
        returned so the caller can log that history was dropped.
        """
        now = now or dt.datetime.now(dt.UTC)
        cutoff = (now - dt.timedelta(days=keep_days)).astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
        with self._lock:
            stale = [r["id"] for r in self._conn.execute(
                "SELECT id FROM audit_log WHERE ts < ? ORDER BY id ASC", (cutoff,)
            )]
            if not stale:
                return 0
            placeholders = ",".join("?" * len(stale))
            self._conn.execute(f"DELETE FROM audit_log WHERE id IN ({placeholders})", stale)
            self._rechain()
        return len(stale)

    def _rechain(self) -> None:
        prev = GENESIS_HASH
        for row in self._conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall():
            detail = json.loads(row["detail_json"])
            digest = record_hash(prev, row["ts"], row["actor"], row["channel"], row["action"],
                                 row["target"], row["outcome"], detail)
            self._conn.execute(
                "UPDATE audit_log SET prev_hash=?, hash=? WHERE id=?", (prev, digest, row["id"])
            )
            prev = digest

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
