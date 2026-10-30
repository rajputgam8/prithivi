"""Session and transcript persistence.

A *session* is one conversation. Its *messages* form a transcript; a
*summary* row holds the rolling digest that keeps long sessions inside the
model's context budget.

Writes go through :meth:`SessionRepository.append_message`, which is the only
place a message row is created, so ``created_at`` and the id ordering are
always consistent.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from ...db.connection import Database
from ...db.ids import new_id

__all__ = ["MessageRecord", "SessionRecord", "SessionRepository"]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


class SessionRecord:
    """One conversation."""

    __slots__ = ("ended_at", "id", "metadata", "modality", "started_at", "summary_id", "title")

    def __init__(
        self,
        *,
        id: str,
        title: str | None,
        modality: str,
        started_at: str,
        ended_at: str | None,
        summary_id: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.title = title
        self.modality = modality
        self.started_at = started_at
        self.ended_at = ended_at
        self.summary_id = summary_id
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "modality": self.modality,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary_id": self.summary_id,
            "metadata": self.metadata,
        }


class MessageRecord:
    """One message in a transcript."""

    __slots__ = (
        "content",
        "created_at",
        "id",
        "latency_ms",
        "modality",
        "parent_id",
        "provenance",
        "redacted",
        "role",
        "session_id",
        "tokens_in",
        "tokens_out",
    )

    def __init__(
        self,
        *,
        id: str,
        session_id: str,
        parent_id: str | None,
        role: str,
        content: str,
        modality: str,
        provenance: list[Any],
        tokens_in: int | None,
        tokens_out: int | None,
        latency_ms: int | None,
        redacted: bool,
        created_at: str,
    ) -> None:
        self.id = id
        self.session_id = session_id
        self.parent_id = parent_id
        self.role = role
        self.content = content
        self.modality = modality
        self.provenance = provenance
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_ms = latency_ms
        self.redacted = redacted
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "role": self.role,
            "content": self.content,
            "modality": self.modality,
            "provenance": self.provenance,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "redacted": self.redacted,
            "created_at": self.created_at,
        }


class SessionRepository:
    """CRUD over ``sessions``, ``messages`` and ``conversations_summary``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- sessions ------------------------------------------------------
    def create_session(
        self,
        *,
        title: str | None = None,
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionRecord:
        sid = session_id or new_id("ses_")
        now = _now()
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, modality, started_at, ended_at, summary_id, metadata_json)"
                " VALUES (?,?,?,?,NULL,NULL,?)",
                (sid, title, modality, now, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)),
            )
        session = self.get_session(sid)
        assert session is not None
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._db.query_one("SELECT * FROM sessions WHERE id=?", (session_id,))
        return self._to_session(row) if row is not None else None

    def list_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        rows = self._db.query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (int(limit),))
        return [self._to_session(row) for row in rows]

    def end_session(self, session_id: str) -> bool:
        with self._db.tx() as conn:
            cur = conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (_now(), session_id))
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._db.tx() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            return cur.rowcount > 0

    # -- messages ------------------------------------------------------
    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        parent_id: str | None = None,
        modality: str = "text",
        provenance: list[Any] | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
        redacted: bool = False,
    ) -> MessageRecord:
        """Append a message to a transcript."""
        message_id = new_id("msg_")
        now = _now()
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, parent_id, role, content, modality,"
                " provenance_json, tokens_in, tokens_out, latency_ms, redacted, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    message_id,
                    session_id,
                    parent_id,
                    role,
                    content,
                    modality,
                    json.dumps(provenance or [], ensure_ascii=False, default=str),
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    int(redacted),
                    now,
                ),
            )
        message = self.get_message(message_id)
        assert message is not None
        return message

    def get_message(self, message_id: str) -> MessageRecord | None:
        row = self._db.query_one("SELECT * FROM messages WHERE id=?", (message_id,))
        return self._to_message(row) if row is not None else None

    def list_messages(self, session_id: str, *, limit: int = 500) -> list[MessageRecord]:
        rows = self._db.query(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at, id LIMIT ?",
            (session_id, int(limit)),
        )
        return [self._to_message(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,))
        return int(row["n"]) if row is not None else 0

    # -- summaries -----------------------------------------------------
    def save_summary(
        self,
        *,
        session_id: str,
        upto_message_id: str,
        summary: str,
        key_points: list[str] | None = None,
        embedding_id: str | None = None,
    ) -> None:
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO conversations_summary (session_id, upto_message_id, summary,"
                " key_points_json, embedding_id, updated_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(session_id) DO UPDATE SET upto_message_id=excluded.upto_message_id,"
                " summary=excluded.summary, key_points_json=excluded.key_points_json,"
                " embedding_id=excluded.embedding_id, updated_at=excluded.updated_at",
                (
                    session_id,
                    upto_message_id,
                    summary,
                    json.dumps(key_points or [], ensure_ascii=False),
                    embedding_id,
                    _now(),
                ),
            )

    def get_summary(self, session_id: str) -> dict[str, Any] | None:
        row = self._db.query_one("SELECT * FROM conversations_summary WHERE session_id=?", (session_id,))
        if row is None:
            return None
        try:
            key_points = json.loads(row["key_points_json"])
        except (json.JSONDecodeError, TypeError):
            key_points = []
        return {
            "session_id": str(row["session_id"]),
            "upto_message_id": str(row["upto_message_id"]),
            "summary": str(row["summary"]),
            "key_points": key_points if isinstance(key_points, list) else [],
            "embedding_id": str(row["embedding_id"]) if row["embedding_id"] is not None else None,
            "updated_at": str(row["updated_at"]),
        }

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _to_session(row: Any) -> SessionRecord:
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return SessionRecord(
            id=str(row["id"]),
            title=str(row["title"]) if row["title"] is not None else None,
            modality=str(row["modality"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
            summary_id=str(row["summary_id"]) if row["summary_id"] is not None else None,
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    @staticmethod
    def _to_message(row: Any) -> MessageRecord:
        try:
            provenance = json.loads(row["provenance_json"])
        except (json.JSONDecodeError, TypeError):
            provenance = []
        return MessageRecord(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
            role=str(row["role"]),
            content=str(row["content"]),
            modality=str(row["modality"]),
            provenance=provenance if isinstance(provenance, list) else [],
            tokens_in=int(row["tokens_in"]) if row["tokens_in"] is not None else None,
            tokens_out=int(row["tokens_out"]) if row["tokens_out"] is not None else None,
            latency_ms=int(row["latency_ms"]) if row["latency_ms"] is not None else None,
            redacted=bool(row["redacted"]),
            created_at=str(row["created_at"]),
        )
