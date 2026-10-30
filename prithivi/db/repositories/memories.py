"""Long-term memory persistence.

The ``memories`` table is the durable record of what Prithivi has been told and
found. Three rules govern it:

* **Soft delete.** ``deleted_at`` marks a memory gone; the row survives so the
  FTS index and the audit trail stay consistent. Hard deletes happen only when
  a user explicitly asks for a purge.
* **One live row per key.** A partial unique index enforces it, so "remember I
  prefer Python" twice updates the same row rather than accumulating
  contradictory duplicates.
* **Provenance is mandatory.** Every row records where the memory came from,
  so an answer can always say *why* it believes something.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from ...db.connection import Database
from ...db.ids import new_id

__all__ = ["MEMORY_TYPES", "MemoryRecord", "MemoryRepository"]

log = logging.getLogger("prithivi.db.memories")


class MEMORY_TYPES:
    """The kinds of memory Prithivi stores."""

    PREFERENCE = "preference"
    """How the user wants things done — 'always answer in Hindi'."""

    FACT = "fact"
    """A durable piece of knowledge about the user or their world."""

    PROCEDURE = "procedure"
    """How to do something the user does repeatedly."""

    PROFILE = "profile"
    """Stable background — name, timezone, occupation."""

    EVENT = "event"
    """Something that happened at a specific time."""

    INSIGHT = "insight"
    """A conclusion Prithivi derived and the user did not correct."""


_ALL_TYPES = (
    MEMORY_TYPES.PREFERENCE,
    MEMORY_TYPES.FACT,
    MEMORY_TYPES.PROCEDURE,
    MEMORY_TYPES.PROFILE,
    MEMORY_TYPES.EVENT,
    MEMORY_TYPES.INSIGHT,
)


class MemoryRecord:
    """One stored memory."""

    __slots__ = (
        "confidence",
        "created_at",
        "deleted_at",
        "embedding_id",
        "hits",
        "id",
        "key",
        "last_used_at",
        "pinned",
        "source",
        "source_ref",
        "text",
        "type",
        "updated_at",
        "value",
    )

    def __init__(
        self,
        *,
        id: str,
        type: str,
        key: str,
        value: Any,
        text: str,
        source: str,
        source_ref: str | None,
        confidence: float,
        pinned: bool,
        hits: int,
        embedding_id: str | None,
        created_at: str,
        updated_at: str,
        last_used_at: str | None,
        deleted_at: str | None,
    ) -> None:
        self.id = id
        self.type = type
        self.key = key
        self.value = value
        self.text = text
        self.source = source
        self.source_ref = source_ref
        self.confidence = confidence
        self.pinned = pinned
        self.hits = hits
        self.embedding_id = embedding_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.last_used_at = last_used_at
        self.deleted_at = deleted_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "text": self.text,
            "source": self.source,
            "source_ref": self.source_ref,
            "confidence": self.confidence,
            "pinned": self.pinned,
            "hits": self.hits,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "deleted_at": self.deleted_at,
        }


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


class MemoryRepository:
    """CRUD over the ``memories`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- writes --------------------------------------------------------
    def remember(
        self,
        *,
        key: str,
        text: str,
        type: str = MEMORY_TYPES.FACT,
        value: Any = None,
        source: str = "user",
        source_ref: str | None = None,
        confidence: float = 0.7,
        pinned: bool = False,
    ) -> MemoryRecord:
        """Store a memory, replacing any live row with the same key.

        Returns the stored record. An existing memory keeps its id and creation
        time but takes the new text and value, so the FTS trigger sees an
        update rather than a delete+insert pair.
        """
        if type not in _ALL_TYPES:
            raise ValueError(f"unknown memory type {type!r}; expected one of {', '.join(_ALL_TYPES)}")
        if not key.strip():
            raise ValueError("memory key must not be empty")
        now = _now()
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        existing = self.get_by_key(key)
        if existing is not None:
            with self._db.tx() as conn:
                conn.execute(
                    "UPDATE memories SET type=?, value_json=?, text=?, source=?, source_ref=?,"
                    " confidence=?, pinned=?, updated_at=? WHERE id=?",
                    (type, payload, text, source, source_ref, float(confidence), int(pinned), now, existing.id),
                )
            return self.get(existing.id)  # type: ignore[return-value]

        memory_id = new_id("mem_")
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO memories (id, type, key, value_json, text, source, source_ref,"
                " confidence, pinned, hits, embedding_id, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,NULL,?,?)",
                (memory_id, type, key, payload, text, source, source_ref, float(confidence), int(pinned), now, now),
            )
        return self.get(memory_id)  # type: ignore[return-value]

    def forget(self, key: str) -> bool:
        """Soft-delete the live memory with this key. Returns True if one went."""
        existing = self.get_by_key(key)
        if existing is None:
            return False
        with self._db.tx() as conn:
            conn.execute("UPDATE memories SET deleted_at=? WHERE id=?", (_now(), existing.id))
        return True

    def purge(self, key: str) -> bool:
        """Hard-delete a memory and its row entirely."""
        existing = self.get_by_key(key)
        if existing is None:
            return False
        with self._db.tx() as conn:
            conn.execute("DELETE FROM memories WHERE id=?", (existing.id,))
        return True

    def clear(self, *, include_pinned: bool = False) -> int:
        """Soft-delete every memory. Returns how many were removed."""
        sql = "UPDATE memories SET deleted_at=? WHERE deleted_at IS NULL"
        params: list[Any] = [_now()]
        if not include_pinned:
            sql += " AND pinned=0"
        with self._db.tx() as conn:
            cur = conn.execute(sql, params)
            return int(cur.rowcount)

    def touch(self, memory_id: str) -> None:
        """Record that a memory was used: bump hits and ``last_used_at``."""
        with self._db.tx() as conn:
            conn.execute(
                "UPDATE memories SET hits=hits+1, last_used_at=? WHERE id=?",
                (_now(), memory_id),
            )

    def set_embedding(self, memory_id: str, embedding_id: str | None) -> None:
        with self._db.tx() as conn:
            conn.execute("UPDATE memories SET embedding_id=? WHERE id=?", (embedding_id, memory_id))

    # -- reads ---------------------------------------------------------
    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._db.query_one("SELECT * FROM memories WHERE id=?", (memory_id,))
        return self._to_record(row) if row is not None else None

    def get_by_key(self, key: str) -> MemoryRecord | None:
        row = self._db.query_one(
            "SELECT * FROM memories WHERE key=? AND deleted_at IS NULL",
            (key,),
        )
        return self._to_record(row) if row is not None else None

    def list(
        self,
        *,
        type: str | None = None,
        limit: int = 50,
        include_deleted: bool = False,
        pinned_only: bool = False,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        if type is not None:
            sql += " AND type=?"
            params.append(type)
        if pinned_only:
            sql += " AND pinned=1"
        sql += " ORDER BY pinned DESC, confidence DESC, updated_at DESC LIMIT ?"
        params.append(int(limit))
        return [self._to_record(row) for row in self._db.query(sql, params)]

    def search_text(self, query: str, *, limit: int = 20) -> list[MemoryRecord]:
        """Full-text search over memory text.

        Uses FTS5 when it is available and falls back to a LIKE scan otherwise,
        because some SQLite builds ship without FTS5 and a memory store that
        cannot be searched is not a memory store.
        """
        terms = [t for t in _tokenise(query) if t]
        if not terms:
            return []
        try:
            rows = self._db.query(
                "SELECT m.* FROM memory_fts f JOIN memories m ON m.rowid=f.rowid"
                " WHERE memory_fts MATCH ? AND m.deleted_at IS NULL"
                " ORDER BY bm25(memory_fts) LIMIT ?",
                (" OR ".join(terms), int(limit)),
            )
            if rows:
                return [self._to_record(row) for row in rows]
        except Exception:  # FTS5 missing or malformed query
            log.debug("memory FTS query failed; falling back to LIKE", exc_info=True)
        like = "%" + "%".join(terms) + "%"
        rows = self._db.query(
            "SELECT * FROM memories WHERE deleted_at IS NULL AND (text LIKE ? OR key LIKE ?)"
            " ORDER BY pinned DESC, confidence DESC LIMIT ?",
            (like, like, int(limit)),
        )
        return [self._to_record(row) for row in rows]

    def count(self, *, include_deleted: bool = False) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        row = self._db.query_one(sql)
        return int(row["n"]) if row is not None else 0

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _to_record(row: Any) -> MemoryRecord:
        try:
            value = json.loads(row["value_json"])
        except (json.JSONDecodeError, TypeError):
            value = None
        return MemoryRecord(
            id=str(row["id"]),
            type=str(row["type"]),
            key=str(row["key"]),
            value=value,
            text=str(row["text"]),
            source=str(row["source"]),
            source_ref=str(row["source_ref"]) if row["source_ref"] is not None else None,
            confidence=float(row["confidence"]),
            pinned=bool(row["pinned"]),
            hits=int(row["hits"]),
            embedding_id=str(row["embedding_id"]) if row["embedding_id"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_used_at=str(row["last_used_at"]) if row["last_used_at"] is not None else None,
            deleted_at=str(row["deleted_at"]) if row["deleted_at"] is not None else None,
        )


def _tokenise(text: str) -> list[str]:
    """Split into lowercase alphanumeric tokens for FTS matching."""
    out: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out
