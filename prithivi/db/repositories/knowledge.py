"""Document and chunk persistence for RAG.

A *document* is one ingested file; a *chunk* is a slice of its text with a
locator (``page 3``, ``#L42``) that an answer can cite.

Ingestion is transactional at the repository level: a document and its chunks
are written in one transaction, so a document with chunks but no vector — or
vice versa — cannot exist. That property is what lets the knowledge base claim
a document is "fully indexed" without a second reconciliation pass.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from ...db.connection import Database
from ...db.ids import new_id

__all__ = ["ChunkRecord", "DocumentRecord", "KnowledgeRepository"]

log = logging.getLogger("prithivi.db.knowledge")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


class DocumentRecord:
    """One ingested document."""

    __slots__ = (
        "byte_size",
        "id",
        "indexed_at",
        "ingested_at",
        "meta",
        "page_count",
        "sha256",
        "source_type",
        "status",
        "title",
        "uri",
    )

    def __init__(
        self,
        *,
        id: str,
        uri: str,
        title: str | None,
        source_type: str,
        sha256: str,
        byte_size: int | None,
        page_count: int | None,
        meta: dict[str, Any],
        ingested_at: str,
        indexed_at: str | None,
        status: str,
    ) -> None:
        self.id = id
        self.uri = uri
        self.title = title
        self.source_type = source_type
        self.sha256 = sha256
        self.byte_size = byte_size
        self.page_count = page_count
        self.meta = meta
        self.ingested_at = ingested_at
        self.indexed_at = indexed_at
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "uri": self.uri,
            "title": self.title,
            "source_type": self.source_type,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "page_count": self.page_count,
            "meta": self.meta,
            "ingested_at": self.ingested_at,
            "indexed_at": self.indexed_at,
            "status": self.status,
        }


class ChunkRecord:
    """One retrievable slice of a document."""

    __slots__ = ("document_id", "embedding_id", "id", "locator", "meta", "ordinal", "text", "token_count")

    def __init__(
        self,
        *,
        id: str,
        document_id: str,
        ordinal: int,
        text: str,
        token_count: int,
        locator: str | None,
        meta: dict[str, Any],
        embedding_id: str | None,
    ) -> None:
        self.id = id
        self.document_id = document_id
        self.ordinal = ordinal
        self.text = text
        self.token_count = token_count
        self.locator = locator
        self.meta = meta
        self.embedding_id = embedding_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "token_count": self.token_count,
            "locator": self.locator,
            "meta": self.meta,
            "embedding_id": self.embedding_id,
        }


class KnowledgeRepository:
    """CRUD over ``documents`` and ``chunks``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- documents -----------------------------------------------------
    def upsert_document(
        self,
        *,
        uri: str,
        source_type: str,
        sha256: str,
        title: str | None = None,
        byte_size: int | None = None,
        page_count: int | None = None,
        meta: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> DocumentRecord:
        """Insert a document, or return the existing one with the same hash.

        Re-ingesting an unchanged file is a no-op, which is what makes
        ``ingest`` idempotent: the same file twice does not double the chunks.
        """
        existing = self.get_document_by_hash(sha256)
        if existing is not None:
            return existing
        document_id = new_id("doc_")
        now = _now()
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO documents (id, uri, title, source_type, sha256, byte_size,"
                " page_count, meta_json, ingested_at, indexed_at, status)"
                " VALUES (?,?,?,?,?,?,?,?,?,NULL,?)",
                (
                    document_id,
                    uri,
                    title,
                    source_type,
                    sha256,
                    byte_size,
                    page_count,
                    json.dumps(meta or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    status,
                ),
            )
        document = self.get_document(document_id)
        assert document is not None
        return document

    def set_document_status(self, document_id: str, status: str, *, indexed: bool = False) -> None:
        sql = "UPDATE documents SET status=?"
        params: list[Any] = [status]
        if indexed:
            sql += ", indexed_at=?"
            params.append(_now())
        sql += " WHERE id=?"
        params.append(document_id)
        with self._db.tx() as conn:
            conn.execute(sql, params)

    def update_document(
        self,
        document_id: str,
        *,
        uri: str,
        source_type: str,
        sha256: str,
        title: str | None = None,
        byte_size: int | None = None,
        page_count: int | None = None,
        meta: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> DocumentRecord:
        """Replace the metadata of an existing document in place.

        Used when a file at the same URI changed: the document keeps its id, so
        any citation that pointed at it still resolves, while its old chunks are
        swapped out separately by :meth:`replace_chunks`.
        """
        with self._db.tx() as conn:
            conn.execute(
                "UPDATE documents SET uri=?, title=?, source_type=?, sha256=?, byte_size=?,"
                " page_count=?, meta_json=?, ingested_at=?, indexed_at=NULL, status=?"
                " WHERE id=?",
                (
                    uri,
                    title,
                    source_type,
                    sha256,
                    byte_size,
                    page_count,
                    json.dumps(meta or {}, ensure_ascii=False, sort_keys=True),
                    _now(),
                    status,
                    document_id,
                ),
            )
        document = self.get_document(document_id)
        assert document is not None
        return document

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self._db.query_one("SELECT * FROM documents WHERE id=?", (document_id,))
        return self._to_document(row) if row is not None else None

    def get_document_by_hash(self, sha256: str) -> DocumentRecord | None:
        row = self._db.query_one("SELECT * FROM documents WHERE sha256=?", (sha256,))
        return self._to_document(row) if row is not None else None

    def get_document_by_uri(self, uri: str) -> DocumentRecord | None:
        row = self._db.query_one("SELECT * FROM documents WHERE uri=? ORDER BY ingested_at DESC LIMIT 1", (uri,))
        return self._to_document(row) if row is not None else None

    def list_documents(self, *, limit: int = 100) -> list[DocumentRecord]:
        rows = self._db.query("SELECT * FROM documents ORDER BY ingested_at DESC LIMIT ?", (int(limit),))
        return [self._to_document(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks (chunks cascade)."""
        with self._db.tx() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            return cur.rowcount > 0

    # -- chunks --------------------------------------------------------
    def replace_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> list[ChunkRecord]:
        """Swap a document's chunks for a new set, in one transaction.

        ``chunks`` items need ``text`` and ``token_count``; ``ordinal``,
        ``locator`` and ``meta`` are optional.
        """
        with self._db.tx() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            for index, item in enumerate(chunks):
                chunk_id = new_id("chk_")
                conn.execute(
                    "INSERT INTO chunks (id, document_id, ordinal, text, token_count, locator, meta_json, embedding_id)"
                    " VALUES (?,?,?,?,?,?,?,NULL)",
                    (
                        chunk_id,
                        document_id,
                        int(item.get("ordinal", index)),
                        str(item["text"]),
                        int(item.get("token_count", 0)),
                        item.get("locator"),
                        json.dumps(item.get("meta", {}), ensure_ascii=False, sort_keys=True),
                    ),
                )
        return self.list_chunks(document_id)

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        rows = self._db.query("SELECT * FROM chunks WHERE document_id=? ORDER BY ordinal", (document_id,))
        return [self._to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        row = self._db.query_one("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        return self._to_chunk(row) if row is not None else None

    def iter_all_chunks(self) -> list[ChunkRecord]:
        rows = self._db.query("SELECT * FROM chunks ORDER BY document_id, ordinal")
        return [self._to_chunk(row) for row in rows]

    def set_chunk_embedding(self, chunk_id: str, embedding_id: str | None) -> None:
        with self._db.tx() as conn:
            conn.execute("UPDATE chunks SET embedding_id=? WHERE id=?", (embedding_id, chunk_id))

    def search_text(self, query: str, *, limit: int = 20) -> list[ChunkRecord]:
        """Full-text search over chunk text, with a LIKE fallback."""
        terms = [t for t in _tokenise(query) if t]
        if not terms:
            return []
        try:
            rows = self._db.query(
                "SELECT c.* FROM chunk_fts f JOIN chunks c ON c.rowid=f.rowid"
                " WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT ?",
                (" OR ".join(terms), int(limit)),
            )
            if rows:
                return [self._to_chunk(row) for row in rows]
        except Exception:  # FTS5 missing or malformed query
            log.debug("chunk FTS query failed; falling back to LIKE", exc_info=True)
        like = "%" + "%".join(terms) + "%"
        rows = self._db.query(
            "SELECT * FROM chunks WHERE text LIKE ? ORDER BY ordinal LIMIT ?",
            (like, int(limit)),
        )
        return [self._to_chunk(row) for row in rows]

    def count_chunks(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM chunks")
        return int(row["n"]) if row is not None else 0

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _to_document(row: Any) -> DocumentRecord:
        try:
            meta = json.loads(row["meta_json"])
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return DocumentRecord(
            id=str(row["id"]),
            uri=str(row["uri"]),
            title=str(row["title"]) if row["title"] is not None else None,
            source_type=str(row["source_type"]),
            sha256=str(row["sha256"]),
            byte_size=int(row["byte_size"]) if row["byte_size"] is not None else None,
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
            meta=meta if isinstance(meta, dict) else {},
            ingested_at=str(row["ingested_at"]),
            indexed_at=str(row["indexed_at"]) if row["indexed_at"] is not None else None,
            status=str(row["status"]),
        )

    @staticmethod
    def _to_chunk(row: Any) -> ChunkRecord:
        try:
            meta = json.loads(row["meta_json"])
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return ChunkRecord(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            token_count=int(row["token_count"]),
            locator=str(row["locator"]) if row["locator"] is not None else None,
            meta=meta if isinstance(meta, dict) else {},
            embedding_id=str(row["embedding_id"]) if row["embedding_id"] is not None else None,
        )


def _tokenise(text: str) -> list[str]:
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
