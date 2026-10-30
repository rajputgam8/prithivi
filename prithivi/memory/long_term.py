"""Long-term memory.

The user-facing memory API. It owns three things:

* the durable rows (:class:`~prithivi.db.repositories.memories.MemoryRepository`),
* the vectors that make memory searchable by meaning
  (:class:`~prithivi.memory.vector_store.VectorStore`),
* the *recall* policy, which is where most of the judgement lives.

Recall is hybrid. A pure vector search misses exact tokens — "the deploy key is
in the vault" is found by keyword, not by similarity. A pure keyword search
misses paraphrases. So both are run and fused with Reciprocal Rank Fusion,
and the result is what the agent sees.

Every recalled memory is returned with its provenance, so an answer can say
*"I remember this because you told me on Tuesday"*. A memory with no source is
not surfaced: it cannot be explained, and an unexplainable memory is a
liability.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db.connection import Database
from ..db.repositories.embeddings import rrf
from ..db.repositories.memories import MEMORY_TYPES, MemoryRecord, MemoryRepository
from ..rag.embeddings import Embedder, HashEmbedder
from .vector_store import VectorStore

__all__ = ["MemoryStore", "RecalledMemory"]

log = logging.getLogger("prithivi.memory")


class RecalledMemory:
    """A memory returned by :meth:`MemoryStore.recall`."""

    __slots__ = ("record", "score", "source")

    def __init__(self, record: MemoryRecord, score: float, source: str) -> None:
        self.record = record
        self.score = score
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        data = self.record.to_dict()
        data["score"] = round(self.score, 6)
        data["match"] = self.source
        return data

    def describe(self) -> str:
        """One line for the narrator: 'I remember X (because Y)'."""
        record = self.record
        tail = f" (from {record.source_ref})" if record.source_ref else f" (from {record.source})"
        return f"{record.text}{tail}"


class MemoryStore:
    """The memory service the agent talks to.

    Args:
        db: the database.
        embedder: encoder used for both indexing and querying.
    """

    def __init__(self, db: Database, *, embedder: Embedder | None = None, enabled: bool = True) -> None:
        self._db = db
        self._repo = MemoryRepository(db)
        self._vectors = VectorStore(db, embedder=embedder or HashEmbedder())
        self._enabled = bool(enabled)

    # -- configuration -------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Turn memory off entirely.

        When disabled, writes are refused and recalls return nothing — not a
        degraded subset, nothing, so "disable memory" means what it says.
        """
        self._enabled = bool(enabled)

    @property
    def repo(self) -> MemoryRepository:
        return self._repo

    @property
    def vectors(self) -> VectorStore:
        return self._vectors

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
        """Store a memory and index it for retrieval."""
        if not self._enabled:
            raise RuntimeError("memory is disabled")
        record = self._repo.remember(
            key=key,
            text=text,
            type=type,
            value=value,
            source=source,
            source_ref=source_ref,
            confidence=confidence,
            pinned=pinned,
        )
        # Embedding failures must not lose the memory itself: the row is the
        # source of truth, the vector is only an index over it.
        try:
            embedding_id = self._vectors.index(
                owner_kind="memory",
                owner_id=record.id,
                text=f"{record.key}\n{record.text}",
            )
            self._repo.set_embedding(record.id, embedding_id)
        except Exception:  # indexing is best-effort
            log.warning("could not index memory %s; the row is still stored", record.key, exc_info=True)
        return record

    def forget(self, key: str) -> bool:
        """Delete one memory and its vector."""
        record = self._repo.get_by_key(key)
        if record is None:
            return False
        self._repo.forget(key)
        self._vectors.remove("memory", record.id)
        return True

    def forget_all(self, *, include_pinned: bool = False) -> int:
        """Delete every memory. Returns how many were removed."""
        records = self._repo.list(limit=100_000)
        removed = self._repo.clear(include_pinned=include_pinned)
        for record in records:
            if record.pinned and not include_pinned:
                continue
            self._vectors.remove("memory", record.id)
        return removed

    def touch(self, memory_id: str) -> None:
        self._repo.touch(memory_id)

    # -- reads ---------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
        include_pinned: bool = True,
    ) -> list[RecalledMemory]:
        """Find memories relevant to ``query``.

        Runs a vector search and a full-text search, fuses the two rankings
        with RRF, and returns the best matches.
        """
        if not self._enabled or not query.strip():
            return []

        vector_ids = [
            match.owner_id
            for match in self._vectors.search(query, owner_kind="memory", top_k=limit * 4)
            if match.score > min_score
        ]
        text_ids = [record.id for record in self._repo.search_text(query, limit=limit * 4)]

        fused = rrf([vector_ids, text_ids])
        if not fused:
            return []

        records = {record.id: record for record in self._repo.list(limit=10_000)}
        results: list[RecalledMemory] = []
        for memory_id, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
            record = records.get(memory_id)
            if record is None or record.deleted_at is not None:
                continue
            if not include_pinned and record.pinned:
                continue
            match = "vector" if memory_id in vector_ids else "text"
            results.append(RecalledMemory(record, score, match))
            if len(results) >= limit:
                break

        for item in results:
            self.touch(item.record.id)
        return results

    def list_memories(self, *, type: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        return self._repo.list(type=type, limit=limit)

    def get(self, key: str) -> MemoryRecord | None:
        return self._repo.get_by_key(key)

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for record in self._repo.list(limit=10_000):
            by_type[record.type] = by_type.get(record.type, 0) + 1
        return {
            "enabled": self._enabled,
            "count": self._repo.count(),
            "by_type": by_type,
            "vectors": self._vectors.stats(),
        }
