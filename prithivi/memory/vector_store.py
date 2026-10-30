"""Vector store over the ``embeddings`` table.

This is the only place that knows how vectors are persisted and scored. The
backend is deliberately swappable in spirit — the SQL lives behind
:class:`~prithivi.db.repositories.embeddings.EmbeddingRepository`, so moving to
``sqlite-vec`` or ``pgvector`` means changing the repository, not the callers.

Scoring is an in-process cosine scan. At personal scale (10^4-10^5 rows) this
is milliseconds, and it avoids shipping a native extension to a phone that
cannot compile one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..db.connection import Database
from ..db.repositories.embeddings import EmbeddingRepository, cosine_similarity
from ..rag.embeddings import Embedder, HashEmbedder

__all__ = ["VectorMatch", "VectorStore"]


class VectorMatch:
    """One scored neighbour."""

    __slots__ = ("owner_id", "owner_kind", "score", "vector")

    def __init__(self, owner_kind: str, owner_id: str, score: float, vector: list[float]) -> None:
        self.owner_kind = owner_kind
        self.owner_id = owner_id
        self.score = score
        self.vector = vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_kind": self.owner_kind,
            "owner_id": self.owner_id,
            "score": round(self.score, 6),
        }


class VectorStore:
    """Stores and retrieves embeddings.

    Args:
        db: the database.
        embedder: the encoder used for queries. Stored vectors keep the model
            name they were created with, so vectors from a different model are
            never compared.
    """

    def __init__(self, db: Database, *, embedder: Embedder | None = None) -> None:
        self._db = db
        self._repo = EmbeddingRepository(db)
        self._embedder = embedder or HashEmbedder()

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def model(self) -> str:
        return self._embedder.model

    def index(
        self,
        *,
        owner_kind: str,
        owner_id: str,
        text: str,
        model: str | None = None,
    ) -> str:
        """Embed ``text`` and store it against ``owner_id``."""
        vector = self._embedder.embed(text)
        return self._repo.upsert(
            owner_kind=owner_kind,
            owner_id=owner_id,
            model=model or self._embedder.model,
            vector=vector,
        )

    def index_vector(
        self,
        *,
        owner_kind: str,
        owner_id: str,
        vector: Sequence[float],
        model: str | None = None,
    ) -> str:
        """Store a pre-computed vector."""
        return self._repo.upsert(
            owner_kind=owner_kind,
            owner_id=owner_id,
            model=model or self._embedder.model,
            vector=vector,
        )

    def search(
        self,
        query: str,
        *,
        owner_kind: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        model: str | None = None,
    ) -> list[VectorMatch]:
        """Nearest neighbours of ``query``.

        Only vectors written by the *same* model as the query are compared. A
        vector from another encoder is not "a bad match", it is not a match at
        all, and mixing them silently would poison the ranking.
        """
        query_vector = self._embedder.embed(query)
        wanted_model = model or self._embedder.model
        rows = (
            self._repo.iter_for_owner(owner_kind, model=wanted_model)
            if owner_kind
            else self._repo.iter_all(model=wanted_model)
        )
        matches: list[VectorMatch] = []
        for row in rows:
            score = cosine_similarity(query_vector, row.vector)
            if score <= min_score:
                continue
            matches.append(VectorMatch(row.owner_kind, row.owner_id, score, row.vector))
        matches.sort(key=lambda m: (-m.score, m.owner_id))
        return matches[:top_k]

    def get(self, owner_kind: str, owner_id: str) -> list[float] | None:
        row = self._repo.find_by_owner(owner_kind, owner_id, self._embedder.model)
        return row.vector if row is not None else None

    def remove(self, owner_kind: str, owner_id: str) -> int:
        return self._repo.delete_for_owner(owner_kind, owner_id)

    def remove_kind(self, owner_kind: str) -> int:
        return self._repo.delete_for_kind(owner_kind)

    def count(self) -> int:
        return self._repo.count()

    def stats(self) -> dict[str, Any]:
        """Counts by owner kind, for ``/memory/stats`` and the HUD."""
        rows = self._db.query(
            "SELECT owner_kind, COUNT(*) AS n FROM embeddings WHERE model=? GROUP BY owner_kind",
            (self._embedder.model,),
        )
        by_kind = {str(r["owner_kind"]): int(r["n"]) for r in rows}
        return {
            "model": self._embedder.model,
            "dims": self._embedder.dims,
            "total": sum(by_kind.values()),
            "by_kind": by_kind,
        }
