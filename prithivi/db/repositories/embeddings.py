"""Vector persistence.

Vectors are stored as raw ``float32`` BLOBs rather than in a ``sqlite-vec``
virtual table: ``sqlite-vec`` ships a native extension that cannot be built on
aarch64-Android, and a personal knowledge base (10^4-10^5 rows) scores
perfectly well with an in-process cosine scan. The table shape matches
``DATA_MODEL.md``'s ``vec0`` definition, so a ``sqlite-vec`` or ``pgvector``
backend can be dropped in behind :class:`~prithivi.memory.vector_store.VectorStore`
without touching a row above it.

The codec here is deliberately boring: ``array('f').tobytes()`` in, and the
inverse out. Endianness is normalised explicitly so a database written on a
little-endian phone loads on a big-endian host.
"""

from __future__ import annotations

import array
import datetime as dt
import math
import struct
from collections.abc import Iterable, Sequence
from typing import Any

from ...db.connection import Database
from ...db.ids import new_id

__all__ = ["EmbeddingRepository", "EmbeddingRow", "pack_vector", "unpack_vector"]

#: ``struct`` format for a little-endian float32, used to normalise byte order.
_F32_LE = "<f"


def pack_vector(vector: Sequence[float]) -> bytes:
    """Encode a float sequence as a little-endian float32 blob."""
    values = array.array("f", (float(v) for v in vector))
    if _F32_LE[0] == "<":
        # array uses native byte order; swap when the host is big-endian.
        import sys

        if sys.byteorder == "big":
            values.byteswap()
    return values.tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Decode a float32 blob back into a list of floats.

    Raises:
        ValueError: if the blob length is not a multiple of 4, which means the
            row was written by something other than :func:`pack_vector`.
    """
    if len(blob) % 4:
        raise ValueError(f"vector blob length {len(blob)} is not a multiple of 4")
    values = array.array("f")
    values.frombytes(blob)
    import sys

    if sys.byteorder == "big":
        values.byteswap()
    return list(values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors.

    Returns ``0.0`` when either vector is empty or zero-length, so a degenerate
    embedding can never rank first.
    """
    if len(left) != len(right):
        raise ValueError(f"vector length mismatch: {len(left)} != {len(right)}")
    if not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class EmbeddingRow:
    """One stored vector."""

    __slots__ = ("created_at", "dims", "embedding_id", "model", "owner_id", "owner_kind", "vector")

    def __init__(
        self,
        embedding_id: str,
        owner_kind: str,
        owner_id: str,
        model: str,
        dims: int,
        vector: list[float],
        created_at: str,
    ) -> None:
        self.embedding_id = embedding_id
        self.owner_kind = owner_kind
        self.owner_id = owner_id
        self.model = model
        self.dims = dims
        self.vector = vector
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "owner_kind": self.owner_kind,
            "owner_id": self.owner_id,
            "model": self.model,
            "dims": self.dims,
            "created_at": self.created_at,
        }


class EmbeddingRepository:
    """CRUD over the ``embeddings`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(
        self,
        *,
        owner_kind: str,
        owner_id: str,
        model: str,
        vector: Sequence[float],
        embedding_id: str | None = None,
    ) -> str:
        """Store a vector, replacing any existing one for the same owner.

        The owner is the logical key: one embedding per (owner_kind, owner_id,
        model). Re-embedding a chunk therefore updates in place instead of
        accumulating duplicates.
        """
        values = [float(v) for v in vector]
        if not values:
            raise ValueError("cannot store an empty vector")
        embed_id = embedding_id or new_id("emb_")
        now = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
        with self._db.tx() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE owner_kind=? AND owner_id=? AND model=?",
                (owner_kind, owner_id, model),
            )
            conn.execute(
                "INSERT INTO embeddings (embedding_id, owner_kind, owner_id, model, dims, vec, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (embed_id, owner_kind, owner_id, model, len(values), pack_vector(values), now),
            )
        return embed_id

    def get(self, embedding_id: str) -> EmbeddingRow | None:
        row = self._db.query_one(
            "SELECT embedding_id, owner_kind, owner_id, model, dims, vec, created_at"
            " FROM embeddings WHERE embedding_id=?",
            (embedding_id,),
        )
        return self._to_row(row) if row is not None else None

    def find_by_owner(self, owner_kind: str, owner_id: str, model: str | None = None) -> EmbeddingRow | None:
        sql = "SELECT embedding_id, owner_kind, owner_id, model, dims, vec, created_at FROM embeddings WHERE owner_kind=? AND owner_id=?"
        params: list[Any] = [owner_kind, owner_id]
        if model is not None:
            sql += " AND model=?"
            params.append(model)
        row = self._db.query_one(sql + " LIMIT 1", params)
        return self._to_row(row) if row is not None else None

    def iter_for_owner(self, owner_kind: str, *, model: str | None = None) -> list[EmbeddingRow]:
        """Every vector belonging to one kind of owner (e.g. all chunks)."""
        sql = "SELECT embedding_id, owner_kind, owner_id, model, dims, vec, created_at FROM embeddings WHERE owner_kind=?"
        params: list[Any] = [owner_kind]
        if model is not None:
            sql += " AND model=?"
            params.append(model)
        return [self._to_row(row) for row in self._db.query(sql, params)]

    def iter_all(self, *, model: str | None = None) -> list[EmbeddingRow]:
        sql = "SELECT embedding_id, owner_kind, owner_id, model, dims, vec, created_at FROM embeddings"
        params: list[Any] = []
        if model is not None:
            sql += " WHERE model=?"
            params.append(model)
        return [self._to_row(row) for row in self._db.query(sql, params)]

    def delete(self, embedding_id: str) -> bool:
        with self._db.tx() as conn:
            cur = conn.execute("DELETE FROM embeddings WHERE embedding_id=?", (embedding_id,))
            return cur.rowcount > 0

    def delete_for_owner(self, owner_kind: str, owner_id: str) -> int:
        with self._db.tx() as conn:
            cur = conn.execute(
                "DELETE FROM embeddings WHERE owner_kind=? AND owner_id=?",
                (owner_kind, owner_id),
            )
            return int(cur.rowcount)

    def delete_for_kind(self, owner_kind: str) -> int:
        """Delete every vector of one owner kind (e.g. all chunks)."""
        with self._db.tx() as conn:
            cur = conn.execute("DELETE FROM embeddings WHERE owner_kind=?", (owner_kind,))
            return int(cur.rowcount)

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM embeddings")
        return int(row["n"]) if row is not None else 0

    @staticmethod
    def _to_row(row: Any) -> EmbeddingRow:
        return EmbeddingRow(
            embedding_id=str(row["embedding_id"]),
            owner_kind=str(row["owner_kind"]),
            owner_id=str(row["owner_id"]),
            model=str(row["model"]),
            dims=int(row["dims"]),
            vector=unpack_vector(bytes(row["vec"])),
            created_at=str(row["created_at"]),
        )


def rrf(rankings: Iterable[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked id lists.

    Each list contributes ``1 / (k + rank)`` to every id it contains; the sums
    are the fused score. RRF is used instead of score normalisation because
    cosine similarity and BM25 ranks are not on comparable scales, and RRF
    only needs the ordering.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for index, identifier in enumerate(ranking):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (k + index + 1)
    return fused


def pack_vector_checked(vector: Sequence[float], dims: int) -> bytes:
    """Pack a vector, refusing one whose length disagrees with ``dims``."""
    if len(vector) != dims:
        raise ValueError(f"vector has {len(vector)} dimensions, expected {dims}")
    return pack_vector(vector)


def float32_size(dims: int) -> int:
    """Bytes occupied by a float32 vector of ``dims`` dimensions."""
    return struct.calcsize(_F32_LE) * dims
