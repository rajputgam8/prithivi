"""Repositories: every SQL statement in Prithivi lives here.

Each repository owns one table family and exposes plain Python methods. Nothing
above this layer writes SQL, so the schema can change in one place, and every
write goes through :meth:`~prithivi.db.connection.Database.tx`, which makes
multi-table writes atomic.

Import rules (ARCHITECTURE.md S2): repositories may import from
:mod:`prithivi.db`, never from ``core`` or ``llm``.
"""

from __future__ import annotations

from .embeddings import (
    EmbeddingRepository,
    EmbeddingRow,
    cosine_similarity,
    pack_vector,
    rrf,
    unpack_vector,
)
from .knowledge import ChunkRecord, DocumentRecord, KnowledgeRepository
from .memories import MEMORY_TYPES, MemoryRecord, MemoryRepository
from .sessions import MessageRecord, SessionRecord, SessionRepository
from .settings import SettingsRepository

__all__ = [
    "MEMORY_TYPES",
    "ChunkRecord",
    "DocumentRecord",
    "EmbeddingRepository",
    "EmbeddingRow",
    "KnowledgeRepository",
    "MemoryRecord",
    "MemoryRepository",
    "MessageRecord",
    "SessionRecord",
    "SessionRepository",
    "SettingsRepository",
    "cosine_similarity",
    "pack_vector",
    "rrf",
    "unpack_vector",
]
