"""The knowledge base: documents in, cited answers out.

This is the facade the tools and the CLI talk to. It owns the wiring between
ingestion, the vector index and retrieval, so no caller has to know that
documents live in ``documents``/``chunks`` and vectors live in ``embeddings``.
"""

from __future__ import annotations

from typing import Any

from ..db.connection import Database
from ..db.repositories.knowledge import DocumentRecord
from ..memory.vector_store import VectorStore
from .ingestion import DocumentIngestor, IngestedDocument
from .retrieval import RetrievedChunk, Retriever

__all__ = ["KnowledgeBase"]


class KnowledgeBase:
    """The user's own documents, searchable and citable.

    Args:
        db: the database.
        embedder: the encoder. Defaults to ``hash-embed-v1``.
        chunk_size: target chunk size in tokens.
        chunk_overlap: overlap between neighbouring chunks.
        top_k: default number of retrieved chunks.
    """

    def __init__(
        self,
        db: Database,
        *,
        embedder: Any = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 8,
    ) -> None:
        self._db = db
        self.vectors = VectorStore(db, embedder=embedder)
        self.ingestor = DocumentIngestor(
            db,
            vector_store=self.vectors,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = Retriever(db, vector_store=self.vectors, top_k=top_k)

    # -- writes --------------------------------------------------------
    def add_text(
        self,
        text: str,
        *,
        uri: str = "memory://text",
        title: str | None = None,
        source_type: str = "text",
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        return self.ingestor.ingest_text(text, uri=uri, title=title, source_type=source_type, meta=meta)

    def add_file(
        self,
        path: str,
        *,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        return self.ingestor.ingest_file(path, title=title, meta=meta)

    def add_html(
        self,
        html: str,
        *,
        uri: str,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        return self.ingestor.ingest_html(html, uri=uri, title=title, meta=meta)

    def delete(self, document_id: str) -> bool:
        """Delete a document, its chunks and their vectors."""
        chunks = self.ingestor.repo.list_chunks(document_id)
        deleted = self.ingestor.repo.delete_document(document_id)
        for chunk in chunks:
            self.vectors.remove("chunk", chunk.id)
        return deleted

    # -- reads ---------------------------------------------------------
    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.search(query, top_k=top_k)

    def context(self, query: str, *, top_k: int | None = None, max_chars: int = 4000) -> str:
        return self.retriever.context(query, top_k=top_k, max_chars=max_chars)

    def documents(self, *, limit: int = 100) -> list[DocumentRecord]:
        return self.retriever.documents(limit=limit)

    def stats(self) -> dict[str, Any]:
        return self.retriever.stats()
