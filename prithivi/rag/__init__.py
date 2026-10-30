"""Retrieval-augmented generation: the user's own documents.

Four pieces:

* :mod:`prithivi.rag.embeddings` — ``hash-embed-v1``, the dependency-free
  encoder.
* :mod:`prithivi.rag.chunking` — sentence-aware splitting with overlap.
* :mod:`prithivi.rag.ingestion` — files, pages and strings into documents.
* :mod:`prithivi.rag.retrieval` — hybrid search and cited context blocks.

Plus the two that were already here: :mod:`prithivi.rag.citations` (claim
verification across sources) and :mod:`prithivi.rag.web_research` (the live
web).

Import rules: the RAG layer may import from :mod:`prithivi.db` and
:mod:`prithivi.memory`, never from ``llm`` or ``interface``.
"""

from __future__ import annotations

from .chunking import Chunk, chunk_text, estimate_tokens, split_sentences
from .embeddings import Embedder, HashEmbedder, build_embedder, hash_embed
from .ingestion import DocumentIngestor, IngestedDocument, IngestionError
from .knowledge_base import KnowledgeBase
from .retrieval import RetrievedChunk, Retriever, format_context

__all__ = [
    "Chunk",
    "DocumentIngestor",
    "Embedder",
    "HashEmbedder",
    "IngestedDocument",
    "IngestionError",
    "KnowledgeBase",
    "RetrievedChunk",
    "Retriever",
    "build_embedder",
    "chunk_text",
    "estimate_tokens",
    "format_context",
    "hash_embed",
    "split_sentences",
]
