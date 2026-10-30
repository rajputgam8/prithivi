"""Tools over the user's document knowledge base.

The knowledge base is the user's own material: PDFs, notes, pages they chose to
keep. Search results are returned with a citation for every chunk, and the whole
payload is flagged as untrusted, because a document is text from outside the
model's own knowledge and must never be able to carry instructions into a turn.
"""

from __future__ import annotations

import hashlib

from ..core.schema import Model
from .base import Evidence, SideEffect, ToolContext, ToolErrorCode, ToolResult, tool

__all__ = ["AddNoteArgs", "IngestDocumentArgs", "SearchKnowledgeArgs", "build_rag_registry"]


class SearchKnowledgeArgs(Model):
    """Tool arguments model."""

    query: str
    """What to look for in the user's saved documents."""
    top_k: int = 5
    """Maximum number of chunks to return (1-10)."""


@tool(
    name="search_knowledge",
    description=(
        "Search the user's own saved documents (PDFs, notes, pages they ingested) "
        "and return cited excerpts. Use it when the user asks about something in "
        "their files, notes, or 'my documents'. Do NOT use it for the live web "
        "(use search_web) or for general knowledge."
    ),
    input_model=SearchKnowledgeArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=True,
    timeout_s=15.0,
)
async def _search_knowledge(ctx: ToolContext, query: str, top_k: int = 5) -> ToolResult:
    kb = getattr(ctx, "knowledge", None)
    if kb is None:
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "the knowledge base is not attached",
            remedy="Start Prithivi with its normal runtime, not an isolated registry.",
        )
    if not query.strip():
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "the knowledge query was empty")
    top_k = max(1, min(int(top_k), 10))

    try:
        results = kb.search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - a search failure is a message, not a crash
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"knowledge search failed: {exc}")

    if not results:
        return ToolResult.ok(
            {"result": "I found nothing in your saved documents about that.", "chunks": []},
            evidence=Evidence("knowledge_search", {"query": query, "count": 0}),
        )

    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.citation()}\n{result.text}")
    payload = {
        "result": "\n\n".join(lines),
        "chunks": [result.to_dict() for result in results],
        "citations": [result.citation() for result in results],
    }
    return ToolResult.ok(
        payload,
        evidence=Evidence(
            "knowledge_search",
            {"query": query, "count": len(results), "citations": payload["citations"]},
        ),
        untrusted_source="knowledge base",
    )


class IngestDocumentArgs(Model):
    """Tool arguments model."""

    path: str
    """An absolute or user-relative path to a .txt, .md, .html or .pdf file."""
    title: str | None = None
    """Optional display title; defaults to the file name."""


@tool(
    name="ingest_document",
    description=(
        "Add a file on the user's device to their searchable knowledge base. Use it "
        "when the user explicitly asks to ingest, index, or 'learn' a document. Do "
        "NOT use it to read a file the user only wants summarised in this turn, and "
        "do NOT read arbitrary paths the user has not named."
    ),
    input_model=IngestDocumentArgs,
    permission_level=2,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=True,
    timeout_s=60.0,
)
async def _ingest_document(ctx: ToolContext, path: str, title: str | None = None) -> ToolResult:
    kb = getattr(ctx, "knowledge", None)
    if kb is None:
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "the knowledge base is not attached",
        )
    if not path.strip():
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "no document path was given")

    from ..rag.ingestion import IngestionError

    try:
        document = kb.add_file(path, title=title)
    except IngestionError as exc:
        return ToolResult.fail(
            exc.code,
            str(exc),
            remedy="Check the path and that the file is a supported .txt, .md, .html or .pdf document.",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"ingestion failed: {exc}")

    label = "already indexed" if document.duplicate else "indexed"
    return ToolResult.ok(
        {
            "result": f"{document.title or document.uri} {label}: {document.chunk_count} chunk(s).",
            "document": document.to_dict(),
        },
        evidence=Evidence(
            "document_indexed",
            {"document_id": document.document_id, "chunks": document.chunk_count},
        ),
    )


class AddNoteArgs(Model):
    """Tool arguments model."""

    text: str
    """The note body to save."""
    title: str | None = None
    """A short title for the note."""


@tool(
    name="add_note",
    description=(
        "Save a piece of text as a note in the user's knowledge base so it can be "
        "searched later. Use it when the user says to save something to their notes. "
        "Do NOT use it for transient conversation or for facts better stored as a "
        "short memory."
    ),
    input_model=AddNoteArgs,
    permission_level=1,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=False,
    timeout_s=30.0,
)
async def _add_note(ctx: ToolContext, text: str, title: str | None = None) -> ToolResult:
    kb = getattr(ctx, "knowledge", None)
    if kb is None:
        return ToolResult.fail(
            ToolErrorCode.BACKEND_UNAVAILABLE,
            "the knowledge base is not attached",
        )
    if not text.strip():
        return ToolResult.fail(ToolErrorCode.INVALID_INPUT, "the note was empty")

    from ..rag.ingestion import IngestionError

    try:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        document = kb.add_text(
            text,
            uri=f"note://{digest}",
            title=title or "Note",
            source_type="note",
        )
    except IngestionError as exc:
        return ToolResult.fail(exc.code, str(exc))

    return ToolResult.ok(
        {
            "result": f"Saved note {document.title!r} ({document.chunk_count} chunk(s)).",
            "document": document.to_dict(),
        },
        evidence=Evidence(
            "document_indexed",
            {"document_id": document.document_id, "chunks": document.chunk_count},
        ),
    )


def build_rag_registry() -> list:
    """Return the knowledge-base tool specs."""
    return [_search_knowledge, _ingest_document, _add_note]
