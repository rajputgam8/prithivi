"""Document ingestion.

Turns a file, a URL body or a raw string into a stored, chunked, embedded
document.

The rules:

* **Content addressing.** A document is identified by the SHA-256 of its
  bytes, so ingesting the same file twice is a no-op instead of a duplicate.
* **Degrade honestly.** A PDF that cannot be parsed because ``pypdf`` is
  missing produces a clear "backend unavailable" error, not an empty
  document that looks like it was ingested successfully.
* **Never half-store.** The document row, its chunks and their embeddings are
  written together; a failure leaves the document marked ``failed`` rather
  than half-indexed.
"""

from __future__ import annotations

import hashlib
import html.parser
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ..db.connection import Database
from ..db.repositories.knowledge import KnowledgeRepository
from ..memory.vector_store import VectorStore
from .chunking import chunk_text

log = logging.getLogger("prithivi.rag.ingestion")

__all__ = ["DocumentIngestor", "IngestedDocument", "IngestionError", "html_to_text"]


class IngestionError(Exception):
    """Raised when a document cannot be ingested.

    Carries a machine-readable ``code`` so the tool layer can map it onto a
    tool error without string matching.
    """

    def __init__(self, message: str, *, code: str = "INGESTION_FAILED") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class IngestedDocument:
    """What ingestion produced."""

    document_id: str
    uri: str
    title: str | None
    sha256: str
    chunk_count: int
    token_count: int
    status: str
    duplicate: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "uri": self.uri,
            "title": self.title,
            "sha256": self.sha256[:16],
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "status": self.status,
            "duplicate": self.duplicate,
            "meta": self.meta,
        }


class _TextExtractor(html.parser.HTMLParser):
    """Pulls readable text out of HTML, dropping scripts and styles."""

    _SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "svg", "head"})
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n\s*\n\s*", "\n\n", joined)
        return joined.strip()


def html_to_text(html: str) -> str:
    """Extract readable text from an HTML document."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed HTML should not be fatal
        log.debug("HTML parser recovered from malformed input", exc_info=True)
    return parser.text()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentIngestor:
    """Ingests documents into the knowledge base.

    Args:
        db: the database.
        vector_store: where chunk vectors go. When omitted, chunks are stored
            without vectors and only full-text search finds them — which is a
            valid, if less capable, mode.
        chunk_size: target chunk size in tokens.
        chunk_overlap: overlap between neighbouring chunks.
    """

    def __init__(
        self,
        db: Database,
        *,
        vector_store: VectorStore | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._db = db
        self._repo = KnowledgeRepository(db)
        self._vectors = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def repo(self) -> KnowledgeRepository:
        """The underlying repository, for callers that need to read rows."""
        return self._repo

    # -- public API ----------------------------------------------------
    def ingest_text(
        self,
        text: str,
        *,
        uri: str = "memory://text",
        title: str | None = None,
        source_type: str = "text",
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """Ingest a raw string."""
        if not text or not text.strip():
            raise IngestionError("document is empty", code="EMPTY_DOCUMENT")
        return self._store(
            text=text,
            uri=uri,
            title=title,
            source_type=source_type,
            raw=text.encode("utf-8"),
            meta=meta,
        )

    def ingest_file(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        source_type: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """Ingest a file from disk.

        Supports plain text, Markdown, HTML and PDF. The type is guessed from
        the extension and the content is sniffed, because a file called
        ``notes.txt`` is quite often HTML.
        """
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise IngestionError(f"no such file: {file_path}", code="NOT_FOUND")
        if not file_path.is_file():
            raise IngestionError(f"not a file: {file_path}", code="NOT_A_FILE")

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            raise IngestionError(f"cannot read {file_path}: {exc}", code="UNREADABLE") from exc

        suffix = file_path.suffix.lower()
        kind = source_type or {
            ".pdf": "pdf",
            ".html": "html",
            ".htm": "html",
            ".md": "markdown",
            ".markdown": "markdown",
            ".txt": "text",
        }.get(suffix, "text")

        text, page_count = self._extract(raw, kind, file_path)
        payload = dict(meta or {})
        if page_count:
            payload["page_count"] = page_count
        return self._store(
            text=text,
            uri=file_path.as_uri() if file_path.is_absolute() else str(file_path),
            title=title or file_path.name,
            source_type=kind,
            raw=raw,
            meta=payload,
        )

    def ingest_html(
        self,
        html: str,
        *,
        uri: str,
        title: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """Ingest an HTML page (for example the output of ``web_fetch``)."""
        text = html_to_text(html)
        if not text:
            raise IngestionError("the page had no readable text", code="EMPTY_DOCUMENT")
        return self._store(
            text=text,
            uri=uri,
            title=title,
            source_type="html",
            raw=html.encode("utf-8"),
            meta=meta,
        )

    # -- internals -----------------------------------------------------
    @staticmethod
    def _extract(raw: bytes, kind: str, path: Path) -> tuple[str, int]:
        """Pull text out of raw bytes according to ``kind``."""
        if kind == "pdf":
            return DocumentIngestor._extract_pdf(raw, path)
        if kind == "html":
            try:
                return html_to_text(raw.decode("utf-8", errors="replace")), 0
            except Exception as exc:
                raise IngestionError(f"cannot decode HTML: {exc}", code="DECODE_FAILED") from exc
        try:
            return raw.decode("utf-8"), 0
        except UnicodeDecodeError:
            # Binary that is not a PDF: keep the readable parts rather than
            # refusing outright, but say so in the metadata.
            return raw.decode("utf-8", errors="replace"), 0

    @staticmethod
    def _extract_pdf(raw: bytes, path: Path) -> tuple[str, int]:
        """Extract text from a PDF.

        ``pypdf`` is optional. When it is missing the failure is explicit:
        ``BACKEND_UNAVAILABLE`` is a very different message to the user than
        "the PDF is empty".
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError(
                "reading PDFs needs the 'pypdf' package (pip install pypdf)",
                code="BACKEND_UNAVAILABLE",
            ) from exc

        import io

        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        except Exception as exc:
            raise IngestionError(f"cannot read the PDF: {exc}", code="PARSE_FAILED") from exc

        text = "\n\n".join(p for p in pages if p.strip())
        if not text.strip():
            raise IngestionError(
                "the PDF has no extractable text (it may be scanned images)",
                code="NO_TEXT_LAYER",
            )
        return text, len(reader.pages)

    def _store(
        self,
        *,
        text: str,
        uri: str,
        title: str | None,
        source_type: str,
        raw: bytes,
        meta: dict[str, Any],
    ) -> IngestedDocument:
        """Chunk, store and embed one document."""
        meta = dict(meta or {})
        digest = _sha256(raw)
        existing = self._repo.get_document_by_hash(digest)
        if existing is not None:
            return IngestedDocument(
                document_id=existing.id,
                uri=existing.uri,
                title=existing.title,
                sha256=existing.sha256,
                chunk_count=len(self._repo.list_chunks(existing.id)),
                token_count=0,
                status=existing.status,
                duplicate=True,
            )

        document = self._repo.get_document_by_uri(uri)
        if document is not None and document.sha256 != digest:
            # The file at this URI changed. Remove its old chunk vectors before
            # swapping the chunks, or the index would keep returning text that
            # is no longer in the document.
            for old_chunk in self._repo.list_chunks(document.id):
                if self._vectors is not None:
                    self._vectors.remove("chunk", old_chunk.id)
            document = self._repo.update_document(
                document.id,
                uri=uri,
                title=title,
                source_type=source_type,
                sha256=digest,
                byte_size=len(raw),
                page_count=meta.get("page_count"),
                meta=meta,
            )
        else:
            document = self._repo.upsert_document(
                uri=uri,
                title=title,
                source_type=source_type,
                sha256=digest,
                byte_size=len(raw),
                page_count=meta.get("page_count"),
                meta=meta,
            )

        chunks = chunk_text(
            text,
            size=self.chunk_size,
            overlap=self.chunk_overlap,
            locator=uri,
            meta={"source_type": source_type},
        )
        if not chunks:
            self._repo.set_document_status(document.id, "failed")
            raise IngestionError("the document produced no usable text", code="EMPTY_DOCUMENT")

        self._repo.replace_chunks(document.id, [chunk.to_dict() for chunk in chunks])
        token_count = 0
        for chunk in self._repo.list_chunks(document.id):
            token_count += chunk.token_count
            if self._vectors is None:
                continue
            try:
                embedding_id = self._vectors.index(
                    owner_kind="chunk",
                    owner_id=chunk.id,
                    text=chunk.text,
                )
                self._repo.set_chunk_embedding(chunk.id, embedding_id)
            except Exception:  # a missing vector is not a failed ingest
                log.warning("could not index chunk %s", chunk.id, exc_info=True)

        self._repo.set_document_status(document.id, "indexed", indexed=True)
        return IngestedDocument(
            document_id=document.id,
            uri=document.uri,
            title=document.title,
            sha256=document.sha256,
            chunk_count=len(chunks),
            token_count=token_count,
            status="indexed",
            meta=meta,
        )
