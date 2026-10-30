"""Text chunking.

Chunks are the unit of retrieval: a document is split into pieces small enough
to fit a prompt and large enough to carry a complete thought.

Two properties matter more than anything else here:

* **Boundaries follow sentences.** A chunk that starts mid-sentence reads as
  broken, and a retrieval that returns half a sentence is worse than one that
  returns nothing.
* **Neighbouring chunks overlap.** A fact that straddles a boundary would
  otherwise be retrievable from neither side. The overlap costs a little
  storage and removes a whole class of missed answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Chunk", "chunk_text", "estimate_tokens", "split_sentences"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    ordinal: int
    text: str
    token_count: int = 0
    locator: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "text": self.text,
            "token_count": self.token_count,
            "locator": self.locator,
            "meta": self.meta,
        }


def estimate_tokens(text: str) -> int:
    """Approximate token count (four characters per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping paragraph breaks as hard stops."""
    if not text:
        return []
    parts: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = _WHITESPACE.sub(" ", sentence).strip()
            if sentence:
                parts.append(sentence)
    return parts


def chunk_text(
    text: str,
    *,
    size: int = 512,
    overlap: int = 64,
    locator: str | None = None,
    meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: the document body.
        size: target chunk size in tokens. Chunks may exceed it slightly when a
            single sentence is longer — a sentence is never cut in half.
        overlap: how many tokens of the previous chunk to repeat at the start of
            the next one. Clamped so it can never reach ``size``.
        locator: a human-facing pointer to this chunk's place in the document
            ("page 4"), stored on every chunk.
        meta: extra metadata copied onto each chunk.

    Returns:
        The chunks, in document order. An empty or whitespace-only text gives
        an empty list.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    sentences = split_sentences(text)
    if not sentences:
        return []

    overlap = max(0, min(overlap, max(0, size // 2)))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = " ".join(current).strip()
        if body:
            ordinal = len(chunks)
            # The locator is what a citation shows. Include the chunk number so
            # "answered from this document" can point at the exact passage.
            chunk_locator = f"{locator}#L{ordinal + 1}" if locator else None
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=body,
                    token_count=estimate_tokens(body),
                    locator=chunk_locator,
                    meta=dict(meta or {}),
                )
            )
        current = []
        current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > size:
            # Carry the tail of the current chunk into the next one.
            carry: list[str] = []
            carried = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if carried + previous_tokens > overlap:
                    break
                carry.insert(0, previous)
                carried += previous_tokens
            flush()
            current = list(carry)
            current_tokens = carried
        current.append(sentence)
        current_tokens += sentence_tokens

    flush()
    for index, chunk in enumerate(chunks):
        chunk.ordinal = index
    return chunks
