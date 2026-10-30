"""Summarisation.

Two implementations behind one interface:

* :class:`ExtractiveSummarizer` — deterministic, offline, no model. Scores
  sentences by the content words they carry and selects the strongest ones in
  their original order. It cannot hallucinate, because it only ever copies
  text that was present.
* :class:`ModelSummarizer` — asks the configured LLM for a summary. Used when
  a provider is available and the user has not opted out.

The extractive one is the default, not a fallback. On a phone with no API key
the assistant still has to be able to remember what a conversation was about,
and "the user asked about X, then Y" is a perfectly good answer to that.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..rag.embeddings import tokenize

__all__ = ["ExtractiveSummarizer", "ModelSummarizer", "Summarizer", "Summary"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Summary:
    """A condensed view of a stretch of conversation."""

    summary: str
    key_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "key_points": self.key_points}


class Summarizer:
    """Interface every summariser implements."""

    def summarize(self, text: str, *, existing: str = "") -> Summary:
        raise NotImplementedError


class ExtractiveSummarizer(Summarizer):
    """Selects the most informative sentences, verbatim.

    Args:
        max_sentences: how many sentences the summary may contain.
        max_chars: hard cap on the summary length.
    """

    def __init__(self, *, max_sentences: int = 4, max_chars: int = 600) -> None:
        self.max_sentences = max(1, max_sentences)
        self.max_chars = max(80, max_chars)

    def summarize(self, text: str, *, existing: str = "") -> Summary:
        sentences = [s.strip() for s in _SENTENCE_RE.split(text or "") if s and s.strip()]
        if not sentences:
            return Summary(summary=existing, key_points=[])

        frequencies = _term_frequencies(sentences)
        scored = [(_score(s, frequencies, index, len(sentences)), index, s) for index, s in enumerate(sentences)]
        scored.sort(key=lambda item: (-item[0], item[1]))

        chosen = sorted(scored[: self.max_sentences], key=lambda item: item[1])
        picked = [sentence for _, _, sentence in chosen]

        summary = " ".join(picked)
        if len(summary) > self.max_chars:
            summary = summary[: self.max_chars].rsplit(" ", 1)[0] + "…"

        points = [f" {_trim(sentence, 140)}" for _, _, sentence in chosen]
        merged = existing.strip()
        if merged:
            summary = f"{merged}\n{summary}".strip()
        return Summary(summary=summary, key_points=points)


class ModelSummarizer(Summarizer):
    """Summarises with the configured LLM.

    Args:
        provider: anything with the ``LLMProvider`` protocol.
        max_sentences: how many sentences to ask for.
        model_hint: recorded in the prompt so the provider can pick a cheap
            model when it supports one.
    """

    _SYSTEM = (
        "You summarise conversations for an assistant's own memory. Write plain prose, "
        "no preamble, no bullet points. Keep decisions, facts, names, numbers and "
        "open questions. Drop pleasantries. Never invent anything that was not said."
    )

    def __init__(self, provider: Any, *, max_sentences: int = 4, model_hint: str = "") -> None:
        self._provider = provider
        self.max_sentences = max_sentences
        self.model_hint = model_hint

    async def summarize_async(self, text: str, *, existing: str = "") -> Summary:
        """Async variant, used by the agent where the event loop is available."""
        from ..llm.provider import Message

        prompt = (
            f"Summarise the following in at most {self.max_sentences} sentences.\n\n"
            f"{text[:8000]}"
        )
        if existing:
            prompt += f"\n\nContext so far:\n{existing[:2000]}"
        completion = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            system=self._SYSTEM,
        )
        summary = (completion.text or "").strip()
        if not summary:
            return ExtractiveSummarizer().summarize(text, existing=existing)
        return Summary(summary=summary, key_points=[summary])

    def summarize(self, text: str, *, existing: str = "") -> Summary:
        """Synchronous variant.

        The model provider is async, so this runs the coroutine to completion
        only when no loop is already running. Inside the agent's event loop the
        async path is used instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.summarize_async(text, existing=existing))
        # A loop is already running: fall back to the deterministic path rather
        # than deadlocking on a nested run().
        return ExtractiveSummarizer().summarize(text, existing=existing)


def _term_frequencies(sentences: Sequence[str]) -> dict[str, float]:
    """How often each content word appears across the sentences."""
    counts: dict[str, int] = {}
    for sentence in sentences:
        for token in set(tokenize(sentence)):
            counts[token] = counts.get(token, 0) + 1
    total = float(len(sentences)) or 1.0
    return {token: count / total for token, count in counts.items()}


def _score(sentence: str, frequencies: dict[str, float], index: int, total: int) -> float:
    """Score one sentence.

    Content-word density drives the score; a mild positional prior favours the
    opening and closing sentences, which in a conversation usually carry the
    request and the outcome.
    """
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    density = sum(frequencies.get(token, 0.0) for token in tokens) / len(tokens)
    position = 1.0
    if total > 1:
        if index == 0 or index == total - 1:
            position = 1.2
        else:
            position = 1.0
    length_penalty = 1.0 if 5 <= len(tokens) <= 40 else 0.7
    return density * position * length_penalty


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
