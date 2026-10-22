"""Text-to-speech backends.

The default is ``termux-tts-speak`` (Android's system TTS). Long replies are
split into sentence-sized chunks so the first words start playing promptly and a
single oversized string never hits the engine's per-utterance limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Callable

from .base import TextToSpeech, VoiceError, find_termux_command

__all__ = ["RecordingTTS", "SilentTTS", "TermuxTTS", "chunk_text", "prepare_speech_text"]

#: Sentence-sized chunks keep latency low and stay well under engine limits.
_MAX_CHUNK = 400

_INSTALL_REMEDY = (
    "Install the Termux:API app from F-Droid and run `pkg install termux-api`, "
    "then run Prithivi in native Termux (not inside proot)."
)

# --- speech cleanup --------------------------------------------------------

#: Text produced for the eye (markdown, code, URLs, symbols) read aloud verbatim
#: sounds robotic — "star star bold star star", "h-t-t-p-s colon slash slash".
#: These rewrite it into something a person would actually say.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?\S)\1")
_MD_PREFIX = re.compile(r"(?m)^\s{0,3}(?:[-*+]|\d+[.)])\s+")
_URL = re.compile(r"https?://(\S+)")
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")

_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\bMr\.", re.IGNORECASE), "Mister"),
    (re.compile(r"\bMrs\.", re.IGNORECASE), "Missus"),
    (re.compile(r"\bDr\.", re.IGNORECASE), "Doctor"),
    (re.compile(r"\bSt\.", re.IGNORECASE), "Saint"),
)


def _spoken_url(match: re.Match[str]) -> str:
    """``https://docs.python.org/3/`` -> ``docs dot python dot org``."""
    target = match.group(1).split("/", 1)[0].split("?", 1)[0]
    return target.replace(".", " dot ") if "." in target else target


def prepare_speech_text(text: str) -> str:
    """Rewrite model output into natural spoken English.

    Deliberately conservative: it removes *presentation* a person would never say
    (markdown emphasis, code fences, bullets, raw URLs, ``%``/``@``/``&``) and
    expands a few common abbreviations. It never edits content words, so nothing
    the model said is changed in meaning.
    """
    text = _CODE_FENCE.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _MD_PREFIX.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _URL.sub(_spoken_url, text)
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = text.replace(" & ", " and ").replace("&amp;", "and")
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    text = text.replace(" @ ", " at ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _clamp_rate(rate: float) -> float:
    """Termux accepts 0.5 (half speed) to 2.0 (double); clamp to stay audible."""
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return 1.0
    return min(2.0, max(0.5, value))


def _clamp_pitch(pitch: float) -> float:
    """Termux pitch is a multiplier around 1.0; clamp so a typo cannot make the
    voice an incomprehensible squeak or growl."""
    try:
        value = float(pitch)
    except (TypeError, ValueError):
        return 1.0
    return min(2.0, max(0.5, value))


def _split_long(sentence: str, limit: int) -> list[str]:
    words = sentence.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, *, limit: int = _MAX_CHUNK) -> list[str]:
    """Split ``text`` into speakable chunks on sentence boundaries.

    Never splits a word, and never returns an empty chunk.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long(sentence, limit))
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


class TermuxTTS(TextToSpeech):
    """Android text-to-speech via ``termux-tts-speak``."""

    name = "termux"

    def __init__(
        self,
        *,
        rate: float = 1.0,
        pitch: float = 1.0,
        language: str = "",
        voice: str = "",
        engine: str = "",
        command: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._command = command if command is not None else find_termux_command("termux-tts-speak")
        self._rate = _clamp_rate(rate)
        self._pitch = _clamp_pitch(pitch)
        self._language = language.strip()
        self._voice = voice.strip()
        self._engine = engine.strip()
        self._timeout = timeout

    async def available(self) -> tuple[bool, str]:
        if self._command is None or not os.path.isfile(self._command):
            return False, "termux-tts-speak not found"
        return True, "android text-to-speech via Termux:API"

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self._command is None:
            raise VoiceError("text-to-speech is unavailable", remedy=_INSTALL_REMEDY)
        for chunk in chunk_text(prepare_speech_text(text)):
            await self._say(chunk)

    async def _say(self, chunk: str) -> None:
        argv = [self._command, "-r", f"{self._rate:.2f}", "-p", f"{self._pitch:.2f}"]
        if self._language:
            argv += ["-l", self._language]
        if self._engine:
            argv += ["-e", self._engine]
        if self._voice:
            argv += ["-n", self._voice]
        argv.append(chunk)

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VoiceError("text-to-speech is unavailable", remedy=_INSTALL_REMEDY) from exc

        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise VoiceError(
                "text-to-speech timed out",
                remedy="Try a lower PRITHIVI_TTS_RATE or a shorter reply.",
            ) from None

        if proc.returncode != 0:
            detail = (err or out).decode(errors="replace").strip() or f"exit status {proc.returncode}"
            raise VoiceError(f"text-to-speech failed: {detail}")


class SilentTTS(TextToSpeech):
    """Produce no sound — headless runs, tests, and the demo path."""

    name = "silent"

    async def speak(self, text: str) -> None:
        return None


class RecordingTTS(TextToSpeech):
    """Record what would have been spoken, for assertions."""

    name = "recording"

    def __init__(self, echo: Callable[[str], None] | None = None) -> None:
        self.spoken: list[str] = []
        self._echo = echo

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        self.spoken.append(text)
        if self._echo is not None:
            self._echo(text)
