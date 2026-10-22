"""The voice conversation loop.

    listen → (strip wake phrase) → agent turn → speak the honest result

Barge-in in the roadmap sense — interrupting mid-sentence — is not implemented
here: ``termux-tts-speak`` blocks until it has finished, and Android hands Termux
no way to hold the microphone open while it plays. Instead this loop uses an
**echo guard** (so the assistant never answers its own last sentence replayed
back through open speakers) and **stop phrases** (so the user can end a session
by voice). Those are the parts of barge-in that are achievable on this device;
they are stated here rather than silently omitted.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.state import Modality, TurnSource
from ..security.confirmation import ConfirmationOutcome, classify_voice_confirmation
from .base import SpeechToText, TextToSpeech, VoiceError
from .wake_word import DEFAULT_STOP_PHRASES, WakeWord, normalize

__all__ = ["VoiceConversation", "VoiceExit", "make_confirm_handler"]

log = logging.getLogger("prithivi.voice")

#: A sentence boundary: terminal punctuation followed by whitespace or the end.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")
#: Don't pause on a boundary inside an abbreviation ("e.g.", "Dr.").
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "st.")


def split_ready_sentences(buffer: str, *, min_chars: int = 12) -> tuple[list[str], str]:
    """Peel whole sentences off ``buffer`` for immediate speaking.

    Returns ``(sentences, remainder)``. A boundary is only honoured once the
    sentence is at least ``min_chars`` long and does not end in a common
    abbreviation, which keeps "Dr." from being read as the end of a sentence.
    Text after the last honoured boundary stays buffered until more arrives.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        candidate = buffer[start:match.end()]
        stripped = candidate.strip()
        if len(stripped) < min_chars or stripped.lower().endswith(_ABBREVIATIONS):
            continue
        sentences.append(stripped)
        start = match.end()
    return sentences, buffer[start:].lstrip()


class _StreamingSpeech:
    """Speak a model's reply sentence by sentence as it streams in.

    Feeding a delta appends to a buffer and speaks whatever whole sentences are
    now available; :meth:`flush` speaks the tail once the reply is complete. The
    failure of one chunk is recorded, not raised, so it cannot tear down an
    in-flight model call — the loop reports it afterwards.

    **Adaptive coalescing.** Sentence-by-sentence speech keeps the time to the
    first word low, but it makes one TTS call per sentence. On Android the
    Termux:API round trip is not milliseconds — a call can cost seconds — so a
    four-sentence reply pays that fixed cost four times and feels laggy. This
    speaker measures its first call: if it was slow, :attr:`_min_chars` is raised
    so subsequent sentences accumulate and are spoken together at :meth:`flush`.
    A fast engine never trips the threshold, so streaming is unchanged there.
    """

    def __init__(
        self,
        tts: TextToSpeech,
        *,
        min_chars: int = 12,
        slow_call_s: float = 1.5,
        coalesce_chars: int = 220,
    ) -> None:
        self._tts = tts
        self._min_chars = max(1, min_chars)
        self._slow_call_s = slow_call_s
        self._coalesce_chars = max(self._min_chars, coalesce_chars)
        self._coalesced = False
        self._buffer = ""
        self.spoke = False
        self.error: VoiceError | None = None

    async def feed(self, delta: str) -> None:
        if self.error is not None:
            return
        self._buffer += delta
        sentences, self._buffer = split_ready_sentences(self._buffer, min_chars=self._min_chars)
        for sentence in sentences:
            await self._say(sentence)

    async def flush(self) -> None:
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder and self.error is None:
            await self._say(remainder)

    async def _say(self, text: str) -> None:
        started = time.monotonic()
        try:
            await self._tts.speak(text)
            self.spoke = True
        except VoiceError as exc:
            self.error = exc
            return
        if not self._coalesced and (time.monotonic() - started) >= self._slow_call_s:
            self._coalesced = True
            self._min_chars = self._coalesce_chars


class VoiceExit(Exception):
    """The user ended the session with a stop phrase."""


@dataclass
class VoiceConversation:
    """Drive an agent from speech in and speech out."""

    agent: Any
    stt: SpeechToText
    tts: TextToSpeech
    bus: Any = None
    console: Any = None
    wake_word: WakeWord = field(default_factory=WakeWord)
    stop_phrases: tuple[str, ...] = DEFAULT_STOP_PHRASES
    listen_timeout: float = 30.0
    max_turns: int | None = None
    max_failures: int = 3
    stream_replies: bool = True
    stream_min_chars: int = 12
    tts_slow_after_s: float = 1.5
    tts_coalesce_chars: int = 220
    greeting: str = "Prithivi is ready. Speak after the prompt."
    quit_message: str = "Goodbye."
    _last_spoken: str = field(default="", repr=False)
    _failures: int = field(default=0, repr=False)

    # -- one exchange --------------------------------------------------
    async def run_once(self) -> Any:
        """Listen for one command and answer it. Returns the turn, or None."""
        self._emit("voice.listening", {})
        self._status("listening…")
        try:
            transcript = await self.stt.listen(timeout=self.listen_timeout)
        except VoiceError as exc:
            await self._report_failure(exc)
            return None

        if transcript is None or transcript.is_empty:
            self._status("heard nothing")
            return None

        self._failures = 0
        match = self.wake_word.match(transcript.text)
        if self.wake_word.required and not match.heard:
            self._emit("voice.ignored", {"text": transcript.text, "reason": "no wake phrase"})
            return None

        command = match.command.strip()
        if not command:
            await self.tts.speak("Yes?")
            return None

        if self.wake_word.is_stop(command):
            raise VoiceExit

        if normalize(command) == normalize(self._last_spoken):
            self._emit("voice.echo", {"text": command})
            self._status("ignored an echo of my own reply")
            return None

        self._emit("voice.transcript", {"text": command, "raw": transcript.text})
        if self.console is not None:
            self.console.print(f"[bold cyan]you ›[/] {command}")

        # Try the deterministic fast path first. "open youtube" or "what time is
        # it" needs no model at all: the router recognises it and the audited
        # executor runs it, which removes the dominant model round trip and the
        # injection surface for the most common spoken commands. Only what the
        # router does not own falls through to the agent loop.
        result = await self.agent.dispatch_deterministic(command, source=TurnSource.VOICE)
        if result is not None:
            await self._respond(result.text)
            return result

        # Slow path: stream the model's answer into speech so the first sentence
        # is heard while the rest is still being generated. That is what turns a
        # long pause-then-full-reply into something that feels conversational.
        speaker = (
            _StreamingSpeech(
                self.tts,
                min_chars=self.stream_min_chars,
                slow_call_s=self.tts_slow_after_s,
                coalesce_chars=self.tts_coalesce_chars,
            )
            if self.stream_replies
            else None
        )
        result = await self.agent.run(
            command,
            source=TurnSource.VOICE,
            modality=Modality.VOICE,
            history=self._history(),
            on_text=speaker.feed if speaker is not None else None,
        )
        if speaker is not None:
            await speaker.flush()
            if speaker.error is not None:
                self._report_failure(speaker.error)
        await self._respond(result.text, already_spoken=bool(speaker and speaker.spoke))
        return result

    # -- the session ---------------------------------------------------
    async def run(self) -> int:
        """Run the loop until a stop phrase, EOF, or ``max_turns``."""
        ok, reason = await self.agent.provider.available()
        if not ok:
            self._print(f"[red]LLM unavailable:[/] {reason}")
            self._print("[dim]Set PRITHIVI_LLM_API_KEY in .env to use voice.[/]")
            return 1

        input_ok, input_reason = await self.stt.available()
        if not input_ok:
            self._print(f"[red]voice input unavailable:[/] {input_reason}")
            self._print(
                "[dim]Install the Termux:API app from F-Droid and run "
                "`pkg install termux-api`.[/]"
            )
            return 1

        if self.greeting:
            await self._safe_speak(self.greeting)

        turns = 0
        while self.max_turns is None or turns < self.max_turns:
            try:
                result = await self.run_once()
            except VoiceExit:
                await self._safe_speak(self.quit_message)
                break
            except KeyboardInterrupt:
                break
            if self._failures >= self.max_failures:
                self._print(
                    f"[red]giving up after {self._failures} voice error(s) in a row[/]"
                )
                return 1
            if result is not None:
                turns += 1
        return 0

    # -- helpers -------------------------------------------------------
    def _history(self, turns: int = 6) -> list[tuple[str, str]]:
        return [
            (entry["role"], entry["content"])
            for entry in self.agent.session.history[-turns:]
            if entry["role"] in {"user", "assistant"}
        ]

    async def _respond(self, text: str, *, already_spoken: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        self._last_spoken = text
        if self.console is not None:
            self.console.print(f"[bold green]prithivi ›[/] {text}")
        self._emit("voice.speaking", {"text": text})
        if already_spoken:
            return
        await self._safe_speak(text)

    async def _safe_speak(self, text: str) -> None:
        try:
            await self.tts.speak(text)
        except VoiceError as exc:
            self._report_failure(exc)

    async def _report_failure(self, exc: VoiceError) -> None:
        self._failures += 1
        message = str(exc)
        if exc.remedy:
            message = f"{message} {exc.remedy}"
        self._print(f"[red]{message}[/]")
        self._emit("voice.error", {"message": message})

    def _status(self, text: str) -> None:
        if self.console is not None:
            self.console.print(f"[dim]{text}[/]")

    def _print(self, text: str) -> None:
        if self.console is not None:
            self.console.print(text)

    def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.publish(topic, payload, source=TurnSource.VOICE.value)


def make_confirm_handler(
    stt: SpeechToText,
    tts: TextToSpeech,
    *,
    console: Any = None,
    listen_timeout: float = 30.0,
):
    """Build a voice second factor for the confirmation broker.

    The consequence is spoken, then one more utterance is classified with the
    same grammar the CLI uses for a typed "yes" — so a spoken and a typed answer
    are judged by identical rules (SECURITY.md S3).
    """

    async def handler(request: Any) -> ConfirmationOutcome:
        prompt = f"Confirmation needed. {request.consequence} Say yes to confirm."
        try:
            await tts.speak(prompt)
        except VoiceError as exc:
            return ConfirmationOutcome.decline(f"could not ask for confirmation: {exc}")
        if console is not None:
            console.print(f"[yellow]confirm[/] {request.consequence}")

        try:
            transcript = await stt.listen(timeout=listen_timeout)
        except VoiceError as exc:
            return ConfirmationOutcome.decline(f"could not hear the confirmation: {exc}")

        if transcript is None:
            return ConfirmationOutcome.decline("no reply heard")
        return classify_voice_confirmation(
            transcript.text, request, confidence=transcript.confidence
        )

    return handler
