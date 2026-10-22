"""Speech-to-text backends.

The default is Android's own recogniser reached through Termux:API
(``termux-speech-to-text``), which pops the system speech dialog and prints the
final transcript. It is free, on-device, and needs no model download — the only
viable option on a phone with roughly 750 MB free RAM.

That command exposes **no language flag**: the recognition language is the
device's system locale. ``PRITHIVI_STT_LANGUAGE`` is therefore advisory for this
backend and cannot be forced; a different locale is an Android setting, not one
this process can change.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Sequence

from .base import SpeechToText, Transcript, VoiceError, find_termux_command

__all__ = ["ScriptedSTT", "TermuxSTT"]

_INSTALL_REMEDY = (
    "Install the Termux:API app from F-Droid and run `pkg install termux-api`, "
    "then run Prithivi in native Termux (not inside proot)."
)


def _last_nonempty_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        if line.strip():
            return line.strip()
    return ""


class TermuxSTT(SpeechToText):
    """Android speech recognition via ``termux-speech-to-text``."""

    name = "termux"

    def __init__(self, *, timeout: float = 30.0, command: str | None = None) -> None:
        self._command = command if command is not None else find_termux_command("termux-speech-to-text")
        self._timeout = timeout

    async def available(self) -> tuple[bool, str]:
        if self._command is None or not os.path.isfile(self._command):
            return False, "termux-speech-to-text not found"
        return True, "android speech recognizer via Termux:API"

    async def listen(self, *, timeout: float | None = None) -> Transcript | None:
        if self._command is None:
            raise VoiceError("speech recognition is unavailable", remedy=_INSTALL_REMEDY)
        budget = self._timeout if timeout is None else timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                self._command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VoiceError("speech recognition is unavailable", remedy=_INSTALL_REMEDY) from exc

        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=budget)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return None

        if proc.returncode != 0:
            detail = (err or out).decode(errors="replace").strip() or f"exit status {proc.returncode}"
            raise VoiceError(
                f"speech recognition failed: {detail}",
                remedy="Check that the Termux:API app can use the microphone, then try again.",
            )

        text = _last_nonempty_line(out.decode(errors="replace"))
        if not text:
            return None
        return Transcript(text=text)


class ScriptedSTT(SpeechToText):
    """Replays a fixed list of utterances — tests and offline demos, no mic.

    An item of ``None`` (or :meth:`push_none`) queues a "heard nothing" result.
    """

    name = "scripted"

    def __init__(self, transcripts: Sequence[str | Transcript | None] = ()) -> None:
        self._items: list[Transcript | None] = [_coerce(item) for item in transcripts]
        self._index = 0
        self.received: list[Transcript | None] = []

    def push(self, item: str | Transcript | None) -> None:
        self._items.append(_coerce(item))

    def push_none(self) -> None:
        """Queue a "heard nothing" result."""
        self._items.append(None)

    @property
    def remaining(self) -> int:
        return max(0, len(self._items) - self._index)

    async def listen(self, *, timeout: float | None = None) -> Transcript | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        self.received.append(item)
        return item


def _coerce(item: str | Transcript | None) -> Transcript | None:
    if item is None or isinstance(item, Transcript):
        return item
    return Transcript(item)
