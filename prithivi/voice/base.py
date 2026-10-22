"""Voice pipeline contracts and shared helpers (ARCHITECTURE.md S12).

Speech in and speech out sit behind two tiny protocols so the conversation loop
never learns whether it is talking to Termux:API, a cloud recogniser, or a test
double. Failures are :class:`VoiceError` carrying a user-facing ``remedy`` — a
missing microphone or recogniser must produce a spoken explanation, never a
silent no-op that looks like the assistant ignored you.
"""

from __future__ import annotations

import abc
import os
import shutil
from dataclasses import dataclass

__all__ = [
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
    "VoiceError",
    "find_termux_command",
]

#: Termux installs its binaries here; the real Android tools live in /system/bin.
_TERMUX_PREFIX = "/data/data/com.termux/files/usr"


class VoiceError(RuntimeError):
    """Speech could not be captured or produced.

    ``remedy`` is the sentence the caller speaks or prints next, so the failure
    tells the user what to do (install Termux:API, grant the mic permission).
    """

    def __init__(self, message: str, *, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


@dataclass(frozen=True)
class Transcript:
    """One recognised utterance.

    ``confidence`` is ``1.0`` when the backend does not report one — Termux's
    recogniser does not, so voice confirmations treat the words as confident and
    the stricter gate remains the typed CLI.
    """

    text: str
    confidence: float = 1.0
    language: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def find_termux_command(name: str) -> str | None:
    """Locate a Termux:API command on ``PATH`` or in the Termux prefix."""
    found = shutil.which(name)
    if found:
        return found
    prefix = os.environ.get("PREFIX") or _TERMUX_PREFIX
    candidate = os.path.join(prefix, "bin", name)
    return candidate if os.path.isfile(candidate) else None


class SpeechToText(abc.ABC):
    """Capture one utterance and return its text."""

    name = "abstract"

    @abc.abstractmethod
    async def listen(self, *, timeout: float | None = None) -> Transcript | None:
        """Return the next utterance, or ``None`` when nothing was heard."""

    async def available(self) -> tuple[bool, str]:
        """``(ok, reason)`` readiness probe for ``prithivi doctor``."""
        return True, "ok"

    async def aclose(self) -> None:
        return None


class TextToSpeech(abc.ABC):
    """Speak text aloud."""

    name = "abstract"

    @abc.abstractmethod
    async def speak(self, text: str) -> None:
        """Speak ``text``; an empty string is a no-op."""

    async def available(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None
