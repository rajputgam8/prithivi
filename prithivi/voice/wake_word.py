"""Wake-phrase handling.

Full always-on keyword spotting (roadmap's openWakeWord) needs a local acoustic
model and an always-open microphone stream; neither fits this device's ~750 MB
free RAM, and Android gives Termux no background mic it can hold open. The
practical equivalent here is **the recogniser dialog itself** as the trigger:
``termux-speech-to-text`` opens the mic only when invoked, so the user taps to
talk.

What this module still provides, cheaply and honestly, is phrase handling on the
transcript that comes back: it detects a leading wake phrase ("hey prithivi …"),
strips it so the remainder is the command, and lets a caller *require* the phrase
(hands-free-ish gating) or pass everything through (push-to-talk).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DEFAULT_STOP_PHRASES", "WakeMatch", "WakeWord", "normalize"]

DEFAULT_WAKE_WORD = "hey prithivi"

#: Saying any of these ends a voice session.
DEFAULT_STOP_PHRASES = (
    "goodbye",
    "bye",
    "stop",
    "exit",
    "quit",
    "that's all",
    "thats all",
    "shut down",
    "shutdown",
)

#: The wake phrase may be prefixed by this many stray tokens and still count
#: ("so hey prithivi, ...").
_MAX_PREFIX_TOKENS = 3


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for phrase matching."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass(frozen=True)
class WakeMatch:
    """Result of matching one transcript against the wake phrase."""

    heard: bool
    command: str
    raw: str


class WakeWord:
    """Detect and strip a leading wake phrase.

    Args:
        phrase: the configured wake phrase.
        required: when True, a transcript without the wake phrase is ignored
            (hands-free gating). When False (default), every transcript is
            accepted and a present phrase is merely stripped.
    """

    def __init__(self, phrase: str = DEFAULT_WAKE_WORD, *, required: bool = False) -> None:
        self.phrase = phrase.strip()
        self.required = required
        self._tokens = normalize(phrase).split()

    def match(self, text: str) -> WakeMatch:
        raw = text
        tokens = normalize(text).split()
        if not tokens or not self._tokens:
            return WakeMatch(heard=False, command=text.strip(), raw=raw)

        limit = min(_MAX_PREFIX_TOKENS, max(0, len(tokens) - len(self._tokens)))
        for start in range(limit + 1):
            if tokens[start:start + len(self._tokens)] == self._tokens:
                remainder = tokens[:start] + tokens[start + len(self._tokens):]
                return WakeMatch(heard=True, command=" ".join(remainder).strip(), raw=raw)

        # Not heard. Keep the raw text (with its original case/punctuation) as the
        # command so push-to-talk does not mangle what the user said.
        return WakeMatch(heard=False, command=text.strip(), raw=raw)

    def is_stop(self, command: str) -> bool:
        normalized = normalize(command)
        return any(normalized == normalize(phrase) for phrase in DEFAULT_STOP_PHRASES)
