"""Voice interface (roadmap Phase 2).

Speech in and speech out behind two small protocols, plus the conversation loop
that turns a spoken command into an agent turn and speaks the honest result.
The default backends are Android's own recogniser and TTS reached through
Termux:API, so nothing heavy is downloaded onto a low-RAM phone.
"""

from __future__ import annotations

from typing import Any

from .base import SpeechToText, TextToSpeech, Transcript, VoiceError
from .conversation import VoiceConversation, VoiceExit, make_confirm_handler
from .speech_to_text import ScriptedSTT, TermuxSTT
from .text_to_speech import (
    RecordingTTS,
    SilentTTS,
    TermuxTTS,
    chunk_text,
    prepare_speech_text,
)
from .wake_word import WakeWord

__all__ = [
    "RecordingTTS",
    "ScriptedSTT",
    "SilentTTS",
    "SpeechToText",
    "TermuxSTT",
    "TermuxTTS",
    "TextToSpeech",
    "Transcript",
    "VoiceConversation",
    "VoiceError",
    "VoiceExit",
    "WakeWord",
    "build_stt",
    "build_tts",
    "chunk_text",
    "make_confirm_handler",
    "prepare_speech_text",
]

_TEXT_TO_SPEECH_LANGUAGES = {"en", "hi"}


def build_stt(settings: Any) -> SpeechToText:
    """Build the configured speech-to-text backend.

    Only the Termux backend ships today; an unknown provider is reported rather
    than silently falling back to a microphone the user did not choose.
    """
    provider = str(getattr(settings, "stt_provider", "termux") or "termux").strip().lower()
    if provider in {"termux", "termux_api"}:
        return TermuxSTT()
    if provider in {"scripted", "fake"}:
        return ScriptedSTT()
    raise VoiceError(
        f"unsupported STT provider {provider!r}",
        remedy="Use PRITHIVI_STT_PROVIDER=termux.",
    )


def build_tts(settings: Any) -> TextToSpeech:
    """Build the configured text-to-speech backend."""
    provider = str(getattr(settings, "tts_provider", "termux") or "termux").strip().lower()
    rate = getattr(settings, "tts_rate", 1.0)
    pitch = getattr(settings, "tts_pitch", 1.0)
    voice = str(getattr(settings, "tts_voice", "") or "")
    engine = str(getattr(settings, "tts_engine", "") or "")
    language = str(getattr(settings, "stt_language", "auto") or "auto").strip().lower()
    tts_language = language if language in _TEXT_TO_SPEECH_LANGUAGES else ""
    if provider == "termux":
        return TermuxTTS(
            rate=rate, pitch=pitch, language=tts_language, voice=voice, engine=engine
        )
    if provider in {"silent", "none"}:
        return SilentTTS()
    if provider in {"scripted", "recording"}:
        return RecordingTTS()
    raise VoiceError(
        f"unsupported TTS provider {provider!r}",
        remedy="Use PRITHIVI_TTS_PROVIDER=termux.",
    )
