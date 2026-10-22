"""Voice pipeline tests (TESTING.md Phase 2).

No microphone and no Termux:API app are required: the recogniser and speaker
subprocesses are replaced with tiny shell stubs written to ``tmp_path``, so the
real argument-passing and output-parsing code paths are exercised. The agent is
scripted, so no turn reaches a network.
"""

from __future__ import annotations

import io
import stat

import pytest
from rich.console import Console

from prithivi.core.agent import Agent
from prithivi.core.events import EventBus
from prithivi.core.state import TurnSource
from prithivi.llm.provider import Completion
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.confirmation import ConfirmationBroker
from prithivi.voice import RecordingTTS, ScriptedSTT, SilentTTS, chunk_text
from prithivi.voice.base import SpeechToText, Transcript, VoiceError
from prithivi.voice.conversation import (
    VoiceConversation,
    VoiceExit,
    _StreamingSpeech,
    make_confirm_handler,
    split_ready_sentences,
)
from prithivi.voice.speech_to_text import TermuxSTT
from prithivi.voice.text_to_speech import TermuxTTS
from prithivi.voice.wake_word import WakeWord, normalize


def _stub(tmp_path, body: str) -> str:
    path = tmp_path / "stub.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(stat.S_IRWXU)
    return str(path)


def _agent(
    completions,
    registry,
    permissions,
    confirmations,
    audit,
    settings,
) -> Agent:
    return Agent(
        provider=ScriptedProvider(list(completions)),
        registry=registry,
        permissions=permissions,
        confirmations=confirmations,
        audit=audit,
        settings=settings,
        bus=EventBus(),
    )


def _console() -> Console:
    return Console(file=io.StringIO(), width=100, force_terminal=False)


# --- text chunking ---------------------------------------------------------

def test_chunk_text_passes_short_text_through() -> None:
    assert chunk_text("Hello there.") == ["Hello there."]


def test_chunk_text_is_empty_for_blank_input() -> None:
    assert chunk_text("   \n ") == []


def test_chunk_text_splits_on_sentences_within_the_limit() -> None:
    text = "One. Two. Three. Four. Five."
    chunks = chunk_text(text, limit=12)
    assert all(len(c) <= 12 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_chunk_text_never_splits_a_word() -> None:
    big = "supercalifragilistic " * 3
    chunks = chunk_text(big.strip(), limit=20)
    for chunk in chunks:
        assert chunk.split() and all(len(w) <= 20 for w in chunk.split())


# --- streaming speech ------------------------------------------------------

def test_split_ready_sentences_waits_for_a_whole_sentence() -> None:
    ready, rest = split_ready_sentences("Hello there. How are")
    assert ready == ["Hello there."]
    assert rest == "How are"


def test_split_ready_sentences_keeps_an_unfinished_clause() -> None:
    ready, rest = split_ready_sentences("just a fragment")
    assert ready == []
    assert rest == "just a fragment"


def test_split_ready_sentences_holds_abbreviations() -> None:
    ready, rest = split_ready_sentences("See e.g. the docs. Then continue")
    assert ready == ["See e.g. the docs."]
    assert rest == "Then continue"


async def test_streaming_speech_speaks_each_sentence_as_it_completes() -> None:
    tts = RecordingTTS()
    speaker = _StreamingSpeech(tts)

    await speaker.feed("First sentence. Sec")
    assert tts.spoken == ["First sentence."]

    await speaker.feed("ond sentence.")
    assert tts.spoken == ["First sentence.", "Second sentence."]

    await speaker.flush()
    assert speaker.spoke is True


async def test_a_slow_tts_engine_coalesces_later_sentences() -> None:
    """A device with a costly per-call TTS must not pay it once per sentence.

    ``slow_call_s=0`` makes the very first call count as slow, so the test is
    deterministic and does not depend on wall-clock timing.
    """
    tts = RecordingTTS()
    speaker = _StreamingSpeech(tts, slow_call_s=0.0, coalesce_chars=200)

    await speaker.feed("First sentence. ")
    assert tts.spoken == ["First sentence."]

    # The engine is known slow now, so these two short sentences are held back
    # and spoken together instead of as two more calls.
    await speaker.feed("Second sentence. Third sentence.")
    assert tts.spoken == ["First sentence."]

    await speaker.flush()
    assert tts.spoken == ["First sentence.", "Second sentence. Third sentence."]


async def test_a_fast_tts_engine_still_streams_every_sentence() -> None:
    """The optimisation must not fire when TTS is quick (the normal case)."""
    tts = RecordingTTS()
    speaker = _StreamingSpeech(tts, slow_call_s=1.5, coalesce_chars=200)

    await speaker.feed("First sentence. Second sentence. Third sentence.")
    await speaker.flush()
    assert tts.spoken == ["First sentence.", "Second sentence.", "Third sentence."]


async def test_streaming_speech_flushes_a_trailing_fragment() -> None:
    tts = RecordingTTS()
    speaker = _StreamingSpeech(tts)
    await speaker.feed("no terminator here")
    assert tts.spoken == []

    await speaker.flush()
    assert tts.spoken == ["no terminator here"]


async def test_a_streamed_reply_is_spoken_before_the_model_finishes(
    registry, permissions, confirmations, audit, settings
) -> None:
    """The whole point of streaming: speech starts on the first sentence.

    The provider hands over two sentences as two deltas, so the recorder must see
    two separate utterances — not the one blob it would see without streaming.
    """
    agent = _agent(
        [Completion(text="placeholder")], registry, permissions, confirmations, audit, settings
    )

    class StreamingProvider(ScriptedProvider):
        async def complete(
            self, messages, *, tools=(), temperature=None, max_tokens=None, model=None, on_delta=None
        ):
            self.calls.append((list(messages), tuple(t.name for t in tools)))
            for piece in ("First sentence. ", "Second sentence."):
                if on_delta is not None:
                    await on_delta(piece)
            return Completion(text="First sentence. Second sentence.")

    agent.provider = StreamingProvider([])
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["tell me a story"]), tts=tts, console=_console()
    )

    result = await convo.run_once()

    assert result is not None
    assert result.text == "First sentence. Second sentence."
    assert tts.spoken == ["First sentence.", "Second sentence."]


async def test_a_slow_voice_engine_coalesces_the_reply(
    registry, permissions, confirmations, audit, settings
) -> None:
    """End-to-end: the conversation wires the coalescing tunables through."""
    agent = _agent(
        [Completion(text="placeholder")], registry, permissions, confirmations, audit, settings
    )

    class ThreeSentenceProvider(ScriptedProvider):
        async def complete(
            self, messages, *, tools=(), temperature=None, max_tokens=None, model=None, on_delta=None
        ):
            self.calls.append((list(messages), tuple(t.name for t in tools)))
            for piece in ("First sentence. ", "Second sentence. ", "Third sentence."):
                if on_delta is not None:
                    await on_delta(piece)
            return Completion(text="First sentence. Second sentence. Third sentence.")

    agent.provider = ThreeSentenceProvider([])
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["tell me a story"]), tts=tts, console=_console(),
        tts_slow_after_s=0.0, tts_coalesce_chars=200,
    )

    await convo.run_once()

    assert tts.spoken == ["First sentence.", "Second sentence. Third sentence."]


# --- speech to text --------------------------------------------------------

async def test_termux_stt_returns_the_final_line(tmp_path) -> None:
    command = _stub(tmp_path, 'echo partial\necho "hello world"\n')
    stt = TermuxSTT(command=command)
    assert (await stt.available())[0] is True
    transcript = await stt.listen()
    assert transcript is not None
    assert transcript.text == "hello world"


async def test_termux_stt_returns_none_on_empty_output(tmp_path) -> None:
    stt = TermuxSTT(command=_stub(tmp_path, "exit 0\n"))
    assert await stt.listen() is None


async def test_termux_stt_raises_voice_error_on_failure(tmp_path) -> None:
    command = _stub(tmp_path, 'echo "recognizer crashed" >&2\nexit 3\n')
    stt = TermuxSTT(command=command)
    with pytest.raises(VoiceError) as excinfo:
        await stt.listen()
    assert excinfo.value.remedy


async def test_termux_stt_reports_unavailable_when_missing() -> None:
    stt = TermuxSTT(command="/nonexistent/termux-speech-to-text")
    ok, reason = await stt.available()
    assert ok is False
    assert "not found" in reason
    with pytest.raises(VoiceError):
        await stt.listen()


# --- text to speech --------------------------------------------------------

async def test_termux_tts_passes_rate_and_language(tmp_path) -> None:
    capture = tmp_path / "args.txt"
    command = _stub(tmp_path, f'printf "%s\\n" "$@" >> "{capture}"\n')
    tts = TermuxTTS(rate=1.5, language="en", command=command)
    await tts.speak("Hello there.")

    argv = capture.read_text().splitlines()
    assert "-r" in argv
    assert argv[argv.index("-r") + 1] == "1.50"
    assert "-l" in argv
    assert argv[-1] == "Hello there."


async def test_termux_tts_clamps_rate_to_the_engine_range(tmp_path) -> None:
    capture = tmp_path / "args.txt"
    command = _stub(tmp_path, f'printf "%s\\n" "$@" >> "{capture}"\n')
    tts = TermuxTTS(rate=99.0, command=command)
    await tts.speak("hi")
    argv = capture.read_text().splitlines()
    assert argv[argv.index("-r") + 1] == "2.00"


async def test_termux_tts_speaks_nothing_for_blank_text() -> None:
    tts = TermuxTTS(command="/nonexistent")
    await tts.speak("   ")


async def test_silent_and_recording_tts() -> None:
    await SilentTTS().speak("ignored")
    recorder = RecordingTTS()
    await recorder.speak("one")
    await recorder.speak("")
    await recorder.speak("two")
    assert recorder.spoken == ["one", "two"]


async def test_termux_tts_passes_pitch_and_voice(tmp_path) -> None:
    capture = tmp_path / "args.txt"
    command = _stub(tmp_path, f'printf "%s\\n" "$@" >> "{capture}"\n')
    tts = TermuxTTS(rate=1.0, pitch=0.9, voice="en-gb-x-gbb-local", command=command)
    await tts.speak("Hi.")
    argv = capture.read_text().splitlines()
    assert argv[argv.index("-p") + 1] == "0.90"
    assert argv[argv.index("-n") + 1] == "en-gb-x-gbb-local"


async def test_termux_tts_clamps_pitch(tmp_path) -> None:
    capture = tmp_path / "args.txt"
    command = _stub(tmp_path, f'printf "%s\\n" "$@" >> "{capture}"\n')
    await TermuxTTS(pitch=99.0, command=command).speak("hi")
    argv = capture.read_text().splitlines()
    assert argv[argv.index("-p") + 1] == "2.00"


def test_prepare_speech_text_strips_markdown_and_urls() -> None:
    from prithivi.voice import prepare_speech_text

    spoken = prepare_speech_text(
        "## Title\n**Bold** and `code`.\nSee https://docs.python.org/3/ for e.g. examples."
    )
    assert "*" not in spoken and "`" not in spoken and "https://" not in spoken
    assert "docs dot python dot org" in spoken
    assert "for example" in spoken


def test_prepare_speech_text_expands_symbols() -> None:
    from prithivi.voice import prepare_speech_text

    assert prepare_speech_text("50% off & free") == "50 percent off and free"


# --- wake word -------------------------------------------------------------

def test_wake_word_strips_a_leading_phrase() -> None:
    match = WakeWord("hey prithivi").match("Hey Prithivi, what time is it?")
    assert match.heard is True
    assert match.command == "what time is it"


def test_wake_word_passes_through_without_the_phrase() -> None:
    match = WakeWord("hey prithivi").match("open instagram")
    assert match.heard is False
    assert match.command == "open instagram"


def test_wake_word_required_ignores_a_missing_phrase() -> None:
    ww = WakeWord("hey prithivi", required=True)
    assert ww.match("do the thing").heard is False
    assert ww.match("hey prithivi do the thing").heard is True


def test_wake_word_detects_stop_phrases() -> None:
    ww = WakeWord("hey prithivi")
    assert ww.is_stop("Goodbye!") is True
    assert ww.is_stop("what time is it") is False


def test_normalize_collapses_punctuation_and_case() -> None:
    assert normalize("Hey, PRITHIVI!!") == "hey prithivi"


# --- conversation ----------------------------------------------------------
async def test_a_spoken_command_runs_a_turn_and_speaks_the_answer(
    registry, permissions, confirmations, audit, settings
) -> None:
    # A conversational request, deliberately not one the deterministic fast path
    # owns: this test is about the *model* turn being spoken. The fast path has
    # its own coverage below.
    agent = _agent(
        [Completion(text="A monad is a design pattern for chaining computations.")],
        registry, permissions, confirmations, audit, settings,
    )
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["explain what a monad is"]), tts=tts, console=_console()
    )

    result = await convo.run_once()

    assert result is not None
    assert result.text == "A monad is a design pattern for chaining computations."
    assert tts.spoken == ["A monad is a design pattern for chaining computations."]


async def test_a_fast_path_command_is_answered_and_spoken_without_the_model(
    registry, permissions, confirmations, audit, settings
) -> None:
    """"what time is it" must not cost a model round trip.

    The router answers it deterministically; the model is never consulted (which
    is why ``provider.calls`` stays empty), and the honest result is still read
    aloud. This is both the latency path and the injection-proof one.
    """
    agent = _agent([], registry, permissions, confirmations, audit, settings)
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["what time is it"]), tts=tts, console=_console()
    )

    result = await convo.run_once()

    assert result is not None
    assert not agent.provider.calls
    assert tts.spoken == [result.text]


async def test_the_turn_is_recorded_as_voice(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent(
        [Completion(text="ok")], registry, permissions, confirmations, audit, settings
    )
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["hello"]), tts=RecordingTTS(), console=_console()
    )
    await convo.run_once()

    starts = [r for r in audit.all() if r.action == "turn.start"]
    assert starts
    assert starts[0].detail["source"] == TurnSource.VOICE.value


async def test_the_wake_phrase_is_not_sent_to_the_agent(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent(
        [Completion(text="opening")], registry, permissions, confirmations, audit, settings
    )
    convo = VoiceConversation(
        agent=agent,
        stt=ScriptedSTT(["hey prithivi tell me about monads"]),
        tts=RecordingTTS(),
        console=_console(),
    )
    await convo.run_once()

    messages, _tools = agent.provider.calls[-1]
    assert messages[-1].content == "tell me about monads"


async def test_heard_nothing_is_a_no_op(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent([], registry, permissions, confirmations, audit, settings)
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT([None]), tts=RecordingTTS(), console=_console()
    )
    assert await convo.run_once() is None
    assert not agent.provider.calls


async def test_echo_guard_ignores_the_assistants_own_reply(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent(
        [Completion(text="The answer is 42.")],
        registry, permissions, confirmations, audit, settings,
    )
    convo = VoiceConversation(
        agent=agent,
        stt=ScriptedSTT(["question one", "The answer is 42."]),
        tts=RecordingTTS(),
        console=_console(),
    )
    await convo.run_once()
    second = await convo.run_once()

    assert second is None
    assert len(agent.provider.calls) == 1


async def test_a_stop_phrase_ends_the_session(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent([], registry, permissions, confirmations, audit, settings)
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["goodbye"]), tts=tts, console=_console()
    )

    with pytest.raises(VoiceExit):
        await convo.run_once()


async def test_run_speaks_a_greeting_and_stops_on_goodbye(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent(
        [Completion(text="Hello.")], registry, permissions, confirmations, audit, settings
    )
    tts = RecordingTTS()
    convo = VoiceConversation(
        agent=agent,
        stt=ScriptedSTT(["hi", "goodbye"]),
        tts=tts,
        console=_console(),
        greeting="ready",
        quit_message="bye now",
    )

    code = await convo.run()

    assert code == 0
    assert tts.spoken[0] == "ready"
    assert tts.spoken[-1] == "bye now"


async def test_run_reports_when_voice_input_is_unavailable(
    registry, permissions, confirmations, audit, settings
) -> None:
    agent = _agent([], registry, permissions, confirmations, audit, settings)
    convo = VoiceConversation(
        agent=agent,
        stt=TermuxSTT(command="/nonexistent/stt"),
        tts=RecordingTTS(),
        console=_console(),
    )
    assert await convo.run() == 1


async def test_run_reports_when_the_llm_is_unavailable(
    registry, permissions, confirmations, audit, settings
) -> None:
    class Unavailable(ScriptedProvider):
        async def available(self):
            return False, "no key"

    agent = _agent([], registry, permissions, confirmations, audit, settings)
    agent.provider = Unavailable([])
    convo = VoiceConversation(
        agent=agent, stt=ScriptedSTT(["hi"]), tts=RecordingTTS(), console=_console()
    )
    assert await convo.run() == 1


async def test_run_gives_up_after_repeated_voice_errors(
    registry, permissions, confirmations, audit, settings
) -> None:
    class BrokenSTT(SpeechToText):
        name = "broken"

        async def listen(self, *, timeout=None):
            raise VoiceError("recognizer exploded", remedy="reinstall it")

    agent = _agent([], registry, permissions, confirmations, audit, settings)
    convo = VoiceConversation(
        agent=agent, stt=BrokenSTT(), tts=RecordingTTS(), console=_console(), max_failures=2
    )
    assert await convo.run() == 1


# --- voice confirmation ----------------------------------------------------

async def _request():
    broker = ConfirmationBroker()
    return broker.issue(
        tool="open_application", args={"application": "chrome"}, level=2,
        consequence="Open the Chrome app",
    )


async def test_voice_confirmation_approves_when_the_action_is_named() -> None:
    handler = make_confirm_handler(
        ScriptedSTT(["yes open chrome"]), RecordingTTS()
    )
    outcome = await handler(await _request())
    assert outcome.approved is True
    assert outcome.factor == "voice"


async def test_voice_confirmation_declines_a_plain_no() -> None:
    handler = make_confirm_handler(ScriptedSTT(["no"]), RecordingTTS())
    outcome = await handler(await _request())
    assert outcome.approved is False


async def test_voice_confirmation_declines_low_confidence() -> None:
    stt = ScriptedSTT([Transcript("yes open chrome", confidence=0.1)])
    handler = make_confirm_handler(stt, RecordingTTS())
    outcome = await handler(await _request())
    assert outcome.approved is False


async def test_voice_confirmation_declines_when_nothing_is_heard() -> None:
    handler = make_confirm_handler(ScriptedSTT([None]), RecordingTTS())
    outcome = await handler(await _request())
    assert outcome.approved is False
    assert "no reply" in outcome.reason
