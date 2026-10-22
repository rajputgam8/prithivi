# PRITHIVI

**Advanced Neural Task Runner & Assistant** — a modular, multimodal personal AI operating layer.
Voice, computer control, vision/gestures, web research, RAG, persistent memory, and automation — built as a real, testable system rather than a demo.

> **Status: Phases 1, 2 and 6 implemented** — text agent + LLM providers + tool system,
> voice (speech in/out on Android via Termux:API), and web intelligence
> (live search, page reading, cited research with cross-checked sources). Vision, RAG, and
> automation are later phases (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).

---

## What this project is

PRITHIVI behaves like a personal AI operating system rather than a chatbot:

```
VOICE / TEXT / CAMERA
        ↓
PERCEPTION  →  INTENT  →  PLANNER  →  TOOL SELECTION  →  ACTION
                                                          ↓
RESPONSE  ←  REASONING  ←  OBSERVATION  ←───────────────
```

Three rules shape everything:

1. **The model never holds authority.** It proposes a tool call; the permission engine decides whether it may run.
2. **No fake results.** A tool returns typed evidence or an explicit error. PRITHIVI never claims an action worked without proof.
3. **Learning is retrieval, never retraining.** New knowledge lands in a vector store with a source URL and timestamp, so answers stay auditable ("I already knew this" vs "I looked this up").

---

## Documentation

| Document | Contents |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System + component diagram, data flow, stack, folder structure, memory/agent/tool/security/voice/vision/RAG/web-research/UI architecture, env vars, implementation order |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | Full schema: sessions, messages, tool calls, memories, RAG chunks, vectors, automations, security tables, retention |
| [`docs/SECURITY.md`](docs/SECURITY.md) | L0–L4 permission model, enforcement chain, confirmation mechanics, 14-entry threat model, auth, secret management, gesture safety invariants |
| [`docs/TESTING.md`](docs/TESTING.md) | Test layers, fixtures, scripted-LLM technique, per-phase test matrix, coverage gates, hardware checklist |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases 0–10 with scope, acceptance criteria, dependencies, risks |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev rules, commit style, how to add a tool safely |

Planned for later phases: `SETUP.md`, `API.md`, `TOOLS.md`, `MEMORY.md`, `GESTURES.md`.

---

## Roadmap at a glance

| Phase | Capability | Acceptance |
|---|---|---|
| 0 | Architecture | ✅ this document set |
| 1 | Text agent + LLM providers + tool system | ✅ implemented |
| 2 | Voice (wake word, STT, TTS) | ✅ implemented — speech in/out via Termux:API on Android; "PRITHIVI, open Chrome" is handled honestly |
| 3 | Computer control (apps, browser, files, system) | launches, navigates, screenshots — sandboxed |
| 4 | Vision core (camera, landmarks, gesture FSM) | gestures classified safely from fixtures |
| 5 | Advanced gestures (cursor, pinch, swipe, zoom) | pinch-clicks accurately; gestures never authorise L2+ |
| 6 | Web intelligence (search, fetch, citations) | ✅ implemented — current-info answers with real sources |
| 7 | RAG + long-term memory | cited answers from your PDFs; preferences persist |
| 8 | Social + productivity via official OAuth | read-only freely; writes always confirmed |
| 9 | Automation (scheduler, triggers, workflows) | "every morning at 8, brief me" actually fires |
| 10 | Futuristic UI (AI core, waveform, HUD) | all 8 agent states rendered, 60 fps |

---

## Quick start

```bash
git clone <repo> && cd prithivi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then add PRITHIVI_LLM_API_KEY
python main.py chat
```

```
you: what is a monad in functional programming?
prithivi: A monad is ... [parametric knowledge]
you: open https://docs.python.org
prithivi: Opening docs.python.org.  [tool: open_url · L1]
you: what is the latest python version and when was it released?
prithivi: Python 3.14 ships a free-threaded build. [1] Two sources agree; ...
```

Web research is on by default with the keyless DuckDuckGo provider
(`PRITHIVI_SEARCH_PROVIDER=duckduckgo`). Ask the agent anything current, run
`/research <question>` in the CLI for a cited, cross-checked answer, or say
"search for X" / "read this page".

- Providers: `duckduckgo` (default, no key), `serper`, `brave`, `tavily`, `searxng`.
- Add a key: `PRITHIVI_SEARCH_API_KEY`, and for `searxng` set `PRITHIVI_SEARCH_ENDPOINT`.
- `PRITHIVI_EGRESS_ALLOWED_HOSTS` (empty = any public host) lists the hosts pages
  may be fetched from; private/loopback addresses are always refused, and
  `PRITHIVI_EGRESS_ALLOW_NETWORK=false` makes everything fully offline.
- One-source facts are labelled unverified, contradictions are surfaced, and anything a
  page says ("ignore your instructions…") is treated as data, never as an order.

---

## Voice mode

`python main.py voice` listens, runs the turn through the same permission engine as
text, and speaks the honest result. On Android it uses the phone's own recogniser
and TTS through [Termux:API](https://wiki.termux.com/wiki/Termux:API) — nothing
heavy is downloaded.

```bash
# one-time: install the Termux:API app from F-Droid, then
pkg install termux-api
python main.py voice
```

- Say "hey prithivi …" to prefix a command (the phrase is stripped), or just speak
  — push-to-talk, because Android gives Termux no always-open microphone.
- Say "goodbye" (or "stop", "bye") to end the session.
- An L2+ action is confirmed by voice: the consequence is spoken, then your
  "yes" is checked against the same grammar as a typed one.
- **Replies stream into speech.** The model's answer is spoken sentence by
  sentence as it is generated (`PRITHIVI_LLM_STREAM=true`), so the first words
  start almost immediately instead of after the whole reply — the biggest
  perceived-latency win, and the reason fast-path commands (`what time is it`)
  answer with no model round trip at all.
- **Slow speech engines coalesce.** A Termux:API call has a fixed cost (seconds
  when Prithivi runs inside proot, milliseconds natively). When the first spoken
  chunk is slow, later sentences are merged into larger utterances
  (`PRITHIVI_TTS_SLOW_AFTER_S`, `PRITHIVI_TTS_COALESCE_CHARS`) so that fixed cost
  is paid once instead of once per sentence. A fast engine is unaffected.
- **The reply is cleaned for the ear, not the eye.** Markdown emphasis, code,
  bullets and raw URLs are stripped before speaking (`**Bold**` is not read as
  "star star bold"), and `%`/`&`/`e.g.` are spoken naturally.
- **Voice character is tunable.** `PRITHIVI_TTS_RATE` and `PRITHIVI_TTS_PITCH`
  (0.5–2.0) shape speed and tone, and `PRITHIVI_TTS_VOICE` / `PRITHIVI_TTS_ENGINE`
  select a specific engine voice when the device offers one.
- True always-on wake-word detection and mid-sentence barge-in need a local
  acoustic model and a persistent mic stream; neither fits this device, and both
  are stated rather than faked.
- **Run voice mode in native Termux, not inside proot.** Opening apps and the
  browser goes through Android's window manager, which PRoot intercepts — so
  "open YouTube" fails there even though the bridge exists. The `~/prithivi`
  launcher (created for you in `$HOME`) runs the project with native Termux's
  python; reach for that when you want app control. `doctor` and the startup
  banner say so explicitly instead of failing cryptically.

---

## Requirements

- Python 3.11–3.13 (3.12 recommended)
- Android + [Termux](https://termux.dev) for voice (Termux:API); a Linux desktop for
  full functionality (vision needs MediaPipe/OpenCV wheels)
- Optionally a local LLM (Ollama / llama.cpp) so PRITHIVI can run fully offline

---

## License

To be chosen by the project owner. All code is original work.