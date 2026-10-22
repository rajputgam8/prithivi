# PRITHIVI — Roadmap & Phase Exit Criteria

Every phase ends with: **what was built · files · how to run · how to test · known limitations · explicit user confirmation before continuing.**

Rule (brief §31.11/§31.12): a phase is only "done" when the listed tests actually pass. No feature is claimed without a test.

---

## Phase 0 — Architecture ✅ (this document set)

Deliverables: `ARCHITECTURE.md`, `docs/DATA_MODEL.md`, `docs/SECURITY.md`, `docs/TESTING.md`, `docs/ROADMAP.md`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, folder skeleton, `README.md` pointer, `CONTRIBUTING.md`.
Exit: all 20 requested artefacts exist and are internally consistent. **No functional code.**

---

## Phase 1 — Basic PRITHIVI (text only)

Scope:
- `core/config.py` (pydantic-settings), `core/events.py` (typed bus), `core/state.py`
- `llm/provider.py` protocol + `openai_compat.py` (works against OpenAI, Groq, Together, Ollama, LM Studio, vLLM) + `anthropic.py`; `llm/prompts.py`
- `tools/base.py` (`ToolSpec`, `ToolResult`, `@tool`), `tools/registry.py` (allowlist)
- Three real tools: `get_time`, `calculate` (safe eval-free arithmetic parser), `open_url`
- `core/router.py` (fast path + LLM structured classification), `core/agent.py` (bounded ReAct loop), `core/narrator.py` (provenance)
- `security/permissions.py` (L0–L4 + policy), `security/audit.py` (hash chain), `security/secrets.py` (redaction), `security/confirmation.py`
- `interface/cli.py` (REPL + streaming + tool tracing), `main.py`
- `observability/logging.py` (the `[22:41:05] Intent: …` format from brief §23)
- `pyproject.toml`, `pytest` suite

Acceptance: `python main.py chat` → typing `what is Python?` returns a real LLM answer with provenance; `open https://…` invokes `open_url` through the permission engine and logs it; unknown tool name from the model is rejected; audit chain verifies.

Explicitly **out of scope** in Phase 1: voice, vision, DB, RAG, web search, browser automation, UI.

---

## Phase 2 — Voice

Scope: `voice/audio_manager.py`, `wake_word.py` (openWakeWord), `speech_to_text.py` (faster-whisper), `text_to_speech.py` (piper), `conversation.py` (barge-in, echo guard), CLI `voice` mode.
Acceptance: real mic → wake word → STT → agent → TTS; "PRITHIVI, open Chrome" routes to `open_application` if the tool exists, otherwise reports honestly that the tool is not implemented yet (no fake success); configurable voice/speed/language; `en` + `hi`/Hinglish mode.
Hardware-dependent tests marked `-m hardware`, skipped in CI.

---

## Phase 3 — Computer control

Scope: `platform/` adapters (linux/macos/windows), `tools/app_control.py`, `browser.py` (Playwright), `filesystem.py`, `system.py` (volume, media, lock, screenshots via `mss`).
Acceptance: launch a real app; open a URL in a real browser; read/write only inside configured roots (traversal denied); screenshot saved and reported with evidence; volume change verified by re-reading the actual state.
Safety: L2 for writes, L3 for deletes, root confinement enforced in the executor.

---

## Phase 4 — Vision core

Scope: `vision/camera.py`, `hand_tracking.py` (MediaPipe Hand Landmarker), `gesture_classifier.py`, `gesture_fsm.py`.
Acceptance: from a recorded video fixture, landmarks and static gestures classify correctly above threshold; FSM refuses transitions that violate hold/cooldown; CPU budget respected (drops frames instead of queueing).
Fallback path: OpenCV contour heuristics when MediaPipe is unavailable, behind the same interface.

---

## Phase 5 — Advanced gestures

Scope: `vision/cursor.py` (smoothing/acceleration), pinch-click, swipe, zoom, drag, rotation; calibration + sensitivity config; `tools/gesture_tools.py`.
Acceptance: pinch moves cursor and clicks at the right screen coordinate (verified against a test window); swipe prev/next is one-shot latched; zoom maps to a real value change; **every gesture-originated L2+ action is refused without a voice/UI second factor** (test asserted).

---

## Phase 6 — Web intelligence *(moved forward; see ARCHITECTURE §18)*

Scope: `llm`-agnostic `SearchProvider` (serper/brave/tavily/searxng), `tools/search.py`, `web_fetch.py` (trafilatura), `rag/web_research.py`, `rag/citations.py`, `security/injection.py`, egress policy.
Acceptance: a real query returns a real sourced answer with ≥2 sources and working citations; single-source claims are labelled unverified; contradictions surfaced; injected instructions in a fetched page cannot cause a tool call; offline mode refuses all egress.

---

## Phase 7 — RAG + memory

Scope: `db/` (schema, migrations, repositories), `memory/*`, `rag/ingestion.py`, `chunking.py`, `embeddings.py` (fastembed), `retrieval.py` (hybrid + RRF), `knowledge_base.py`, `tools/memory_tools.py`, `tools/rag_tools.py`, `interface/cli.py memory` subcommands, `api/server.py`.
Acceptance: ingest a real PDF → ask a question whose answer only exists in the document → answer cites `uri#Lchunk`; "remember I prefer Python" stores a structured row and the next session uses it; delete removes rows + vectors; memory can be disabled entirely.

---

## Phase 8 — Social + productivity

Scope: OAuth manager (PKCE, minimal scopes, encrypted refresh tokens), `tools/social_media.py` (YouTube, X, Discord, LinkedIn, Instagram, Spotify per official API availability), `tools/productivity.py` (calendar, notes, tasks, timers).
Acceptance: connect a real account via official OAuth; read-only actions work; **every** write/post/DM requires confirmation (tested); API errors surfaced with rate-limit handling; no ToS/CAPTCHA/MFA bypass anywhere.
Note: some platforms (Instagram, LinkedIn) have restrictive APIs — features will be limited to what the official API permits, and that limit will be stated plainly rather than worked around.

---

## Phase 9 — Automation

Scope: `automation/scheduler.py` (APScheduler), `triggers.py` (cron/interval/date/window-focus/device-connect/file-change), `workflows.py` (DAG steps), `conditions.py`, `tools/automation_tools.py`, proactive `notifications` with levels OFF/LOW/NORMAL/HIGH.
Acceptance: a real cron automation fires at the scheduled time and produces a real answer; reminders fire once (dedupe); each run re-checks permissions and is audited; proactive notifications respect level and dedupe keys (no spam); disabling an automation stops it.

---

## Phase 10 — Futuristic UI + hardening

Scope: `interface/ui/` (React + TS + Vite + three.js): AI core, voice waveform, system status, active tool indicator, memory/network/task indicators, notifications, conversation, confirm modal; `interface/web/` FastAPI REST + WebSocket event protocol; CLI HUD polish; packaging (`pyproject` build, single `prithivi` entrypoint); performance pass (frame budget, lazy model load, caching); full documentation set.
Acceptance: all 8 agent states render distinctly; 60 fps with vision running; reduced-motion honoured; WS reconnect works; confirmation modal blocks action until decided; full test suite green; docs complete (`API.md`, `TOOLS.md`, `MEMORY.md`, `GESTURES.md`, `SETUP.md`, `CONTRIBUTING.md`).

---

## Dependencies (by phase)

| Phase | New runtime dependencies |
|---|---|
| 1 | `pydantic`, `pydantic-settings`, `httpx`, `rich`, `python-dotenv`; dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `respx`, `ruff`, `mypy`, `hypothesis` |
| 2 | `faster-whisper`, `openwakeword`, `piper-tts` (or `pyttsx3` fallback), `webrtcvad` |
| 3 | `playwright`, `mss`, `pynput`, `platformdirs`, `psutil` |
| 4 | `opencv-python`, `mediapipe` |
| 5 | — (reuses 4) |
| 6 | `trafilatura`, `beautifulsoup4`, `duckduckgo-search`(optional), search-provider client (`httpx` suffices) |
| 7 | `aiosqlite`, `sqlite-vec`, `fastembed`, `pypdf`, `markdown`, `python-docx`, `tiktoken`(or `tokenizers`), `chromadb`(optional alternative) |
| 8 | `authlib` or raw `httpx` OAuth2 PKCE client, provider SDKs only where necessary |
| 9 | `apscheduler` |
| 10 | `fastapi`, `uvicorn`, `websockets`; frontend: `react`, `typescript`, `vite`, `three` |

Full pinned lists live in `requirements*.txt`.

---

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| MediaPipe/OpenCV unavailable on some platforms (e.g. aarch64 Android) | Phase 4–5 blocked on that host | `HandLandmarker` interface + OpenCV fallback; develop on Linux desktop |
| API key absent | Nothing works in cloud mode | Local model path (llama.cpp/Ollama) documented as a first-class option; clear startup error naming the missing var |
| Social APIs are restrictive | Phase 8 scope shrinks | Ship only what official APIs allow; document limits; never work around them |
| Model chooses wrong tool | Poor UX | Negative-case tool descriptions, fast-path router, constrained decoding, structured outputs, and narration of the actual result |
| Context growth in long sessions | Cost/latency | Rolling summaries, retrieval over history, bounded memories, token budgets |
| Permission fatigue | User rubber-stamps everything | Narrow grants with use counts, explicit L4 phrasing, decline memory, audit review command |
| Platform API drift | Breaks phases 3/4 | All OS access behind `platform/base.py` adapters; contract tests with recorded fixtures |