# PRITHIVI — Phase 0: System Architecture

Status: **DESIGN ONLY — no implementation code in this phase.**
Version: 0.1 · Target platform: Linux desktop (primary), macOS/Windows (adapters), Android/Termux (limited).

---

## 1. Core philosophy (normative rules for every future module)

1. **The agent never executes model-authored code.** The LLM may only emit a tool name + JSON arguments that matches a registered `ToolSchema`. Anything else is rejected.
2. **No fake results.** Tools return typed `ToolResult` with `success`, `data|error`, and `evidence`. The narrator may only claim success if `evidence` is present.
3. **Capability is added at the edges, never the centre.** New tools/providers/sensors register against stable interfaces. `core/` does not change when a tool is added.
4. **Degrade, don't lie.** If a sensor/provider is unavailable, the capability reports `UNAVAILABLE` with a reason. No silent stubs.
5. **Everything is auditable.** Every routed intent, tool call, confirmation decision, and permission change is written to the audit log.
6. **Local-first.** Everything with a viable local implementation runs locally by default; cloud is opt-in per capability.

---

## 2. Component diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYER                             │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │ Desktop HUD   │  │ Web UI (React) │  │ Voice CLI     │  │ FastAPI   │  │
│  │ (PySide6)     │  │ WebGL core     │  │ (REPL/TUI)    │  │ REST/WS   │  │
│  └───────┬───────┘  └────────┬───────┘  └───────┬───────┘  └─────┬─────┘  │
│          │  Local IPC        │  WebSocket        │  stdin          │ HTTP  │
└──────────┼───────────────────┼───────────────────┼────────────────┼────────┘
           │                   │                   │                │
┌──────────▼───────────────────▼───────────────────▼────────────────▼────────┐
│                        EVENT BUS  (asyncio, typed topics)                  │
│  perception.* · intent.* · agent.* · tool.* · memory.* · rag.* · ui.*       │
└──────────┬──────────────────────────────┬──────────────────────────────────┘
           │                              │
┌──────────▼──────────────┐   ┌───────────▼────────────────────────────────┐
│   PERCEPTION LAYER      │   │              BRAIN (core/)                │
│  ┌────────┐ ┌─────────┐ │   │  ┌────────────┐  ┌──────────────┐          │
│  │ Voice  │ │ Vision  │ │   │  │ Normalizer │→ │ Intent Router│          │
│  │ VAD    │ │ Camera  │ │   │  └────────────┘  └──────┬───────┘          │
│  │ Wake   │ │ Hands   │ │   │                        │                  │
│  │ STT    │ │ Face    │ │   │              ┌─────────▼─────────┐        │
│  │ TTS    │ │ Screen  │ │   │              │  Agent Loop      │        │
│  └────┬───┘ └────┬────┘ │   │              │  (ReAct, async)  │        │
│       │          │      │   │              └─────────┬─────────┘        │
│  ┌────▼──────────▼────┐ │   │        ┌───────────────┼──────────────┐   │
│  │ Gesture FSM        │ │   │        │               │              │   │
│  │ thr/debounce/cool  │ │   │  ┌─────▼──────┐ ┌──────▼─────┐ ┌──────▼────┐│
│  └────────────────────┘ │   │  │  Planner   │ │ Tool Router │ │ Narrator  ││
└─────────────────────────┘   │  │ (optional) │ │ allowlist   │ │ (voice/UI) ││
                              │  └────────────┘ └──────┬─────┘ └───────────┘│
                              └────────────────────────┼────────────────────┘
                                                        │
┌───────────────────────────────────────────────────────▼────────────────────┐
│                            TOOL REGISTRY (allowlist)                       │
│  ToolSpec{id, version, input_model, permission_level, side_effects,       │
│            timeout_s, idempotent, requires_confirmation, executor}          │
│  app_control · browser · filesystem · system · web_search · web_fetch ·   │
│  youtube · social · productivity · memory_* · rag_* · automation_* ·      │
│  gesture_* · vision_describe                                              │
└──────────┬─────────────────────────────┬────────────────────────┬─────────┘
           │                             │                        │
┌──────────▼───────────┐  ┌──────────────▼───────┐  ┌─────────────▼────────┐
│ MEMORY (core)        │  │ RAG / KNOWLEDGE      │  │ AUTOMATION           │
│  short_term (RAM)    │  │  chunker · embedder  │  │  scheduler (APScheduler│
│  session  (SQLite)   │  │  vector store (sqlite-│ │   / custom asyncio)   │
│  long_term (SQLite)  │  │   vec or pgvector)    │  │  triggers (fs/window/ │
│  profile (key-value) │  │  retriever (hybrid)    │  │   device/voice/sched)│
│  summarizer          │  │  citation builder      │  │  workflow (DAG steps)│
└──────────────────────┘  └───────────────────────┘  └─────────────────────┘
           │                             │                        │
┌──────────▼─────────────────────────────▼────────────────────────▼─────────┐
│                        SECURITY & PLATFORM SERVICES                       │
│  permission engine · confirmation broker · auth/session · secret store    │
│  audit log (append-only, hash-chained) · platform adapters (linux/mac/   │
│  win/android) · rate limiter · redaction · network egress policy         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Dependency rules (enforced by a test in Phase 1)
- `core/` may import `security/`, `tools/`(registry interface only), `llm/`, `memory/`, `rag/`, `events/`.
- `tools/*` **must not** import `core/agent*`. Tools are dumb executors + permission metadata.
- `voice/`, `vision/` must not import `tools/`. Perception emits events; it does not act.
- `llm/` must not import `tools/`. The provider returns text/tool-calls; routing happens in `core/`.
- `interface/` may import anything; nothing may import `interface/`.
- No module may import `os.environ` secrets directly — only `security/secrets.py`.

---

## 3. Data flow

### 3.1 Turn lifecycle (the canonical loop)
```
1. TRIGGER     voice wake-word | UI submit | CLI line | API request | scheduler tick
2. PERCEIVE    PerceptionFrame{ text?, image?, audio?, gestures?, screen?, source, confidence }
3. NORMALIZE   NormalizedUtterance{ text, lang, confidence, modality, context_refs }
4. ROUTE       Router.decide() -> IntentDecision
                 - fast path: regex/keyword intent table (deterministic, <5ms)
                 - slow path: LLM structured-output classification (pydantic schema)
                 - always: safety pre-classification (prompt-injection / sensitive)
5. AUTHZ       PermissionEngine.check(tool, level, ctx) -> ALLOW | CONFIRM_REQUIRED | DENY
6. PLAN        if Decision.needs_planning: Planner -> Plan{ steps[] } (bounded, max 6)
7. ACT         for step: ToolRouter.execute(name, args) -> ToolResult
8. OBSERVE     ToolResult{ success, data, evidence, error } appended to scratchpad
9. REASON      Agent decides: continue loop | synthesize | ask clarifying question
10. NARRATE    Narrator.render(synthesis, provenance) -> text (+ optional TTS, optional UI animation)
11. PERSIST     MemoryWriter: session row, optional long-term facts, audit entries, telemetry
12. EMIT        ui.speak / ui.state / tool.activity events -> interfaces
```
Loop steps 7–9 repeat up to `MAX_AGENT_ITERATIONS` (default 6) with a wall-clock budget (default 45 s). Every iteration emits `tool.activity` so the UI can show what is happening.

### 3.2 Provenance model (answers must be auditable)
Every factual claim in a response carries provenance:
```python
class Provenance(BaseModel):
    kind: Literal["model_parametric", "retrieved_context", "web_live", "user_stated", "tool_observation", "local_memory"]
    source_id: str | None      # document id / url / memory key
    url: str | None
    retrieved_at: datetime | None
    confidence: float | None
```
UI renders provenance chips: **"I already knew this"** vs **"retrieved from web (3 sources)"**. This directly implements §10 of the brief.

### 3.3 Event topics
| Topic | Payload | Subscribers |
|---|---|---|
| `perception.utterance` | `NormalizedUtterance` | router, ui, audit |
| `perception.gesture` | `GestureEvent` | gesture tools, ui |
| `agent.state` | `AgentState` (IDLE/THINKING/ACTING/WAITING_USER/ERROR) | ui, tts |
| `tool.activity` | `ToolActivity{name, phase, summary}` | ui, audit, log |
| `memory.written` | `MemoryRecordRef` | ui (memory indicator), audit |
| `rag.retrieval` | `RetrievalReport` | ui (searching state), audit |
| `security.confirm` | `ConfirmationRequest` | ui modal, voice prompt |
| `system.status` | `SystemStatus` | ui status bar |
| `automation.fired` | `AutomationRun` | ui, log |

---

## 4. Technology stack

| Concern | Choice | Rationale / alternative |
|---|---|---|
| Language | Python ≥3.11 (3.12 target) | Ecosystem for CV/LLM/automation. Note: dev sandbox here is 3.14/aarch64 — see §19. |
| Validation | **Pydantic v2** | Structured tool args + LLM structured output; single schema source of truth |
| Async runtime | **asyncio** + `uvloop` (optional) | One event loop owns bus, sensors, agent, UI |
| HTTP layer | **FastAPI** + **Uvicorn** | REST + WebSocket for UI and remote clients |
| LLM abstraction | **httpx** + provider adapters | No vendor SDK lock-in; OpenAI-compatible endpoints cover most providers |
| Local LLM | **llama-cpp-python** (GGUF) → later Ollama/llama.cpp server | No PyTorch required for basic local inference |
| Cloud LLM | OpenAI-compatible HTTP; Anthropic Messages API adapter | Provider-agnostic `LLMProvider` interface |
| Embeddings | `fastembed` (ONNX) default → optional API embeddings | CPU-friendly, no GPU requirement, no torch |
| Vector store | **sqlite-vec** (default, embedded) → pgvector (scale) | Zero-ops default; pluggable `VectorStore` protocol |
| Relational store | **SQLite** (WAL, `aiosqlite`) → PostgreSQL later | Single-user assistant: file DB is correct default |
| Web search | Serper / Brave / Tavily adapter behind `SearchProvider` | Provider swap without touching the agent |
| Web fetch/extract | `httpx` + `trafilatura` | Readability-style main-content extraction |
| Browser automation | Playwright (chromium) | Real DOM control, not blind keystrokes |
| Desktop automation | Platform adapters: `pyautogui`/`pynput` (fallback), `dbus` (Linux), AppleScript (macOS), PowerShell+UIA (Windows) | Native API preferred; keystroke injection is last resort |
| Vision (hands) | **MediaPipe Hand Landmarker** → fallback OpenCV + contour heuristics | 21 landmarks, runs on CPU, real-time |
| Vision (screen) | `mss` capture + OCR (`tesseract`/`paddleocr`) | Error-text extraction from screenshots |
| Wake word | `openWakeWord` (ONNX) → `pvporcupine` optional → energy VAD fallback | Offline, no cloud wake |
| STT | `faster-whisper` local default → `faster-whisper` large-v3 or cloud STT optional | Offline-first per §25 |
| TTS | `piper` (local, fast, good quality) → `pyttsx3` fallback → cloud TTS optional | Low-latency speech |
| Frontend | **React 18 + TypeScript + Vite**, `three` for the AI core, plain CSS/WebGL for HUD | Requirement §27 |
| Desktop UI | PySide6 only if the web UI cannot serve the need (Phase 10 decision) | Avoid two UI stacks |
| Scheduling | `apscheduler` (async) | Cron + interval + date triggers |
| Tests | `pytest`, `pytest-asyncio`, `pytest-cov`, `respx` (HTTP mocks), `hypothesis` | Fast, hermetic, no network |
| Lint/format | `ruff` + `mypy --strict` | Type hints enforced (rule §3) |

Explicit non-dependencies for now: no LangChain/LangGraph (the agent loop is ~200 lines and must be ours), no PyTorch (ONNX runtime instead), no Redis/Celery (asyncio queues).

---

## 5. Folder structure

```
prithivi/
├── main.py                     # single entrypoint: `python main.py chat|voice|vision|serve|ui`
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── requirements.txt            # pinned runtime deps (generated per phase)
├── requirements-dev.txt        # test/lint deps
├── .env.example                # documented env template, no real values
├── .gitignore
├── README.md  ARCHITECTURE.md  SETUP.md  SECURITY.md  API.md  TOOLS.md
├── MEMORY.md  GESTURES.md  CONTRIBUTING.md
├── docs/
│   ├── DATA_MODEL.md  ROADMAP.md  TESTING.md  ADR/  diagrams/
├── prithivi/                      # importable package
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py           # pydantic-settings; all tunables in one place
│   │   ├── events.py           # typed event bus
│   │   ├── state.py            # AgentState + SessionState (immutable updates)
│   │   ├── router.py           # intent routing (fast path + LLM slow path)
│   │   ├── planner.py          # bounded multi-step planning
│   │   ├── agent.py            # ReAct loop, budgets, tool-call protocol
│   │   ├── narrator.py         # response rendering + provenance
│   │   └── session.py          # session lifecycle, transcript
│   ├── llm/
│   │   ├── provider.py         # LLMProvider protocol + ToolCall/Message types
│   │   ├── registry.py         # provider factory
│   │   ├── openai_compat.py    # OpenAI, Groq, Together, Ollama, LM Studio, vLLM
│   │   ├── anthropic.py        # Messages API adapter
│   │   ├── local_model.py      # llama.cpp GGUF runner
│   │   └── prompts.py          # system prompt, tool-use policy, injection defence
│   ├── tools/
│   │   ├── base.py             # ToolSpec, ToolResult, @tool decorator, ToolContext
│   │   ├── registry.py         # allowlist registry + discovery
│   │   ├── app_control.py  browser.py  filesystem.py  system.py
│   │   ├── search.py  web_fetch.py  youtube.py  social_media.py
│   │   ├── productivity.py  media.py  gesture_tools.py  vision_tools.py
│   │   └── memory_tools.py  rag_tools.py  automation_tools.py
│   ├── voice/
│   │   ├── audio_manager.py    # device enum, VAD, ring buffer, 16k mono
│   │   ├── wake_word.py        # openWakeWord + configurable phrase
│   │   ├── speech_to_text.py   # faster-whisper + cloud adapter
│   │   ├── text_to_speech.py   # piper + fallback
│   │   └── conversation.py     # barge-in, turn-taking, echo guard
│   ├── vision/
│   │   ├── camera.py           # capture thread, backpressure, frame budget
│   │   ├── hand_tracking.py    # MediaPipe landmarks
│   │   ├── gesture_classifier.py  # static poses + dynamic motion features
│   │   ├── gesture_fsm.py      # debounce, cooldown, dwell, confidence state machine
│   │   ├── face_detection.py   # presence/attention gate (deprecated flag)
│   │   ├── screen_vision.py    # screenshot + OCR + region of interest
│   │   └── cursor.py           # smoothing, acceleration, pixel-accurate mapping
│   ├── memory/
│   │   ├── short_term.py  session.py  long_term.py  vector_store.py
│   │   ├── user_profile.py  summarizer.py  consolidation.py
│   ├── rag/
│   │   ├── ingestion.py  chunking.py  embeddings.py  retrieval.py
│   │   ├── knowledge_base.py  citations.py  web_research.py
│   ├── automation/
│   │   ├── scheduler.py  triggers.py  workflows.py  conditions.py
│   ├── security/
│   │   ├── permissions.py      # L0-L4 engine
│   │   ├── confirmation.py     # broker + timeout policy
│   │   ├── authentication.py   # local user profile, trusted devices, API tokens
│   │   ├── audit.py            # append-only hash-chained log
│   │   ├── secrets.py          # .env loader, redaction, never-log
│   │   ├── injection.py        # prompt-injection sanitiser for web/OCR content
│   │   └── egress.py           # network allowlist for outbound calls
│   ├── platform/               # OS abstraction (BREAKS the 'no platform libs in tools' rule on purpose)
│   │   ├── base.py  linux.py  macos.py  windows.py  detect.py
│   ├── interface/
│   │   ├── cli.py  hud.py     # terminal HUD renderer
│   │   ├── web/                # FastAPI: rest.py, ws.py, static/
│   │   └── ui/                 # React+TS source (built into interface/web/static)
│   ├── api/
│   │   ├── server.py  schemas.py  deps.py
│   ├── db/
│   │   ├── schema.sql  migrations/  repositories/
│   ├── observability/
│   │   ├── logging.py  metrics.py  tracing.py
│   └── tests/                  # mirrors package layout
├── data/                       # gitignored: prithivi.db, vectors/, audio/, screenshots/
└── logs/
```

> Platform-specific libraries are confined to `prithivi/platform/`. `tools/*` call the `PlatformAdapter` interface. This is the one deliberate exception to the import rules, and it is what keeps `tools/` portable.

---

## 6. Database design

SQLite in WAL mode, single file `data/prithivi.db`, accessed only through repositories. Schemas are additive-only migrations (`user_version` pragma). Full DDL in `docs/DATA_MODEL.md`.

Core tables:

| Table | Purpose | Key columns |
|---|---|---|
| `sessions` | conversation sessions | id, started_at, ended_at, modality, title |
| `messages` | transcript | id, session_id, role, content, modality, tokens, created_at, parent_id (branching), redacted |
| `tool_calls` | every tool invocation (audit trail core) | id, message_id, tool, args_json, result_json, status, latency_ms, permission_level, confirmed_by |
| `permissions` | user policy overrides | scope_pattern, level, effect (allow/confirm/deny), expires_at |
| `confirmations` | sensitive-action approvals | tool_call_id, decided, method (voice/ui/gesture+voice), decided_at, expires_at |
| `memories` | long-term structured facts | id, type, key, value_json, source, confidence, pinned, created_at, updated_at, embedding_id |
| `memory_fts` | FTS5 mirror of memories for exact/keyword lookup | content, key |
| `conversations_summary` | rolling summaries for long sessions | session_id, upto_message_id, summary, embedding_id |
| `documents` | RAG sources | id, uri, title, source_type, sha256, ingested_at, meta_json |
| `chunks` | RAG chunks | id, document_id, ordinal, text, token_count, meta_json, embedding_id |
| `embeddings` | vectors (sqlite-vec virtual table) | embedding_id, chunk_id/memory_id, vec(N), model, dims |
| `web_knowledge` | "learning" store | id, url, domain, title, fetched_at, topic, summary, trust_score, content_hash |
| `web_chunks` | chunks of fetched pages | id, web_knowledge_id, ordinal, text, embedding_id |
| `automations` | user workflows | id, name, cron/interval/trigger_json, prompt, enabled, next_run_at, last_run_at |
| `automation_runs` | execution log | id, automation_id, started_at, finished_at, status, output, error |
| `devices` | trusted devices / auth | id, name, public_key, trusted, created_at, last_seen_at |
| `api_tokens` | hashed tokens for API access | id, label, token_hash, scopes, created_at, last_used_at, revoked_at |
| `audit_log` | append-only, hash-chained | id, ts, actor, action, target, outcome, detail_json, prev_hash, hash |
| `settings_kv` | runtime feature flags | key, value_json, updated_at |
| `notifications` | proactive alerts | id, level, title, body, created_at, read_at, dedupe_key |

Indexes: FTS5 on `memories` and `chunks`; btree on `messages(session_id, created_at)`, `tool_calls(created_at)`, `automations(next_run_at)`, `audit_log(ts)`.
Retention: `tool_calls` and `audit_log` are pruned to a configurable window (default 90 days) — audit is append-only within its retention, never edited.
Encryption: sensitive columns (tokens, OAuth refresh tokens) are encrypted at rest with `cryptography`'s AES-GCM, key from `PRITHIVI_MASTER_KEY` in the environment or an OS keyring. Embeddings stay unencrypted (not secret).

---

## 7. Memory architecture

Four layers, distinct lifecycles and write rules:

| Layer | Store | Lifetime | Written by | Read by |
|---|---|---|---|---|
| **Working** (short_term) | In-process ring buffer (deque, token-budgeted) | One turn | Agent loop | Agent loop |
| **Session** | `messages` + rolling `conversations_summary` | One session | Every turn | Agent at turn start (summary + last N turns) |
| **Long-term** | `memories` (structured rows) | Until deleted | Explicit user consent, or high-confidence extraction rules | Router (profile), RAG (hybrid search) |
| **Knowledge** | `chunks` + `web_chunks` | Until source removed | Ingestion pipeline / web research | Retriever |

Write policy (the important part — this is what prevents PRITHIVI from hoarding):
- **Explicit**: "remember X" → structured row, `user_stated`, confidence 1.0.
- **Derived-preference**: patterns like "I prefer/use/always" → candidate row, confidence 0.6, surfaced in chat once as *"Noted — want me to remember that?"*. Only promoted on yes.
- **Never auto-stored**: secrets, credentials, one-off task details, third-party personal data, raw transcripts.
- Every stored item carries `source` and `embedding_id`; users can view/edit/delete each row and disable the layer entirely.

Recall policy: profile facts are injected into the system prompt (bounded to top-N by relevance); memories also go through hybrid retrieval (FTS5 + vector + recency + pin boost) with a hard cap of ~8 items per turn to protect context.

**Consolidation** (Phase 7): after each session, dedupe/merge similar memories, decay unpinned low-confidence items, and produce a session summary.

---

## 8. Agent architecture

**Loop**: bounded ReAct with explicit tool protocol.

```
agent.py
  async def run(utterance) -> AgentResult
    1. hydrate context  : summary + recent turns + profile + retrieved knowledge
    2. decide           : provider.complete(messages, tools=registry.schemas())
                          → tool_calls[] | final_text | clarification_request
    3. for each tool_call (parallel-safe group, max 3 concurrent):
           permission.check → confirm.broker (if required) → registry.execute
           append ToolObservation to scratchpad
    4. if observations and budget remains → goto 2 (max 6 iterations, 45 s)
    5. else narrator.render(final_text, provenance, observations)
```

Key design points:
- **Tool protocol**: OpenAI-style `tools=[{type:"function", function:{name, description, parameters}}]`, generated from each tool's Pydantic model via `model_json_schema()`. One schema definition serves validation, LLM prompting, and the UI form.
- **Structured output**: non-tool answers use a Pydantic `AnswerSchema` so the narrator and UI always have provenance and optional UI directives (e.g. show a card, open a URL).
- **No arbitrary code**: the registry rejects unknown tool names before any dispatch; arguments are validated by the tool's Pydantic model; there is no `eval`, `exec`, or shell passthrough tool anywhere. Shell access exists only as fixed, parameterised, allowlisted subcommands inside `system.py`.
- **Budgets & cancellation**: per-turn token budget, wall-clock timeout, and interrupt event (voice barge-in, UI stop button). Cancelled turns emit `agent.state=WAITING_USER` and log an audit entry.
- **Router**: fast deterministic path (exact/regex intent table, e.g. `open <app>`) avoids an LLM round-trip for common commands — faster and immune to prompt injection for those. Slow path = LLM classification with a Pydantic `IntentDecision`.
- **Planner**: only for multi-step goals (intent `needs_planning=True`, e.g. "research X and summarise into a note"). Produces `Plan{goal, steps[{tool, args, why}]}`, max 6 steps, each step independently permission-checked and confirmed if needed. No recursive sub-planning in v1.
- **Injection defence**: all web/OCR/document text is wrapped as untrusted data (`<untrusted>` tags) and the system prompt forbids obeying instructions found in retrieved content. `security/injection.py` strips/neutralises common override patterns and records hits.
- **Determinism where possible**: fixed temperature for routing, low temperature for narration, and a `seed` where supported.

---

## 9. Tool architecture

```python
class ToolSpec(BaseModel):
    name: str                       # snake_case, globally unique
    version: str = "1"
    description: str               # written for the LLM: when to use AND when not to
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    permission_level: PermissionLevel     # L0..L4
    side_effects: Literal["none","reversible","irreversible","external"]
    requires_confirmation: bool
    idempotent: bool
    timeout_s: float = 20.0
    executor: Callable[[ToolContext, BaseModel], Awaitable[ToolResult]]

class ToolResult(BaseModel):
    success: bool
    data: dict | None
    error: ToolError | None         # code, message, retryable, remedy
    evidence: Evidence | None       # how we know it succeeded (pid, http status, file hash…)
    duration_ms: int
```

Rules:
- Registration is explicit (`registry.register(spec)`); discovery is by metadata, never by filesystem scanning at runtime.
- Each tool's LLM-facing description states its negative cases ("do not use for X"), which measurably reduces mis-selection.
- Tool-level timeouts via `asyncio.wait_for`; a timeout is reported as an error, never as success.
- Every result is logged to `tool_calls` with redacted args.
- Capability reporting: if a tool's backend is unavailable (no ffmpeg, no browser installed), the tool returns `error.code=BACKEND_UNAVAILABLE` and the narrator says so plainly.
- Gesture-driven tools are ordinary tools with `permission_level` ≥1 and `requires_confirmation` where the brief demands it — gestures enter the same authorisation path as voice, never a bypass.

---

## 10. Security architecture

See `docs/SECURITY.md` for the full threat model. Normative summary:

**Permission levels** (per tool, user-overridable per scope):

| Level | Name | Examples | Default gate |
|---|---|---|---|
| L0 | Information | web_search, rag_query, read_notes, summarise | none |
| L1 | Low-risk action | open_app, open_url, media control, volume, screenshot | none (audit only) |
| L2 | User-data action | write/create/modify files in allowed roots, create notes, send *prepared* content to a user-specified target | confirm unless scope allowlisted |
| L3 | Sensitive | messages, social posts, purchases, account/OAuth changes, delete user files outside trash, automation creation | explicit confirmation, every time, unless a *narrow* trusted-automation rule matches exactly |
| L4 | Critical | security/system setting changes, credential storage, granting permissions, wiping memory, remote access enabling, LLM self-modification of policy | typed confirmation + explicit per-action consent token |

**Enforcement chain** (no component may skip a step):
```
trigger → IntentDecision → PermissionEngine.check(scope, level)
        → ConfirmationBroker.decide() [voice | UI | gesture+voice | none]
        → executor (re-checks scope at the OS boundary where possible)
        → AuditLog.append(hash-chained)
```

Additional controls:
- **Authentication**: local single-user profile; API access via hashed bearer tokens with scopes and expiry; optional OS-level auth hook for L4 actions.
- **Secret management**: `.env` (gitignored) loaded by `security/secrets.py`; keys never logged; a redaction filter scrubs anything matching key patterns before it reaches logs, LLM prompts, or transcripts.
- **Prompt-injection defence**: untrusted-content tagging, tool-arg validation, and — critically — the LLM's *proposals* are never authority. Authority comes from the permission engine only.
- **Egress policy**: outbound HTTP restricted to configured providers/search engines; optional "no external network" mode for fully offline operation.
- **Audit**: append-only, hash-chained (`hash = sha256(prev_hash + canonical_record)`), verified by `prithivi/audit verify`.
- **Trusted devices**: device-bound public keys; untrusted surfaces (e.g. remote API) cannot trigger L3/L4 without re-confirmation.
- **Data protection**: memory/RAG deletion is real deletion (plus vector-row removal); OAuth refresh tokens encrypted at rest; transcript redaction available per session.

---

## 11. Voice architecture

```
mic ──► AudioManager(16 kHz mono ring buffer, 200 ms frames)
          │
          ├─► VAD (energy + webrtcvad) ──► speech/silence boundaries
          │
          ├─► WakeWord (openWakeWord ONNX, configurable phrase, run on speech-ish frames only)
          │      └─ hit ──► reset timer, play chime (or silent if configured), arm STT
          │
          ├─► STT (faster-whisper local; cloud optional) ──► text + confidence + language
          │      └─ low confidence or empty ──► "I didn't catch that" (no tool calls)
          │
          ├─► ConversationManager: barge-in (TTS ducking/cancel), echo guard, turn timeout
          │
          └─► TextResponse ──► TTS (piper) ──► speaker, with chunked streaming so the first
                 sentence starts speaking while the rest is still generating
```
- **Language**: `auto | en | hi | hinglish`. Hinglish is handled by STT with `language="hi"` + an LLM normaliser that maps romanised Hindi to Devanagari-aware intent text before routing (do not route on raw transliteration).
- **Wake word**: default `"hey prithivi"`, user-changeable, with a documented enrolment path.
- **Config**: device index, sample rate, VAD aggressiveness, STT model size, TTS voice/speed, barge-in on/off.
- **Failure transparency**: mic permission denied, no input device, or STT model missing produce explicit spoken+logged errors.

---

## 12. Vision architecture

```
Camera (thread, 640x480@30, frame budget 33 ms) ──► latest-frame slot (drops old frames)
   └─► HandTracker (MediaPipe, 21 landmarks × up to 2 hands, ~5 ms)
         └─► GestureFeatureExtractor
               ├─ static: finger-extension bitmask → OPEN_PALM / INDEX_ONLY / PINCH / VICTORY / THUMB_UP / THUMB_DOWN / FIST
               └─ dynamic: palm-centre velocity, angle, spread → SWIPE_L/R, SPREAD, ROTATION, MOVE
               └─► GestureFSM  (enter-hold 250 ms, exit-hold 250 ms, confidence ≥0.75,
                                cooldown 800 ms, one-shot latching for discrete gestures,
                                dwell-to-repeat for continuous ones)
                     └─► GestureEvent (published on the bus) → gesture tools / UI cursor
```

Safety interlocks (non-negotiable, §8 of the brief):
1. **Gesture never authorises.** Gesture input can only select an intent or drive a cursor. Any L2+ action reached via gesture requires a second factor: voice confirmation or an on-screen confirm.
2. **Idle disarm** after N seconds of no voice/UI activity.
3. **Single-hand requirement for destructive gestures** + open-palm "stop" always wins (priority interrupt).
4. **Confidence + dwell + cooldown** on every discrete gesture; one discrete action per hold.
5. **Attention gate**: face presence required (opt-in, deprecated hook) before cursor control engages.
6. Vision runs in a separate process/thread with its own CPU budget (`VISION_MAX_FPS`, default 20) and never blocks the agent loop; frames are dropped, not queued.
7. **Screen vision** is a pull tool (`screenshot`, `read_screen_text`) — PRITHIVI does not continuously upload screen content.

---

## 13. RAG architecture

```
ingestion:  file/URI → parser (pypdf, markdown/text, python-docx, code splitter)
            → clean → chunk (structure-aware, 512 tok, 64 overlap, keep headings/page refs)
            → embed (fastembed ONNX) → sqlite-vec row + chunks row + metadata
retrieval:  query → hybrid = BM25(FTS5) ⊕ vector(cosine) ⊕ recency
            → RRF merge (k=60) → optional rerank (cross-encoder, optional)
            → top-k with per-document diversity cap → provenance → context block
answer:     context + query → LLM → AnswerSchema{answer, claims[], citations[]}
```
- Chunk metadata always carries `document_id, uri, title, page/section, ordinal, hash` so citations resolve to `uri#Lchunk`.
- Ingestion is explicit and local. Remote URLs are fetched with the egress policy applied and stored with `trust_score`.
- Deduplication by `sha256`; re-ingesting a changed document replaces prior chunks atomically.
- "Learning" = writing to `web_chunks`/`web_knowledge` only. **No model weight updates, ever.** New knowledge is retrievable context with a source, not a parameter change — this is what makes it auditable.

---

## 14. Web-research architecture

```
question
  → QueryPlanner: decompose into 1..4 sub-queries (LLM, structured)
  → SearchProvider (serper|brave|tavily|searxng) → ranked urls + snippets        [web_live]
  → Fetcher (parallel, per-domain politeness, robots-respecting, egress-checked)
      → trafilatura main-content extraction → clean text
      → security/injection.py sanitise + trust scoring (domain, date, https)
  → Extractor: claim extraction per source, with the sentence span that supports it
  → Cross-check: cluster claims by semantic similarity; flag contradictions,
      mark single-source claims as "unverified"
  → Synthesiser: answer citing [1][2]… with agreement/conflict labels
  → Store: optionally persist top sources as web_knowledge (user-toggled "remember this research")
  → Output: answer + source list (title, url, date, trust) + provenance=retrieved/web_live
```
- Answers explicitly separate: parametric knowledge / retrieved web / user-stated / local memory.
- Contradictions are surfaced, not silently averaged.
- Hard cap: 8 fetches, 20 s budget, all source claims traceable to a URL.

---

## 15. UI architecture

Three surfaces over one WebSocket event protocol (`docs/API.md` will define it fully):

1. **Terminal HUD (always available)** — state, active tool, last result, provenance, errors. Works with zero GUI deps.
2. **Web HUD (default GUI)** — React 18 + TypeScript + Vite. Three.js for the AI core, WebGL particle field, glassmorphism panels, CSS for HUD chrome.
3. **Desktop shell (optional, Phase 10 decision)** — PySide6 wrapper around the same web assets if a native window/tray/global-hotkey is required. Not built unless needed.

**AI core states** → visuals → voice behaviour:
| State | Visual | Voice |
|---|---|---|
| IDLE | slow breathing core, dim particles | silent |
| LISTENING | radial waveform reacting to amplitude | mic open |
| THINKING | counter-rotating rings, pulsing core | short "thinking" tick (optional) |
| SEARCHING | network of nodes lighting up along query path | silent |
| EXECUTING | orbital activity indicators + active tool label | silent (or "Opening Chrome.") |
| SPEAKING | waveform synchronised to TTS amplitude | TTS active |
| WAITING | pulsing confirm dialog with level + consequence text | asks for confirmation aloud |
| ERROR | red core, error card with cause + remedy | reads the error plainly |

Design rules: dark base (#05070d), 1 accent hue, `prefers-reduced-motion` respected, all effects GPU-composited, target 60 fps with vision running, WCAG AA contrast for text, never animate text the user must read.

---

## 16. Environment variables

Full list in `.env.example`. Phase 1 requires only:

| Var | Default | Purpose |
|---|---|---|
| `PRITHIVI_ENV` | `dev` | `dev`/`prod` (prod refuses debug logging) |
| `PRITHIVI_LLM_PROVIDER` | `openai_compatible` | `openai_compatible`/`anthropic`/`local` |
| `PRITHIVI_LLM_BASE_URL` | — | e.g. `http://localhost:11434/v1` |
| `PRITHIVI_LLM_API_KEY` | — | secret; never logged |
| `PRITHIVI_LLM_MODEL` | — | model id |
| `PRITHIVI_PERMISSION_DEFAULT_LEVEL` | `2` | baseline gating |
| `PRITHIVI_DATA_DIR` | `./data` | db, vectors, audio, screenshots |
| `PRITHIVI_MASTER_KEY` | — | for at-rest encryption (required in prod) |
| `PRITHIVI_LOG_LEVEL` | `INFO` | `DEBUG` enables verbose tool tracing |
| `PRITHIVI_EGRESS_ALLOW_NETWORK` | `true` | `false` = fully offline |

Later phases add `PRITHIVI_STT_*`, `PRITHIVI_TTS_*`, `PRITHIVI_WAKE_WORD`, `PRITHIVI_SEARCH_PROVIDER`, `PRITHIVI_EMBEDDING_MODEL`, `PRITHIVI_VECTOR_BACKEND`, `PRITHIVI_VISION_*`, `PRITHIVI_OAUTH_*`.

---

## 17. Testing strategy

Full plan in `docs/TESTING.md`. Principles:
- **Hermetic by default.** No test touches the network, real mic, real camera, or a real API key. HTTP is mocked (`respx`), audio/camera use fixture files and fake devices.
- **Layered.** Unit (pure logic: permissions, FSM, chunking, routing, redaction) → integration (agent↔tools, memory↔DB, RAG end-to-end on a tiny corpus) → contract (each tool's schema/output, each LLM provider adapter against recorded fixtures) → manual/soak (real hardware checklist).
- **Determinism.** Fake clock, seeded RNG, fixed fixture utterances; the agent loop is tested with a scripted provider that returns recorded tool-call sequences.
- **Property tests.** Permission engine (no bypass: a fuzzed tool call can never execute above the granted level) and gesture FSM (random gesture sequences never produce forbidden transitions).
- **Coverage gates.** ≥85% on `core/`, `security/`, `tools/base.py`, `memory/`; ≥70% overall.
- **CI.** ruff, mypy --strict, pytest -m "not hardware", all on every push.
- **Hardware tests** are opt-in via `-m hardware` and skipped in CI.

---

## 18. Implementation order (dependency-driven)

1. **Phase 0** — architecture + docs (this document set).
2. **Phase 1** — config, event bus, Pydantic models, LLM provider abstraction + one real provider, tool registry with base + 3 tools, router, agent loop, CLI, logging, tests. *Nothing else can be built without this.*
3. **Phase 2** — voice pipeline on top of the same agent turn path.
4. **Phase 3** — platform adapters, then app/browser/filesystem/system tools.
5. **Phase 6** (pulled forward) — web search/fetch/research: it is the highest-value capability and depends only on Phase 1's tool path.
6. **Phase 7** — DB + memory + embeddings + RAG (needed before automation and before knowledge-grounded chat).
7. **Phase 4 → 5** — vision: camera → landmarks → FSM → cursor/advanced gestures.
8. **Phase 8** — social/productivity integrations (OAuth).
9. **Phase 9** — scheduler, triggers, workflows.
10. **Phase 10** — web UI/HUD polish, packaging, hardening.

Rationale for the one deviation from the listed order: Phase 6 (web intelligence) depends on nothing but Phase 1, while Phase 4 (vision) depends on platform-specific native deps that are the slowest to get working. Building web research early proves the tool architecture end-to-end and gives real value sooner. Phases 4 and 5 remain in their relative order.

---

## 19. Environment note for this machine

This sandbox is **Termux/PRoot on Android aarch64, Python 3.14**. Consequences to plan around:
- MediaPipe and some OpenCV wheels have no aarch64-Android build → the vision stack will need a desktop host, or an alternative landmark model. The `HandLandmarker` interface is abstracted precisely for this: swapping implementations must not touch `gesture_fsm.py`.
- No audio input device (no mic, no speakers) → voice phases need a desktop target or synthetic-audio fixtures.
- No display server → the web UI runs headless and is verified via HTTP; the desktop shell is out of scope here.
- Python 3.14 is ahead of what most CV wheels target; Phase 1 should target 3.11–3.13 with 3.14 tolerated for pure-Python paths.

**Recommended development target: Linux desktop** (Ubuntu 22.04+/Fedora 39+) for full functionality; this machine is adequate for Phase 1 and Phases 6/7 (pure Python + ONNX).

---

## 20. Phase 0 exit criteria

- [x] System architecture, component diagram, data flow
- [x] Technology stack with rationale
- [x] Folder structure with import rules
- [x] Database design (`docs/DATA_MODEL.md`)
- [x] Memory architecture
- [x] Agent architecture
- [x] Tool architecture
- [x] Security architecture (`docs/SECURITY.md`)
- [x] Voice architecture
- [x] Vision architecture
- [x] RAG architecture
- [x] Web-research architecture
- [x] UI architecture
- [x] Roadmap + implementation order (`docs/ROADMAP.md`)
- [x] Dependencies (`requirements.txt`)
- [x] Environment variables (`.env.example`)
- [x] Testing strategy (`docs/TESTING.md`)

**No functional code has been written. Awaiting `START PHASE 1`.**