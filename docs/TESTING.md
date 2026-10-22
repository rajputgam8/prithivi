# PRITHIVI — Testing Strategy

Applies from Phase 1 onward. Rules of the game:

1. **Hermetic by default.** No test may hit the network, a real microphone, a real camera, a real filesystem outside a tmp dir, or a real API key.
2. **No fakes pretending to be real.** Test doubles stand in for *infrastructure* (HTTP, devices), never for *behaviour under test* (permissions, FSM, routing, retrieval).
3. **Deterministic.** Fake clock, seeded RNG, recorded fixtures. A test that flakes is a bug.
4. **Honest coverage.** Coverage gates are enforced in CI; a module without tests cannot be merged as "done".

---

## 1. Test layers

| Layer | Scope | Examples | Runs in CI |
|---|---|---|---|
| **Unit** | Pure logic, no I/O | permission engine, gesture FSM, chunker, RRF merge, redaction, intent fast-path, arg validation | ✅ always |
| **Integration** | Real components wired together, still no external I/O | agent↔tool registry, memory↔SQLite, RAG end-to-end on a tiny corpus, confirmation flow, audit chain | ✅ always |
| **Contract** | External boundaries against recorded fixtures | each `LLMProvider` adapter, each `SearchProvider`, each `PlatformAdapter`, each tool's input/output schema | ✅ always |
| **E2E (simulated)** | Full turn loop with a scripted provider | "open chrome" → tool called → permission checked → narrated | ✅ always |
| **Hardware** | Real mic/camera/OS integration | wake word on live audio, gesture on live camera, real app launch, real volume | ❌ `-m hardware`, manual |
| **Live** | Real third-party APIs, opt-in, minimal cost | one real search, one real OAuth read, one real `pip`-free API call | ❌ `-m live`, `--run-live` |
| **Soak** | Long-running stability | 500-turn session, 2 h vision loop, log/DB growth caps | ❌ nightly |

---

## 2. Directory layout

```
prithivi/tests/
├── conftest.py                  # fixtures: settings, tmp data dir, fake clock, fake bus
├── fixtures/
│   ├── llm/                     # recorded provider responses (tool-call sequences)
│   ├── audio/*.wav              # short real recordings: silence, speech, noise, wake phrase
│   ├── video/*.mp4              # labelled gesture recordings
│   ├── corpus/*.pdf|md|txt      # tiny RAG corpus incl. one adversarial doc
│   └── web/*.html               # fetched-page fixtures incl. prompt-injection payloads
├── unit/
│   ├── test_permissions.py      # L0-L4 matrix, scope patterns, expiry, precedence
│   ├── test_confirmation.py     # binding, single-use, TTL, voice grammar
│   ├── test_audit.py            # hash chain, tamper detection
│   ├── test_secrets.py          # redaction, no-env-read rule, prod validation
│   ├── test_router.py           # fast path, LLM path, fallback, injection pre-check
│   ├── test_agent.py            # loop bounds, budgets, tool protocol, narration honesty
│   ├── test_registry.py         # duplicate names, schema generation, unknown tool rejection
│   ├── test_tools_*.py          # per tool: happy path, bad input, backend-unavailable
│   ├── test_gesture_fsm.py      # hold/cooldown/latch/confidence invariants
│   ├── test_gesture_features.py # static + dynamic classification on fixtures
│   ├── test_memory.py           # write policy, dedupe, deletion, disable
│   ├── test_rag.py              # chunking, hybrid retrieval, citation resolution
│   ├── test_injection.py        # sanitiser on injection corpus
│   └── test_automation.py       # triggers, dedupe, permission re-check
├── integration/
│   ├── test_turn_e2e.py         # scripted provider drives full turns
│   ├── test_memory_db.py        # repositories + migrations + vectors
│   ├── test_rag_end_to_end.py   # ingest → retrieve → cited answer
│   ├── test_web_research.py     # mocked search/fetch → cited, contradiction-aware answer
│   ├── test_permissions_flow.py # end-to-end gating incl. declines and gesture source
│   └── test_api_server.py       # REST + WebSocket contract
├── property/                    # hypothesis
│   ├── test_perm_nobypass.py    # invariant: no fuzzed call executes above granted level
│   └── test_fsm_safety.py       # invariant: random gesture streams never fire forbidden transitions
├── contract/
│   ├── test_providers.py        # all LLM adapters vs recorded fixtures
│   ├── test_platform_adapters.py
│   └── test_tool_schemas.py     # every registered tool has valid JSON Schema + docstring
└── hardware/                    # -m hardware
    ├── test_wake_word_live.py  test_stt_live.py  test_tts_live.py
    ├── test_hand_tracking_live.py  test_cursor_live.py
    └── test_app_control_live.py
```

---

## 3. Key techniques

**Scripted LLM provider.** A `FakeProvider` replays recorded `Completion` objects. Turn tests become deterministic sequences:
```python
PROMPT_CHROME = [
    Completion(tool_calls=[ToolCall("open_application", {"application": "chrome"})]),
    Completion(text="Opening Chrome."),
]
```
Assertions then cover the whole chain: tool chosen, args validated, permission checked, executor called with the right context, narration produced, audit row written.

**HTTP mocking with `respx`.** Search engines, web fetch, LLM endpoints, and OAuth token endpoints are mocked at the transport level, so adapters are exercised for real (headers, JSON shapes, error handling) without network.

**Fake devices.** `FakeAudioDevice` replays `.wav` fixtures; `FakeCamera` yields decoded video frames from fixtures; both expose realistic failure modes (device busy, permission denied, frame drop) so error paths get tested too.

**Fake clock.** `freezegun`-style injected clock for scheduler, TTLs, cooldowns, retention, and rate limits. No `sleep` in tests.

**Injected randomness.** `random.Random(seed)` passed to gesture noise simulation and any sampling; property tests use Hypothesis-generated inputs.

**In-memory SQLite.** `sqlite+aiosqlite:///:memory:` for unit-ish integration tests; a file DB in `tmp_path` for migration tests.

---

## 4. Phase-by-phase minimum test matrix

| Phase | Must-have tests |
|---|---|
| 1 | router fast/LLM paths, agent loop bounds + narration honesty, registry rejection, permission matrix, confirmation binding, audit chain, redaction, secrets prod-validation, tool schemas, E2E scripted turns, CLI smoke |
| 2 | VAD segmentation on fixtures, wake-word true/false accept-reject rates, STT confidence handling, TTS text chunking, barge-in state machine (hardware tests manual) |
| 3 | root confinement + traversal denial, realpath symlink escape, app resolution per platform (contract), browser navigation timeout/error, screenshot evidence, volume verified by re-read |
| 4 | landmark extraction on fixtures, static gesture accuracy, dynamic feature extraction, FSM invariants (property), frame-drop behaviour under load |
| 5 | cursor coordinate mapping accuracy, smoothing/acceleration curves, pinch-click position, swipe latch once, zoom/rotate value mapping, gesture+voice confirmation enforcement |
| 6 | search provider adapters (contract), fetch + extraction, citation resolution, single-source labelling, contradiction detection, injection corpus, egress policy refusal |
| 7 | migration up/down, repository CRUD, memory write policy + consent, dedupe/consolidation, deletion removes vectors, chunking boundaries, hybrid ranking, RRF, cited answer correctness, memory disabled mode |
| 8 | OAuth PKCE flow (mocked), scope minimisation, token encryption at rest, read-only actions, write requires confirmation, rate-limit handling, provider API error mapping |
| 9 | cron/interval/date trigger computation, device/window triggers (mocked), run dedupe, permission re-check per run, run budget, notification levels + dedupe keys, disabled automation no-op |
| 10 | WS event contract, all 8 states rendered, confirmation modal gating, reconnect/resume, reduced-motion, 60 fps budget check (headless), a11y contrast, package build/install smoke |

---

## 5. Coverage & quality gates

```toml
[tool.coverage.report]
fail_under = 70                    # overall
# per-package floors enforced in CI:
#   core/        85
#   security/    90   (highest risk, highest bar)
#   tools/base   90
#   memory/      80
```

CI pipeline (GitHub Actions):
1. `ruff check` + `ruff format --check`
2. `mypy --strict` (no untyped defs; brief rule §3)
3. `pytest -m "not hardware and not live" --cov=prithivi --cov-fail-under=70`
4. `pip-audit` on pinned dependencies
5. secret scan (gitleaks / regex) — no keys in source
6. Optional: build the web UI to ensure the frontend still compiles

---

## 6. Manual hardware checklist (printed in SETUP.md)

- [ ] Wake word fires on the real phrase; false-accept rate acceptable in a noisy room
- [ ] Latency budget: end-of-speech → agent state `THINKING` < 400 ms (local STT) or documented for cloud
- [ ] TTS starts speaking within 300 ms of first sentence available
- [ ] Camera opens, hands tracked at target FPS, gestures fire as designed, open-palm stop always interrupts
- [ ] Cursor mapping accurate at screen corners; no drift after 2 min
- [ ] Voice gesture L2 action refused without second factor (spoken confirmation)
- [ ] Permission prompts appear for L2/L3 with the correct consequence text
- [ ] Audit log verifies after a full session of mixed voice/gesture/UI actions
- [ ] Memory delete removes the vector row as well as the DB row
- [ ] Offline mode: all L0 features work, egress refuses, narration says why

---

## 7. Definition of Done (per phase)

- [ ] All tests for that phase pass locally and in CI, above coverage floors
- [ ] `mypy --strict` and `ruff` clean
- [ ] No secrets in source; `.env.example` updated
- [ ] Docs for the phase written (`SETUP.md` run instructions, `TOOLS.md` entries)
- [ ] Known limitations documented honestly (no "works great" claims without a test)
- [ ] User has been shown the files and the run instructions and has confirmed to proceed