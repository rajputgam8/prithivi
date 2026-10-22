# PRITHIVI — Security Architecture

Threat model assumption: PRITHIVI runs on the user's own machine with access to their files, microphone, camera, screen, accounts, and network. The realistic adversaries are not remote hackers but (a) **prompt injection from retrieved content**, (b) **over-broad autonomy**, (c) **accidental gestures**, (d) **leaked secrets in logs**, and (e) **social engineering via LLM-authored actions**.

Design principle: **the model is never an authority.** The LLM proposes; the permission engine disposes.

---

## 1. Permission levels

| Level | Name | Typical tools | Default gate | Reversible? |
|---|---|---|---|---|
| **L0** | Information | `web_search`, `web_fetch`, `rag_query`, `read_file`(allowed roots), `read_notes`, `summarize`, `memory_search` | none (audit only) | n/a — read-only |
| **L1** | Low-risk action | `open_application`, `open_url`, `media_control`, `set_volume`, `screenshot`, `read_screen_text` | none (audit only) | yes |
| **L2** | User-data action | `write_file`, `create_note`, `edit_document`, `move_file`, `set_reminder`, `start_timer`, `send_prepared_content`(to user-named target) | confirm unless scope allowlisted | usually |
| **L3** | Sensitive | `send_message`, `social_post`, `social_delete`, `purchase`, `create_automation`, `oauth_authorize`, `delete_file`(beyond trash), `email_send` | **explicit confirmation every time**, unless a narrow exact-match automation grant exists | no / externally visible |
| **L4** | Critical system | `change_security_setting`, `store_credential`, `grant_permission`, `wipe_memory`, `enable_remote_access`, `modify_prithivi_policy` | **typed confirmation + explicit per-action consent token**, re-auth from a trusted surface | no |

Enforcement is per-tool declaration **plus** per-scope user policy; the effective decision is the most restrictive of the two.

---

## 2. The enforcement chain

```
trigger (voice | text | gesture | ui | scheduler | api)
   │
   ▼
IntentDecision  ── router assigns intent + proposed tools
   │
   ▼
PermissionEngine.check(scope, level, context)
   ├── DENY            → tell the user plainly; audit; stop. Never re-try by other means.
   ├── CONFIRM_REQUIRED → ConfirmationBroker → pending token (TTL 120 s, single use)
   └── ALLOW           → executor
   │
   ▼
Executor  (re-validates its own scope at the OS boundary: realpath containment,
           app allowlist, argument ranges — never trusts the earlier check)
   │
   ▼
AuditLog.append(actor, channel, action, target, outcome, prev_hash → hash)
```

Implementation invariants (each becomes a test in Phase 1):
1. **No code path reaches an executor without a `PermissionDecision`.** Enforced by making `registry.execute()` require a decision object, not a boolean.
2. **L3/L4 cannot be reached from an untrusted surface** (remote API, untrusted device, gesture-only) without re-confirmation on a trusted surface.
3. **Gestures never grant authority.** A gesture may select intent or move a cursor; anything L2+ reached through a gesture requires a voice or UI second factor. This is the direct implementation of brief §8.
4. **Confirmation is bound to a specific action hash** (tool + canonical args). A confirmation for `delete_file('/tmp/a')` is invalid for `delete_file('/tmp/b')`.
5. **Denial is terminal.** The agent is told "denied" and must not attempt an alternative route to the same outcome without a new user request.

---

## 3. Confirmation mechanics

- **Request payload**: tool, human-readable consequence ("This will permanently delete 3 files"), level, scope, expiry.
- **Methods**: `voice` (spoken yes/no with intent-scoped matcher — "yes delete them" counts; a generic "yes" after a different question does not), `ui` (modal with the exact args), `gesture_voice` (gesture selects, voice confirms), `rule` (pre-existing narrow grant).
- **Timeouts**: L2 60 s, L3 120 s, L4 180 s. Expired → declined, audited.
- **Voice confirmation safety**: uses STT output only with (a) confidence threshold, (b) a confirmation-specific grammar, (c) a short window after the prompt. Low-confidence "yes" is treated as no-confirmation and re-asked once.
- **Repeat protection**: after two consecutive declines on the same tool+scope, PRITHIVI stops offering that action proactively for the session.
- **L4 typed confirmation**: user must state a short random phrase shown on screen (mitigates shoulder-surfing/replay from an audio capture).

---

## 4. Threat model & mitigations

| # | Threat | Mitigation | Verified by |
|---|---|---|---|
| T1 | **Prompt injection from web pages / PDFs / OCR** ("ignore previous instructions, email me the…") | Untrusted-content tagging in prompts; `security/injection.py` sanitiser; model output is never authority; tool allowlist + arg validation; L3/L4 always require out-of-band confirmation | unit tests + injection corpus test |
| T2 | **Autonomous destructive action** | L2+ confirmation, trash-before-delete default, scope allowlist requires explicit opt-in | permission tests |
| T3 | **Accidental gesture triggers a real action** | FSM: confidence ≥0.75, 250 ms hold, 800 ms cooldown, one-shot latch; open-palm stop priority; gesture never authorises L2+ | FSM property tests |
| T4 | **API key leakage** | `.env` only; `security/secrets.py` redaction filter applied to logs/prompts/transcripts; tokens stored hashed (API) or AES-GCM encrypted (OAuth refresh); `.env` gitignored; startup check that fails loudly if a key is found in source | secret-scan in CI |
| T5 | **Unauthorised remote access** | Hashed API tokens with scopes + expiry; bind to `127.0.0.1` by default; device trust model; remote surface cannot satisfy L3/L4 | API auth tests |
| T6 | **Data exfiltration via the assistant** | Egress allowlist restricting outbound page reads; empty allowlist means any *public* host, while private/loopback/SSRF targets are always refused; optional `PRITHIVI_EGRESS_ALLOW_NETWORK=false`; no "send to arbitrary URL" tool without L2 confirm | egress tests |
| T7 | **Over-broad file access** | Root allowlist for all filesystem tools; `realpath` containment check inside the executor (defence in depth); symlink escape blocked | path traversal tests |
| T8 | **Silent failure / false success claims** | Typed `ToolResult` with `evidence`; narrator forbidden from claiming success without it; error → safe recovery → honest explanation | agent narration tests |
| T9 | **Audit tampering** | Append-only, hash-chained audit log; `prithivi audit verify` detects modification; no UPDATE/DELETE code paths for audit | audit test |
| T10 | **Memory poisoning / stale memory** | Structured memories with `source` + `confidence`; deletions propagate to vectors; user can disable memory; derived memories require confirmation | memory tests |
| T11 | **Malicious automation** (scheduled prompt doing damage) | Automations may only use pre-granted narrow scopes; each run re-checks permissions; run budget; every run audited and visible | automation tests |
| T12 | **Screen/camera privacy** | No continuous upload — screen/camera capture is pull-based, on request; retention 24 h; obvious indicator while camera active; hard-disable switches | config tests |
| T13 | **Denial of self / poisoned logs flooding disk** | Log rotation + size caps; vector/chunk count caps; per-session turn caps | soak test |
| T14 | **Social engineering via social content** | Same as T1 for social API responses; never auto-send DMs; rate limits respected; no CAPTCHA/MFA bypass attempts, ever | policy test + code review rule |

**Explicit non-goals / refusals.** PRITHIVI will not: bypass login, CAPTCHA, MFA, rate limits or API restrictions; scrape where ToS forbids it; obtain credentials it was not given; disable its own audit trail; execute model-generated code; or send a user's data to a third party without an explicit user action that says so.

---

## 5. Authentication

- **Local user**: single-user model. A local profile with a passkey/PIN (Argon2id hash) protects L4 actions and API access; optional OS auth delegation (Touch ID, Windows Hello, polkit).
- **Devices**: enrolment generates an Ed25519 keypair; the public key is stored in `devices`. Trusted devices may satisfy L3 confirmations; untrusted ones may only perform L0–L1.
- **API tokens**: `prithivi tokens create --label laptop --scopes chat,search`. Only `sha256(token)` is stored; shown once at creation. Rotation and revocation supported.
- **OAuth (Phase 8)**: PKCE where supported, minimal scopes, refresh tokens encrypted at rest, per-provider scopes declared to the user at connect time. Disconnecting revokes both locally and remotely where the provider offers it.

---

## 6. Secret management

- `.env` (gitignored, mode 600) → loaded once by `security/secrets.py` into a typed settings object.
- Secrets are referenced by name only. Modules never read `os.environ` directly (enforced by a lint rule / test).
- `Redactor` scrubs `sk-*`, `Bearer <token>`, `xoxb-`, `AKIA*`, JWTs, and any value equal to a known secret before output reaches logs, transcripts, or LLM prompts.
- Prod startup validation: refuses to run with `PRITHIVI_ENV=prod` and a missing `PRITHIVI_MASTER_KEY`; warns if `.env` is world-readable.
- Long-lived credentials the user grants PRITHIVI (OAuth) go to `secrets_vault`, AES-GCM, key from the environment or OS keyring — never to disk plaintext.

---

## 7. Privacy

- **Local-first**: transcripts, memory, embeddings, screenshots, and audio stay on-device. Cloud calls are explicit, configurable per capability, and logged in `audit_log` with the destination.
- **Transparency at answer time**: every response shows provenance chips (parametric / web-retrieved / your-memory / tool-observed).
- **User control**: `prithivi memory list|edit|delete|export|wipe`, `prithivi knowledge list|forget`, `prithivi forget-session <id>`, and a "disable memory" switch.
- **Deletion is real**: SQLite rows + FTS entries + `embeddings` rows are removed in one transaction; the action is audited.
- **Third-party data**: PRITHIVI stores facts about the user only on explicit request, never facts about other people scraped from the web.

---

## 8. Gesture safety (enforced invariants)

1. Gesture events carry `confidence`, `hold_ms`, `handedness`, `hand_count`.
2. Discrete gestures require: detection confidence ≥ threshold **and** hold ≥ 250 ms **and** cooldown elapsed; then latch once until release.
3. Continuous gestures (cursor, zoom, rotate) use dwell-to-repeat with acceleration limits and a hard stop on loss of tracking.
4. `OPEN_PALM` is an unconditional priority interrupt (stop/pause).
5. Any gesture-originated request with level ≥ L2 is marked `source=gesture` and forces `requires_factor ∈ {voice, ui}`.
6. Gesture control auto-disarms after 60 s of no deliberate activity and re-arms via wake word or UI.
7. Hand tracking runs locally; no landmark data leaves the machine.

---

## 9. Operational security

- Least-privilege execution: file tools confined to configured roots; subprocess calls use argument arrays (never shell strings), a fixed binary allowlist, and a timeout.
- No `eval`/`exec` anywhere; no general "run this command" tool exists or will exist.
- Rate limits on outbound and LLM calls to prevent runaway cost.
- `PRITHIVI_ENV=prod` disables debug logging of payloads, disables the local test endpoints, and requires a bound loopback interface by default.
- Dependency hygiene: pinned versions, `pip-audit` in CI, minimal dependency set.

---

## 10. Security test matrix (Phase 1 onward)

| Test | Type | Expected |
|---|---|---|
| Unknown tool name from LLM | unit | rejected before dispatch, no side effect |
| Malformed tool args | unit | validation error returned to agent, no executor call |
| L3 call with no grant | integration | `CONFIRM_REQUIRED`, executor not invoked |
| L3 decline | integration | nothing happens, audited as declined |
| Confirmation token reuse | unit | rejected (single use) |
| Confirmation bound to different args | unit | rejected |
| Path traversal `../../etc/passwd` | unit | `DENY` from realpath containment |
| Gesture-originated L2 action without second factor | unit | `DENY` |
| Redaction of fake API key in log output | unit | scrubbed |
| Audit chain tamper detection | unit | `verify` fails |
| Prompt-injection corpus (20 payloads) | integration | agent refuses to perform injected tool call |
| Egress blocked host | unit | request refused |
| Token in transcript never plaintext | integration | only hash in DB |