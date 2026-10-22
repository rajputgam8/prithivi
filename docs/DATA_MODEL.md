# PRITHIVI — Data Model (Phase 0)

Engine: **SQLite** (WAL) by default, PostgreSQL-compatible subset for later scale-out.
File: `data/prithivi.db` (`PRITHIVI_DATA_DIR`). Access only via repositories in `prithivi/db/repositories/`.
Migrations: forward-only, tracked with `PRAGMA user_version`. Each migration is an idempotent SQL script in `prithivi/db/migrations/NNN_name.sql`.

Conventions:
- Timestamps: `TEXT` ISO-8601 UTC (`2026-10-04T21:35:12.482Z`). Portable across SQLite/Postgres.
- JSON: `TEXT` containing canonical JSON (`json.dumps(..., sort_keys=True, separators=(',',':'))`) so hashes over records are reproducible.
- Booleans: `INTEGER` 0/1.
- IDs: `TEXT` ULID-like, sortable by creation time, generated app-side (no DB dependency for id generation).
- Deletion: hard delete for user-visible data (memory, documents); soft delete (`deleted_at`) only where referential history is required.

---

## DDL (abridged; full script is `prithivi/db/schema.sql`)

### Session & transcript
```sql
CREATE TABLE sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    modality      TEXT NOT NULL DEFAULT 'text',   -- text|voice|camera|automation|api
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    summary_id    TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_id     TEXT REFERENCES messages(id),          -- branch support for replan
    role          TEXT NOT NULL,                          -- user|assistant|system|tool
    content       TEXT NOT NULL,
    modality      TEXT NOT NULL DEFAULT 'text',
    provenance_json TEXT NOT NULL DEFAULT '[]',           -- Provenance[]
    tokens_in     INTEGER, tokens_out INTEGER, latency_ms INTEGER,
    redacted      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);

CREATE TABLE conversations_summary (      -- rolling summaries, one row per session (upsert)
    session_id     TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    upto_message_id TEXT NOT NULL,
    summary        TEXT NOT NULL,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    embedding_id   TEXT,
    updated_at     TEXT NOT NULL
);
```

### Tool execution & audit
```sql
CREATE TABLE tool_calls (
    id               TEXT PRIMARY KEY,
    session_id       TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    message_id       TEXT REFERENCES messages(id) ON DELETE SET NULL,
    trace_id         TEXT NOT NULL,                       -- one agent run
    turn_index       INTEGER NOT NULL DEFAULT 0,          -- loop iteration
    tool             TEXT NOT NULL,
    tool_version     TEXT NOT NULL DEFAULT '1',
    args_json        TEXT NOT NULL,                       -- redacted before write
    result_json      TEXT,                                -- redacted
    status           TEXT NOT NULL,                       -- ok|error|denied|timeout|confirmed_declined
    permission_level INTEGER NOT NULL,
    confirmed_by     TEXT,                                -- voice|ui|gesture_voice|rule|none
    source           TEXT NOT NULL DEFAULT 'agent',       -- agent|voice|gesture|automation|api|router
    latency_ms       INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX idx_tool_calls_created ON tool_calls(created_at);
CREATE INDEX idx_tool_calls_trace   ON tool_calls(trace_id);

CREATE TABLE audit_log (                 -- append-only, hash-chained
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    actor      TEXT NOT NULL,            -- user|agent|automation|api|system
    channel    TEXT NOT NULL,            -- cli|voice|ui|gesture|scheduler|api
    action     TEXT NOT NULL,            -- e.g. tool.execute, memory.delete, permission.change
    target     TEXT,
    outcome    TEXT NOT NULL,            -- success|denied|error|info
    detail_json TEXT NOT NULL DEFAULT '{}',
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
);
CREATE INDEX idx_audit_ts ON audit_log(ts);
```

### Security
```sql
CREATE TABLE permissions (
    id            TEXT PRIMARY KEY,
    scope_pattern TEXT NOT NULL,          -- 'tool:filesystem.write' | 'app:chrome' | 'web:*.com'
    level         INTEGER NOT NULL,       -- requested level
    effect        TEXT NOT NULL,          -- allow|confirm|deny
    max_uses      INTEGER,                -- NULL = unlimited (trusted automation)
    remaining_uses INTEGER,
    requires_factor TEXT,                 -- NULL|'voice'|'ui'|'gesture_voice'
    source        TEXT NOT NULL,          -- default|user|automation|trusted_device
    created_at    TEXT NOT NULL,
    expires_at    TEXT
);
CREATE INDEX idx_permissions_scope ON permissions(scope_pattern);

CREATE TABLE confirmations (
    id          TEXT PRIMARY KEY,
    tool_call_id TEXT REFERENCES tool_calls(id) ON DELETE SET NULL,
    tool        TEXT NOT NULL,
    level       INTEGER NOT NULL,
    consequence TEXT NOT NULL,             -- human-readable, shown before deciding
    decided     INTEGER,                   -- 1 approve, 0 decline, NULL pending/expired
    method      TEXT,                      -- voice|ui|gesture_voice|rule
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    expires_at  TEXT NOT NULL
);

CREATE TABLE devices (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,             -- local|remote|ui
    public_key  TEXT,
    trusted     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    last_seen_at TEXT
);

CREATE TABLE api_tokens (
    id         TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,       -- sha256, never the token itself
    scopes_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT
);

CREATE TABLE secrets_vault (               -- AES-GCM encrypted; key = PRITHIVI_MASTER_KEY
    name        TEXT PRIMARY KEY,
    ciphertext  TEXT NOT NULL,
    nonce       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    rotated_at  TEXT
);
```

### Memory
```sql
CREATE TABLE memories (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,            -- preference|fact|profile|skill|constraint|event_ref
    key          TEXT NOT NULL,
    value_json   TEXT NOT NULL,
    text         TEXT NOT NULL,            -- flat searchable projection
    source       TEXT NOT NULL,            -- user_stated|derived|consolidated|imported
    source_ref   TEXT,                     -- message id / document id / url
    confidence   REAL NOT NULL DEFAULT 0.5,
    pinned       INTEGER NOT NULL DEFAULT 0,
    hits         INTEGER NOT NULL DEFAULT 0,
    embedding_id TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    last_used_at TEXT,
    deleted_at   TEXT
);
CREATE UNIQUE INDEX idx_memories_key ON memories(key) WHERE deleted_at IS NULL;
CREATE INDEX idx_memories_type ON memories(type, confidence DESC);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    key, text, content='memories', content_rowid='rowid', tokenize='porter unicode61'
);
```

Memory kinds and example rows:
```json
{"type":"preference","key":"preferred_language","value":"Python","text":"User prefers Python","source":"user_stated","confidence":1.0,"pinned":true}
{"type":"constraint","key":"no_posting_without_confirm","value":true,"source":"user_stated","confidence":1.0}
{"type":"profile","key":"study_time","value":"19:00","source":"derived","confidence":0.6}
```

### Knowledge / RAG
```sql
CREATE TABLE documents (
    id           TEXT PRIMARY KEY,
    uri          TEXT NOT NULL,
    title        TEXT,
    source_type  TEXT NOT NULL,            -- pdf|markdown|text|code|docx|url|note
    sha256       TEXT NOT NULL,
    byte_size    INTEGER,
    page_count   INTEGER,
    meta_json    TEXT NOT NULL DEFAULT '{}',
    ingested_at  TEXT NOT NULL,
    indexed_at   TEXT,
    status       TEXT NOT NULL DEFAULT 'pending'  -- pending|indexing|ready|failed
);
CREATE UNIQUE INDEX idx_documents_hash ON documents(sha256);

CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    locator     TEXT,                      -- 'p.4' | 'L20-L34' | '#section'
    meta_json   TEXT NOT NULL DEFAULT '{}',
    embedding_id TEXT
);
CREATE INDEX idx_chunks_document ON chunks(document_id, ordinal);

CREATE TABLE web_knowledge (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    domain       TEXT NOT NULL,
    title        TEXT,
    topic        TEXT,
    summary      TEXT,
    fetched_at   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    trust_score  REAL NOT NULL DEFAULT 0.5,
    meta_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX idx_webk_url_hash ON web_knowledge(url, content_hash);

CREATE TABLE web_chunks (
    id              TEXT PRIMARY KEY,
    web_knowledge_id TEXT NOT NULL REFERENCES web_knowledge(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    claims_json     TEXT NOT NULL DEFAULT '[]',   -- [{claim, span, support:'single'|'corroborated'|'conflicting'}]
    embedding_id    TEXT
);
```

### Vectors
```sql
-- sqlite-vec (default). Dimension fixed by the embedding model in config.
CREATE VIRTUAL TABLE embeddings USING vec0(
    embedding_id TEXT PRIMARY KEY,
    owner_kind   TEXT NOT NULL,      -- chunk|memory|summary
    owner_id     TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vec          FLOAT[{dim}]
);
```
Postgres equivalent: `CREATE EXTENSION vector;` + `CREATE TABLE embeddings (embedding_id uuid PRIMARY KEY, owner_kind text, owner_id uuid, embedding vector(384));` with an IVFFlat/HNSW index. The `VectorStore` protocol hides the difference.

### Automation
```sql
CREATE TABLE automations (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    trigger_kind TEXT NOT NULL,        -- cron|interval|date|window_focus|device_connect|file_change|voice_command
    trigger_json TEXT NOT NULL,        -- {"expr":"0 8 * * *"} | {"event":"headphones_connected"}
    prompt       TEXT NOT NULL,        -- natural-language task for the agent
    enabled      INTEGER NOT NULL DEFAULT 1,
    permission_grants_json TEXT NOT NULL DEFAULT '[]', -- narrow, exact-match grants (auto-approved)
    max_runs     INTEGER,              -- NULL = unlimited
    last_run_at  TEXT,
    next_run_at  TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX idx_automations_next ON automations(enabled, next_run_at);

CREATE TABLE automation_runs (
    id           TEXT PRIMARY KEY,
    automation_id TEXT NOT NULL REFERENCES automations(id) ON DELETE CASCADE,
    trigger_kind TEXT NOT NULL,
    trigger_detail TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,        -- running|ok|error|skipped|declined
    output       TEXT,
    error        TEXT,
    session_id   TEXT
);
```

### Preferences / notifications / settings
```sql
CREATE TABLE notifications (
    id          TEXT PRIMARY KEY,
    level       TEXT NOT NULL,          -- low|normal|high
    channel     TEXT NOT NULL DEFAULT 'ui',  -- ui|voice|desktop
    title       TEXT NOT NULL,
    body        TEXT,
    dedupe_key  TEXT NOT NULL,
    source_ref  TEXT,
    created_at  TEXT NOT NULL,
    read_at     TEXT
);
CREATE UNIQUE INDEX idx_notif_dedupe ON notifications(dedupe_key, created_at);

CREATE TABLE settings_kv (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### Vector-store swap & scale notes
- **Default (single user, one machine)**: SQLite + WAL + sqlite-vec. Zero operations, file-backed, easy backup.
- **Scale-up trigger** (>200k chunks or concurrent writers): PostgreSQL + pgvector; repositories are the only change.
- **Backup**: `sqlite3 data/prithivi.db ".backup"` or file copy in WAL-safe mode; `prithivi export --include vectors` produces a portable bundle.

### Retention & privacy
| Data | Default retention | User control |
|---|---|---|
| `messages` | until deleted | per-session delete; redact content |
| `tool_calls` | 90 days | configurable, `--keep forever` |
| `audit_log` | 180 days | hash chain must be truncated only from the end |
| `web_knowledge` | until deleted | "forget this research" removes chunks + vectors |
| `audio` | not stored by default | opt-in debug capture, auto-purged |
| `screenshots` | 24 h | immediate delete after use unless saved by user |