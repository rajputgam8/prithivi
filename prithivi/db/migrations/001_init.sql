-- 001_init.sql — Phase 7: sessions, memory, knowledge (RAG), vectors.
--
-- Conventions (docs/DATA_MODEL.md): timestamps are ISO-8601 UTC TEXT, JSON is
-- canonical TEXT, booleans are INTEGER 0/1, ids are generated app-side.
-- Every statement is idempotent so an interrupted migration can re-run.

-- ---------------------------------------------------------------------------
-- Session & transcript
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    modality      TEXT NOT NULL DEFAULT 'text',
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    summary_id    TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_id       TEXT REFERENCES messages(id) ON DELETE SET NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    modality        TEXT NOT NULL DEFAULT 'text',
    provenance_json TEXT NOT NULL DEFAULT '[]',
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    latency_ms      INTEGER,
    redacted        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS conversations_summary (
    session_id      TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    upto_message_id TEXT NOT NULL,
    summary         TEXT NOT NULL,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    embedding_id    TEXT,
    updated_at      TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Long-term memory
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    key          TEXT NOT NULL,
    value_json   TEXT NOT NULL,
    text         TEXT NOT NULL,
    source       TEXT NOT NULL,
    source_ref   TEXT,
    confidence   REAL NOT NULL DEFAULT 0.5,
    pinned       INTEGER NOT NULL DEFAULT 0,
    hits         INTEGER NOT NULL DEFAULT 0,
    embedding_id TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    last_used_at TEXT,
    deleted_at   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_key
    ON memories(key) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    key, text, content='memories', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, key, text) VALUES (new.rowid, new.key, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, key, text)
    VALUES ('delete', old.rowid, old.key, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, key, text)
    VALUES ('delete', old.rowid, old.key, old.text);
    INSERT INTO memory_fts(rowid, key, text) VALUES (new.rowid, new.key, new.text);
END;

-- ---------------------------------------------------------------------------
-- Knowledge / RAG
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    uri          TEXT NOT NULL,
    title        TEXT,
    source_type  TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    byte_size    INTEGER,
    page_count   INTEGER,
    meta_json    TEXT NOT NULL DEFAULT '{}',
    ingested_at  TEXT NOT NULL,
    indexed_at   TEXT,
    status       TEXT NOT NULL DEFAULT 'pending'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    locator      TEXT,
    meta_json    TEXT NOT NULL DEFAULT '{}',
    embedding_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunk_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- ---------------------------------------------------------------------------
-- Vectors
--
-- Stored as float32 BLOBs rather than a sqlite-vec virtual table: sqlite-vec
-- ships a native extension that cannot be built on aarch64-Android, and a
-- personal knowledge base (10^4-10^5 chunks) scores perfectly well in
-- process. The shape matches DATA_MODEL.md's vec0 table so a sqlite-vec or
-- pgvector backend can be dropped in behind memory.vector_store.VectorStore
-- without touching a single row above it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id TEXT PRIMARY KEY,
    owner_kind   TEXT NOT NULL,
    owner_id     TEXT NOT NULL,
    model        TEXT NOT NULL,
    dims         INTEGER NOT NULL,
    vec          BLOB NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_owner ON embeddings(owner_kind, owner_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- ---------------------------------------------------------------------------
-- Settings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings_kv (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
