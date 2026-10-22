"""SQLite connection, transactions, and forward-only migrations.

One process, one file (``data/prithivi.db``), accessed only through the
repositories in :mod:`prithivi.db.repositories`. The engine is stdlib
:mod:`sqlite3` in WAL mode — no compiled extension, because the target device
is aarch64-Android where building one needs an NDK toolchain (ARCHITECTURE S19).

Design rules:

* **Explicit transactions only.** ``isolation_level=None`` puts SQLite in
  autocommit mode, so a stray ``execute`` can never half-open a transaction;
  :meth:`Database.tx` is the only place a transaction exists, and it issues
  ``BEGIN IMMEDIATE`` and commits or rolls back as a unit. Ingestion is
  transactional — a document with rows but no vectors must be impossible.
* **Thread-safe by construction.** The connection is shared with
  ``check_same_thread=False`` and every statement runs under one re-entrant
  lock, because the CLI, a future scheduler, and the API server all write.
* **Forward-only, idempotent migrations.** ``PRAGMA user_version`` holds the
  schema version and ``schema_migrations`` names what ran. Scripts use
  ``IF NOT EXISTS`` throughout, so a migration interrupted halfway re-runs
  cleanly instead of wedging the database.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["MIGRATIONS_DIR", "Database", "DatabaseError"]

log = logging.getLogger("prithivi.db")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class DatabaseError(RuntimeError):
    """A storage-level failure with a message safe to show a user."""


class Database:
    """A migrated, thread-safe SQLite connection.

    Args:
        path: file path, or ``":memory:"`` for tests.
        migrations_dir: override for the SQL scripts; tests point this at a
            fixture directory to exercise the migration runner itself.
    """

    def __init__(self, path: str | Path = ":memory:", *, migrations_dir: Path | None = None) -> None:
        self.path = str(path)
        self._migrations_dir = migrations_dir or MIGRATIONS_DIR
        self._lock = threading.RLock()
        self._depth = 0
        self._closed = False

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

    # -- raw access ----------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def closed(self) -> bool:
        return self._closed

    # -- statements ----------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Run one statement, returning the cursor.

        Outside :meth:`tx` this is autocommit; inside it, it joins the open
        transaction. Raises :class:`DatabaseError` once closed, so a background
        job cannot fail with an opaque ``ProgrammingError``.
        """
        with self._lock:
            self._check_open()
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction, rolling back on any exception.

        Re-entrant: a nested ``tx`` joins the outer one, so a repository method
        that already opens a transaction can be composed into a larger one.
        """
        with self._lock:
            self._check_open()
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
        try:
            yield self._conn
        except BaseException:
            with self._lock:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
            raise
        else:
            with self._lock:
                self._depth -= 1
                if outermost:
                    self._conn.execute("COMMIT")

    # -- migrations ----------------------------------------------------
    @property
    def user_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    @property
    def applied_migrations(self) -> tuple[str, ...]:
        if not self._has_migration_table():
            return ()
        rows = self.query("SELECT name FROM schema_migrations ORDER BY name")
        return tuple(str(r["name"]) for r in rows)

    def migrate(self) -> list[str]:
        """Apply every pending migration. Returns the names applied now.

        Each script runs on its own; the ledger row and ``user_version`` are
        written only after the script succeeds. A crash between those steps
        leaves an idempotent script to re-run, never a lost schema change.
        """
        files = sorted(self._migrations_dir.glob("*.sql"))
        applied: list[str] = []
        with self._lock:
            self._ensure_migration_table()
            done = set(self.applied_migrations)
            for path in files:
                if path.name in done:
                    continue
                script = path.read_text(encoding="utf-8")
                try:
                    self._conn.executescript(script)
                    version = int(path.name.split("_", 1)[0])
                    self._conn.execute(
                        "INSERT OR REPLACE INTO schema_migrations (name, version, applied_at)"
                        " VALUES (?,?,?)",
                        (path.name, version, dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")),
                    )
                    self._conn.execute(f"PRAGMA user_version = {version}")
                except sqlite3.Error as exc:
                    raise DatabaseError(f"migration {path.name} failed: {exc}") from exc
                applied.append(path.name)
                log.info("applied migration %s", path.name)
        return applied

    def _has_migration_table(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
        return row is not None

    def _ensure_migration_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                version    INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )

    def tables(self) -> tuple[str, ...]:
        """Names of user tables and views, excluding SQLite internals."""
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return tuple(str(r["name"]) for r in rows)

    # -- lifecycle -----------------------------------------------------
    def checkpoint(self) -> None:
        """Fold the WAL back into the main file (backup/copy safety)."""
        with self._lock:
            if not self._closed:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._depth:
                log.warning("closing the database with %d open transaction(s)", self._depth)
                self._depth = 0
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - only on a failed rollback
                    pass
            self._conn.close()
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise DatabaseError("database is closed")

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
