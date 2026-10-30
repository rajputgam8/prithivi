"""Key/value settings persistence.

Small durable preferences that are not memories: the user's locale, which
search provider to prefer, whether proactive notifications are on.

Separated from :mod:`~prithivi.db.repositories.memories` on purpose. A memory is
something the user *told* Prithivi and can be inspected and deleted; a setting
is configuration Prithivi owns.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from ...db.connection import Database

__all__ = ["SettingsRepository"]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


class SettingsRepository:
    """CRUD over the ``settings_kv`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def set(self, key: str, value: Any) -> None:
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO settings_kv (key, value_json, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,"
                " updated_at=excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), _now()),
            )

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.query_one("SELECT value_json FROM settings_kv WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (json.JSONDecodeError, TypeError):
            return default

    def delete(self, key: str) -> bool:
        with self._db.tx() as conn:
            cur = conn.execute("DELETE FROM settings_kv WHERE key=?", (key,))
            return cur.rowcount > 0

    def all(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in self._db.query("SELECT key, value_json FROM settings_kv ORDER BY key"):
            try:
                out[str(row["key"])] = json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                continue
        return out
