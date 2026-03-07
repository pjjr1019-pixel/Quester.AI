"""SQLite-backed storage and structured event logging."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config import APP_CONFIG, AppConfig
from utils import ensure_directory, utc_now_iso


class StorageManager:
    """Owns local persistence lifecycle and append-only event logs."""

    def __init__(self, config: AppConfig = APP_CONFIG, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("quester.storage")
        self._db_path: Path = config.storage.sqlite_path
        self._logs_dir: Path = config.storage.logs_dir
        self._events_path: Path = self._logs_dir / config.storage.events_log_name
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    async def start(self) -> None:
        """Open database and initialize tables."""
        if self._conn is not None:
            return
        ensure_directory(self._logs_dir)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        self.logger.info("StorageManager started (db=%s).", self._db_path)

    async def stop(self) -> None:
        """Close database connection cleanly."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        self.logger.info("StorageManager stopped.")

    async def log_event(self, stage: str, payload: dict[str, Any]) -> None:
        """Persist event to SQLite and JSONL."""
        conn = self._require_conn()
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        timestamp = utc_now_iso()
        with self._lock:
            conn.execute(
                "INSERT INTO runtime_events (timestamp, stage, payload_json) VALUES (?, ?, ?)",
                (timestamp, stage, payload_json),
            )
            conn.commit()
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"timestamp": timestamp, "stage": stage, "payload": payload},
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )

    async def set_kv(self, key: str, value: Any) -> None:
        """Set or replace a JSON-serializable value in kv_store."""
        conn = self._require_conn()
        value_json = json.dumps(value, sort_keys=True, default=str)
        with self._lock:
            conn.execute(
                "INSERT INTO kv_store (key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, value_json),
            )
            conn.commit()

    async def get_kv(self, key: str) -> Any | None:
        """Read and decode a value from kv_store."""
        conn = self._require_conn()
        with self._lock:
            cursor = conn.execute("SELECT value_json FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StorageManager must be started before use.")
        return self._conn

