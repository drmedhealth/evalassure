"""SQLite storage (one file, WAL mode). Every query that touches user data filters by user_id."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  password_hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  csrf TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS api_tokens (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  label TEXT NOT NULL, token_hash TEXT UNIQUE NOT NULL, prefix TEXT NOT NULL,
  created_at REAL NOT NULL, last_used_at REAL);
CREATE TABLE IF NOT EXISTS provider_keys (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL, label TEXT NOT NULL, base_url TEXT, ciphertext TEXT NOT NULL,
  masked TEXT NOT NULL, created_at REAL NOT NULL, last_used_at REAL);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT NOT NULL, path TEXT NOT NULL,
  rows INTEGER NOT NULL, columns TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, config TEXT NOT NULL,
  key_id TEXT, summary TEXT, error TEXT, log TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL, started_at REAL, finished_at REAL);
CREATE INDEX IF NOT EXISTS runs_user ON runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS datasets_user ON datasets(user_id, created_at DESC);
"""


class DB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.Lock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.c = c
        return c

    def q(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn().execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        r = self.conn().execute(sql, args).fetchone()
        return dict(r) if r else None

    def x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            c = self.conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.rowcount

    # ------------------------------------------------------------ helpers
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        row = {"id": self.new_id(), "created_at": time.time(), **row} if table != "sessions" else row
        cols = ", ".join(row)
        self.x(f"INSERT INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
        return row


def loads(s: str | None, default: Any = None) -> Any:
    try:
        return json.loads(s) if s else default
    except (TypeError, ValueError):
        return default
