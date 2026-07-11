"""Local memory: SQLite-backed log of handled commands."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    input_text TEXT NOT NULL,
    intent TEXT NOT NULL,
    tool TEXT,
    status TEXT NOT NULL,
    result_json TEXT
);
"""


def _connect() -> sqlite3.Connection:
    db_path: Path = get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    log.info("memory initialized at %s", get_settings().db_path)


def log_command(
    input_text: str,
    intent: str,
    tool: str | None,
    status: str,
    result: dict[str, Any] | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO command_log (input_text, intent, tool, status, result_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (input_text, intent, tool, status, json.dumps(result) if result else None),
        )


def recent_commands(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM command_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
