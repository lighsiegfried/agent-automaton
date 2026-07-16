"""SQLite persistence for multi-step tasks (Phase 5B).

A separate local database (``storage/tasks/fifi_tasks.db``) in WAL mode with a
schema version in ``PRAGMA user_version``. State is committed after every
transition (crash-safe checkpoints), owner-scoped, and audited. A corrupt file is
moved aside and rebuilt rather than blocking startup — a paused task can then be
inspected after a restart, but nothing auto-resumes.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.tasks.models import AWAITING_CONFIRMATION, Step, Task

log = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    user_request  TEXT NOT NULL,
    title         TEXT NOT NULL,
    status        TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT 'es',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    current_step  INTEGER NOT NULL DEFAULT 0,
    max_steps     INTEGER NOT NULL DEFAULT 12,
    memory_refs   TEXT NOT NULL DEFAULT '[]',
    final_summary TEXT NOT NULL DEFAULT '',
    error_code    TEXT,
    approved_at   TEXT,
    started_at    TEXT,
    completed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_owner_status ON tasks(owner, status);

CREATE TABLE IF NOT EXISTS task_steps (
    step_id               TEXT PRIMARY KEY,
    task_id               TEXT NOT NULL,
    position              INTEGER NOT NULL,
    intent                TEXT NOT NULL,
    arguments             TEXT NOT NULL DEFAULT '{}',
    domain                TEXT NOT NULL DEFAULT '',
    risk_level            TEXT NOT NULL,
    dependencies          TEXT NOT NULL DEFAULT '[]',
    status                TEXT NOT NULL DEFAULT 'pending',
    attempt_count         INTEGER NOT NULL DEFAULT 0,
    idempotency_key       TEXT NOT NULL,
    result_summary        TEXT NOT NULL DEFAULT '',
    requires_confirmation INTEGER NOT NULL DEFAULT 0,
    confirmation_phrase   TEXT NOT NULL DEFAULT '',
    action_id             TEXT,
    error_code            TEXT,
    started_at            TEXT,
    completed_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_task ON task_steps(task_id, position);
CREATE UNIQUE INDEX IF NOT EXISTS idx_steps_idem ON task_steps(idempotency_key);

CREATE TABLE IF NOT EXISTS task_audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    task_id  TEXT,
    step_id  TEXT,
    event    TEXT NOT NULL,
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_audit ON task_audit(task_id, id);
"""

_STEP_COLS = (
    "step_id, task_id, position, intent, arguments, domain, risk_level, dependencies, "
    "status, attempt_count, idempotency_key, result_summary, requires_confirmation, "
    "confirmation_phrase, action_id, error_code, started_at, completed_at"
)
_TASK_COLS = (
    "task_id, owner, user_request, title, status, language, created_at, updated_at, "
    "current_step, max_steps, memory_refs, final_summary, error_code, approved_at, "
    "started_at, completed_at"
)


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row) and row[0] == "ok"


def _backup_corrupt(db_path: Path) -> None:
    if not db_path.exists():
        return
    i = 0
    while (cand := db_path.with_suffix(db_path.suffix + f".corrupt.{i}")).exists():
        i += 1
    try:
        db_path.replace(cand)
    except OSError:
        try:
            db_path.unlink()
        except OSError:
            return
    for side in ("-wal", "-shm"):
        s = db_path.with_name(db_path.name + side)
        if s.exists():
            try:
                s.unlink()
            except OSError:
                pass
    log.warning("tasks: corrupt database backed up/reset (%s)", db_path)


def open_database(db_path: Path, *, recover: bool = True) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if recover and db_path.exists() and not _integrity_ok(conn):
        conn.close()
        _backup_corrupt(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        log.info("tasks: schema migrated to v%d", SCHEMA_VERSION)
    return conn


class TaskRepository:
    def __init__(self, db_path, *, now_fn=None, id_factory=None):
        self.db_path = Path(db_path)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: uuid.uuid4().hex)
        open_database(self.db_path).close()

    # -- connection / ids ----------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def now_iso(self) -> str:
        return self._now().astimezone(timezone.utc).isoformat(timespec="seconds")

    def future_iso(self, seconds: float) -> str:
        return (self._now().astimezone(timezone.utc)
                + timedelta(seconds=seconds)).isoformat(timespec="seconds")

    def new_task_id(self) -> str:
        return "task_" + self._id()

    def new_step_id(self) -> str:
        return "step_" + self._id()

    # -- create / save -------------------------------------------------------------

    def create_task(self, task: Task) -> Task:
        now = self.now_iso()
        task.created_at = task.created_at or now
        task.updated_at = now
        with self._cursor() as conn:
            conn.execute(
                f"INSERT INTO tasks ({_TASK_COLS}) VALUES "
                "(:task_id, :owner, :user_request, :title, :status, :language, :created_at, "
                ":updated_at, :current_step, :max_steps, :memory_refs, :final_summary, "
                ":error_code, :approved_at, :started_at, :completed_at)",
                task.to_row(),
            )
            for step in task.steps:
                conn.execute(
                    f"INSERT INTO task_steps ({_STEP_COLS}) VALUES "
                    "(:step_id, :task_id, :position, :intent, :arguments, :domain, :risk_level, "
                    ":dependencies, :status, :attempt_count, :idempotency_key, :result_summary, "
                    ":requires_confirmation, :confirmation_phrase, :action_id, :error_code, "
                    ":started_at, :completed_at)",
                    step.to_row(),
                )
        return task

    def save_task(self, task: Task) -> Task:
        task.updated_at = self.now_iso()
        with self._cursor() as conn:
            conn.execute(
                "UPDATE tasks SET status=:status, current_step=:current_step, "
                "updated_at=:updated_at, memory_refs=:memory_refs, final_summary=:final_summary, "
                "error_code=:error_code, approved_at=:approved_at, started_at=:started_at, "
                "completed_at=:completed_at WHERE task_id=:task_id",
                task.to_row(),
            )
        return task

    def save_step(self, step: Step) -> Step:
        with self._cursor() as conn:
            conn.execute(
                "UPDATE task_steps SET status=:status, attempt_count=:attempt_count, "
                "result_summary=:result_summary, requires_confirmation=:requires_confirmation, "
                "confirmation_phrase=:confirmation_phrase, action_id=:action_id, "
                "error_code=:error_code, started_at=:started_at, completed_at=:completed_at "
                "WHERE step_id=:step_id",
                step.to_row(),
            )
        return step

    # -- reads ---------------------------------------------------------------------

    def get_task(self, task_id: str, *, with_steps: bool = True) -> Task | None:
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_TASK_COLS} FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                return None
            task = Task.from_row(row)
            if with_steps:
                step_rows = conn.execute(
                    f"SELECT {_STEP_COLS} FROM task_steps WHERE task_id=? ORDER BY position",
                    (task_id,)).fetchall()
                task.steps = [Step.from_row(r) for r in step_rows]
        return task

    def list_tasks(self, owner: str, *, statuses=None, limit: int = 50) -> list[Task]:
        sql = f"SELECT {_TASK_COLS} FROM tasks WHERE owner=?"
        params: list = [owner]
        if statuses:
            sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
            params.extend(statuses)
        sql += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Task.from_row(r) for r in rows]

    def latest_task(self, owner: str, *, statuses=None) -> Task | None:
        tasks = self.list_tasks(owner, statuses=statuses, limit=1)
        return self.get_task(tasks[0].task_id) if tasks else None

    def active_effect_task(self, owner: str, *, exclude: str | None = None) -> str | None:
        """The task_id currently holding the effect slot (awaiting a confirmation),
        used to enforce 'only one task executes effects at a time'."""
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT task_id FROM tasks WHERE owner=? AND status=?",
                (owner, AWAITING_CONFIRMATION)).fetchall()
        for r in rows:
            if r["task_id"] != exclude:
                return r["task_id"]
        return None

    def step_by_action_id(self, action_id: str) -> Step | None:
        if not action_id:
            return None
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_STEP_COLS} FROM task_steps WHERE action_id=? AND status=?",
                (action_id, AWAITING_CONFIRMATION)).fetchone()
        return Step.from_row(row) if row else None

    # -- audit ---------------------------------------------------------------------

    def audit(self, event: str, *, task_id=None, step_id=None, detail=None) -> None:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO task_audit(ts, task_id, step_id, event, detail) VALUES (?,?,?,?,?)",
                (self.now_iso(), task_id, step_id, event, detail))

    def audit_recent(self, task_id: str | None = None, limit: int = 100) -> list[dict]:
        with self._cursor() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT ts, task_id, step_id, event, detail FROM task_audit "
                    "WHERE task_id=? ORDER BY id DESC LIMIT ?", (task_id, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts, task_id, step_id, event, detail FROM task_audit "
                    "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
