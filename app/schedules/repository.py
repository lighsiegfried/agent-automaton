"""SQLite persistence for schedules (Phase 5C).

A separate local database (``storage/schedules/fifi_schedules.db``) in WAL mode with
schema versioning, owner scope, audit, and corruption recovery. The key primitive is
``acquire_lease`` — a single conditional UPDATE (atomic in SQLite) that lets exactly
one worker claim a schedule for a run, so a restart, a second poller, or a clock
change can never double-run the same occurrence.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.schedules.models import ACTIVE, Schedule, ScheduleRun

log = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    schedule_id      TEXT PRIMARY KEY,
    owner            TEXT NOT NULL,
    title            TEXT NOT NULL,
    timezone         TEXT NOT NULL,
    trigger_type     TEXT NOT NULL,
    next_run_at      TEXT,
    recurrence       TEXT,
    task_template    TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'active',
    language         TEXT NOT NULL DEFAULT 'es',
    last_run_at      TEXT,
    last_result      TEXT,
    run_count        INTEGER NOT NULL DEFAULT 0,
    max_runs         INTEGER,
    lease_owner      TEXT,
    lease_expires_at TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sched_owner_status ON schedules(owner, status);
CREATE INDEX IF NOT EXISTS idx_sched_due ON schedules(status, next_run_at);

CREATE TABLE IF NOT EXISTS schedule_runs (
    run_id        TEXT PRIMARY KEY,
    schedule_id   TEXT NOT NULL,
    scheduled_for TEXT,
    ran_at        TEXT NOT NULL,
    status        TEXT NOT NULL,
    task_id       TEXT,
    summary       TEXT,
    overdue       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_sched ON schedule_runs(schedule_id, ran_at);

CREATE TABLE IF NOT EXISTS schedule_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    schedule_id TEXT,
    event       TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_sched_audit ON schedule_audit(schedule_id, id);
"""

_COLS = (
    "schedule_id, owner, title, timezone, trigger_type, next_run_at, recurrence, "
    "task_template, status, language, last_run_at, last_result, run_count, max_runs, "
    "lease_owner, lease_expires_at, created_at, updated_at"
)
_RUN_COLS = "run_id, schedule_id, scheduled_for, ran_at, status, task_id, summary, overdue"


def _integrity_ok(conn) -> bool:
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
    log.warning("schedules: corrupt database backed up/reset (%s)", db_path)


def open_database(db_path: Path, *, recover: bool = True):
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
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        log.info("schedules: schema migrated to v%d", SCHEMA_VERSION)
    return conn


class ScheduleRepository:
    def __init__(self, db_path, *, now_fn=None, id_factory=None):
        self.db_path = Path(db_path)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: uuid.uuid4().hex)
        open_database(self.db_path).close()

    def _conn(self):
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

    def new_schedule_id(self) -> str:
        return "sch_" + self._id()

    def new_run_id(self) -> str:
        return "run_" + self._id()

    # -- create / save -------------------------------------------------------------

    def create(self, schedule: Schedule) -> Schedule:
        now = self.now_iso()
        schedule.created_at = schedule.created_at or now
        schedule.updated_at = now
        with self._cursor() as conn:
            conn.execute(
                f"INSERT INTO schedules ({_COLS}) VALUES "
                "(:schedule_id, :owner, :title, :timezone, :trigger_type, :next_run_at, "
                ":recurrence, :task_template, :status, :language, :last_run_at, :last_result, "
                ":run_count, :max_runs, :lease_owner, :lease_expires_at, :created_at, :updated_at)",
                schedule.to_row())
        return schedule

    def save(self, schedule: Schedule) -> Schedule:
        schedule.updated_at = self.now_iso()
        with self._cursor() as conn:
            conn.execute(
                "UPDATE schedules SET title=:title, timezone=:timezone, trigger_type=:trigger_type, "
                "next_run_at=:next_run_at, recurrence=:recurrence, task_template=:task_template, "
                "status=:status, language=:language, last_run_at=:last_run_at, "
                "last_result=:last_result, run_count=:run_count, max_runs=:max_runs, "
                "lease_owner=:lease_owner, lease_expires_at=:lease_expires_at, "
                "updated_at=:updated_at WHERE schedule_id=:schedule_id",
                schedule.to_row())
        return schedule

    # -- reads ---------------------------------------------------------------------

    def get(self, schedule_id: str) -> Schedule | None:
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_COLS} FROM schedules WHERE schedule_id=?", (schedule_id,)).fetchone()
        return Schedule.from_row(row) if row else None

    def list(self, owner: str, *, statuses=None, limit: int = 100) -> list[Schedule]:
        sql = f"SELECT {_COLS} FROM schedules WHERE owner=?"
        params: list = [owner]
        if statuses:
            sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
            params.extend(statuses)
        sql += " ORDER BY (next_run_at IS NULL), next_run_at LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Schedule.from_row(r) for r in rows]

    def due(self, now_iso: str, *, limit: int = 100) -> list[Schedule]:
        """Active schedules whose next_run_at is at or before ``now`` (any owner —
        the single owned scheduler serves the whole install)."""
        with self._cursor() as conn:
            rows = conn.execute(
                f"SELECT {_COLS} FROM schedules WHERE status=? AND next_run_at IS NOT NULL "
                "AND next_run_at <= ? ORDER BY next_run_at LIMIT ?",
                (ACTIVE, now_iso, limit)).fetchall()
        return [Schedule.from_row(r) for r in rows]

    def latest(self, owner: str, *, statuses=None, trigger_types=None) -> Schedule | None:
        cands = self.list(owner, statuses=statuses, limit=100)
        if trigger_types:
            cands = [c for c in cands if c.trigger_type in trigger_types]
        # Most recently updated first.
        cands.sort(key=lambda c: c.updated_at, reverse=True)
        return cands[0] if cands else None

    # -- atomic execution lease ----------------------------------------------------

    def acquire_lease(self, schedule_id: str, owner_token: str, now_iso: str,
                      lease_until_iso: str) -> bool:
        """Atomically claim a schedule for a run. Returns True iff this caller now
        holds the lease. A single conditional UPDATE — SQLite serializes writers, so
        two concurrent pollers can never both win."""
        with self._cursor() as conn:
            cur = conn.execute(
                "UPDATE schedules SET lease_owner=?, lease_expires_at=? "
                "WHERE schedule_id=? AND status=? "
                "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                (owner_token, lease_until_iso, schedule_id, ACTIVE, now_iso))
            return cur.rowcount == 1

    def release_lease(self, schedule_id: str) -> None:
        with self._cursor() as conn:
            conn.execute(
                "UPDATE schedules SET lease_owner=NULL, lease_expires_at=NULL WHERE schedule_id=?",
                (schedule_id,))

    # -- runs / audit --------------------------------------------------------------

    def record_run(self, run: ScheduleRun) -> ScheduleRun:
        with self._cursor() as conn:
            conn.execute(
                f"INSERT INTO schedule_runs ({_RUN_COLS}) VALUES "
                "(:run_id, :schedule_id, :scheduled_for, :ran_at, :status, :task_id, "
                ":summary, :overdue)", run.to_row())
        return run

    def runs(self, schedule_id: str, *, limit: int = 50) -> list[ScheduleRun]:
        with self._cursor() as conn:
            rows = conn.execute(
                f"SELECT {_RUN_COLS} FROM schedule_runs WHERE schedule_id=? "
                "ORDER BY ran_at DESC, run_id DESC LIMIT ?", (schedule_id, limit)).fetchall()
        return [ScheduleRun.from_row(r) for r in rows]

    def audit(self, event: str, *, schedule_id=None, detail=None) -> None:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO schedule_audit(ts, schedule_id, event, detail) VALUES (?,?,?,?)",
                (self.now_iso(), schedule_id, event, detail))

    def audit_recent(self, schedule_id: str | None = None, limit: int = 100) -> list[dict]:
        with self._cursor() as conn:
            if schedule_id:
                rows = conn.execute(
                    "SELECT ts, schedule_id, event, detail FROM schedule_audit "
                    "WHERE schedule_id=? ORDER BY id DESC LIMIT ?", (schedule_id, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts, schedule_id, event, detail FROM schedule_audit "
                    "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
