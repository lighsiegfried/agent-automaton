"""Unified event model, allowlist redaction, and the read-optimized event store (5D).

The redaction is an ALLOWLIST: only known-safe, scalar metadata keys survive, and
every stored string is scanned for secrets. Content-bearing keys (bodies, drafts,
results, message text, form values, tokens, cookies) are never in the allowlist, so
they can't be stored even if a caller passes them. The store is a WAL SQLite database
with schema migrations, corruption recovery, owner scope, dedup (a UNIQUE key so a
double-fired hook can't duplicate an event), and incremental retention pruning.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.text.secrets import contains_likely_secret

log = get_logger(__name__)

# Natural-language credential form ("contraseña es hunter2") on top of the technical
# secret detector — a defensive backstop, since sources are already safe.
_CREDENTIAL_COPULA = re.compile(
    r"\b(contrase[nñ]a|password|passwd|clave|pin|otp|token|secret|api\s*key)\b"
    r"\s*(?:es|son|is|are|:|=)\s*\S+", re.IGNORECASE)

# --- domains / severities ----------------------------------------------------------
DOMAINS = ("command", "pending", "memory", "tasks", "schedules", "browser", "text",
           "whatsapp", "email", "wake", "voice", "runtime")
SEV_INFO = "info"
SEV_WARNING = "warning"
SEV_ERROR = "error"
SEVERITIES = (SEV_INFO, SEV_WARNING, SEV_ERROR)

# The ONLY metadata keys that may be stored. Everything else (body_preview, content,
# results, messages, text, subject, fields, values, tokens, cookies…) is dropped.
ALLOWED_METADATA_KEYS = frozenset({
    "count", "result_count", "target", "recipient", "recipients", "action_id",
    "step_count", "current_step", "position", "intent", "risk_level", "mode",
    "trigger_type", "recurrence", "run_count", "overdue", "next_run_at",
    "duration_ms", "field_count", "character_count", "heading_count", "language",
    "title", "task_status", "schedule_status", "provider", "direction",
})
_MAX_TEXT = 200


def redact_text(value) -> str:
    """Truncate and scrub a display string. Anything that trips the secret detector
    becomes ``[redacted]`` — defense in depth on top of already-safe sources."""
    if value is None:
        return ""
    text = str(value)
    is_secret, _ = contains_likely_secret(text)
    if is_secret or _CREDENTIAL_COPULA.search(text):
        return "[redacted]"
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT].rstrip() + "…"


def safe_metadata(raw: dict | None) -> dict:
    """Allowlist + secret-scan metadata. Only scalar values (and a short recipients
    list) survive; content-bearing keys and structures are dropped."""
    out: dict = {}
    for key, val in (raw or {}).items():
        if key not in ALLOWED_METADATA_KEYS:
            continue
        if key == "recipients" and isinstance(val, (list, tuple)):
            out[key] = [redact_text(v) for v in list(val)[:10]]
        elif isinstance(val, str):
            out[key] = redact_text(val)
        elif isinstance(val, (int, float, bool)) or val is None:
            out[key] = val
    return out


@dataclass
class Event:
    event_id: str
    ts: str
    domain: str
    event_type: str
    severity: str = SEV_INFO
    status: str = ""
    title: str = ""
    summary: str = ""
    related_id: str | None = None
    duration_ms: int | None = None
    error_code: str | None = None
    metadata: dict = field(default_factory=dict)
    dedup_key: str = ""
    owner: str = "local"

    def redacted(self) -> "Event":
        """Return a copy with title/summary/metadata scrubbed."""
        return Event(
            event_id=self.event_id, ts=self.ts, domain=self.domain, event_type=self.event_type,
            severity=self.severity if self.severity in SEVERITIES else SEV_INFO,
            status=redact_text(self.status), title=redact_text(self.title),
            summary=redact_text(self.summary), related_id=self.related_id,
            duration_ms=self.duration_ms, error_code=self.error_code,
            metadata=safe_metadata(self.metadata), dedup_key=self.dedup_key or self.event_id,
            owner=self.owner)

    def to_row(self) -> dict:
        return {"event_id": self.event_id, "ts": self.ts, "domain": self.domain,
                "event_type": self.event_type, "severity": self.severity, "status": self.status,
                "title": self.title, "summary": self.summary, "related_id": self.related_id,
                "duration_ms": self.duration_ms, "error_code": self.error_code,
                "metadata": json.dumps(self.metadata, ensure_ascii=False),
                "dedup_key": self.dedup_key or self.event_id, "owner": self.owner}

    @classmethod
    def from_row(cls, row) -> "Event":
        d = dict(row)
        return cls(event_id=d["event_id"], ts=d["ts"], domain=d["domain"],
                   event_type=d["event_type"], severity=d["severity"], status=d["status"] or "",
                   title=d["title"] or "", summary=d["summary"] or "", related_id=d["related_id"],
                   duration_ms=d["duration_ms"], error_code=d["error_code"],
                   metadata=json.loads(d["metadata"] or "{}"), dedup_key=d["dedup_key"],
                   owner=d["owner"])

    def public(self, *, with_metadata=True) -> dict:
        data = {"event_id": self.event_id, "ts": self.ts, "domain": self.domain,
                "event_type": self.event_type, "severity": self.severity, "status": self.status,
                "title": self.title, "summary": self.summary, "related_id": self.related_id,
                "duration_ms": self.duration_ms, "error_code": self.error_code}
        if with_metadata:
            data["metadata"] = self.metadata
        return data


# --- storage -----------------------------------------------------------------------

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    ts          TEXT NOT NULL,
    domain      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    status      TEXT,
    title       TEXT,
    summary     TEXT,
    related_id  TEXT,
    duration_ms INTEGER,
    error_code  TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    dedup_key   TEXT NOT NULL,
    owner       TEXT NOT NULL DEFAULT 'local'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup ON events(dedup_key);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(owner, ts);
CREATE INDEX IF NOT EXISTS idx_events_domain ON events(owner, domain, ts);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(owner, severity, ts);
CREATE INDEX IF NOT EXISTS idx_events_related ON events(related_id);
"""


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
    log.warning("activity: corrupt database backed up/reset (%s)", db_path)


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
        conn.execute("PRAGMA busy_timeout=2000")
    except sqlite3.DatabaseError:
        pass
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        log.info("activity: schema migrated to v%d", SCHEMA_VERSION)
    return conn


_COLS = ("event_id, ts, domain, event_type, severity, status, title, summary, "
         "related_id, duration_ms, error_code, metadata, dedup_key, owner")


class EventRepository:
    def __init__(self, db_path, *, now_fn=None):
        self.db_path = Path(db_path)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        open_database(self.db_path).close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=2000")
        except sqlite3.DatabaseError:
            pass
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

    def new_id(self) -> str:
        return "evt_" + uuid.uuid4().hex

    def insert(self, event: Event) -> bool:
        """Insert one redacted event. Returns True if stored, False if a duplicate
        dedup_key already existed (so a double-fired hook never duplicates)."""
        row = event.redacted().to_row()
        with self._cursor() as conn:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO events ({_COLS}) VALUES "
                "(:event_id, :ts, :domain, :event_type, :severity, :status, :title, :summary, "
                ":related_id, :duration_ms, :error_code, :metadata, :dedup_key, :owner)", row)
            return cur.rowcount == 1

    def get(self, event_id: str, owner: str) -> Event | None:
        with self._cursor() as conn:
            r = conn.execute(f"SELECT {_COLS} FROM events WHERE event_id=? AND owner=?",
                             (event_id, owner)).fetchone()
        return Event.from_row(r) if r else None

    def query(self, owner: str, *, domain=None, severity=None, status=None, since=None,
              until=None, related_id=None, limit=50, offset=0) -> tuple[list[Event], int]:
        where = ["owner=?"]
        params: list = [owner]
        for col, val in (("domain", domain), ("severity", severity), ("status", status),
                         ("related_id", related_id)):
            if val:
                where.append(f"{col}=?")
                params.append(val)
        if since:
            where.append("ts >= ?")
            params.append(since)
        if until:
            where.append("ts <= ?")
            params.append(until)
        clause = " AND ".join(where)
        with self._cursor() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM events WHERE {clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT {_COLS} FROM events WHERE {clause} ORDER BY ts DESC, id DESC "
                "LIMIT ? OFFSET ?", (*params, max(1, min(limit, 500)), max(0, offset))).fetchall()
        return [Event.from_row(r) for r in rows], total

    def severity_counts(self, owner: str, *, since=None) -> dict:
        clause = "owner=?"
        params: list = [owner]
        if since:
            clause += " AND ts >= ?"
            params.append(since)
        with self._cursor() as conn:
            rows = conn.execute(
                f"SELECT severity, COUNT(*) n FROM events WHERE {clause} GROUP BY severity",
                params).fetchall()
        return {r["severity"]: r["n"] for r in rows}

    def domain_counts(self, owner: str) -> dict:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT domain, COUNT(*) n FROM events WHERE owner=? GROUP BY domain",
                (owner,)).fetchall()
        return {r["domain"]: r["n"] for r in rows}

    def total(self, owner: str) -> int:
        with self._cursor() as conn:
            return conn.execute("SELECT COUNT(*) FROM events WHERE owner=?", (owner,)).fetchone()[0]

    def prune(self, *, retention_days: int, max_events: int, batch: int = 1000) -> int:
        """Incrementally drop events older than the window or beyond the cap. Bounded
        by ``batch`` so it never blocks the runtime with a huge delete."""
        removed = 0
        cutoff = (self._now().astimezone(timezone.utc)
                  - timedelta(days=retention_days)).isoformat(timespec="seconds")
        with self._cursor() as conn:
            cur = conn.execute(
                "DELETE FROM events WHERE id IN "
                "(SELECT id FROM events WHERE ts < ? ORDER BY id LIMIT ?)", (cutoff, batch))
            removed += cur.rowcount
            over = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] - max_events
            if over > 0:
                cur = conn.execute(
                    "DELETE FROM events WHERE id IN "
                    "(SELECT id FROM events ORDER BY id LIMIT ?)", (min(over, batch),))
                removed += cur.rowcount
        return removed
