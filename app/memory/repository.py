"""SQLite persistence for personal memory (Phase 5A).

Owns every SQL statement: typed-memory CRUD, revision history, the audit trail,
and keeping the FTS5 index in sync with the ``memories`` table. Timestamps and IDs
are injectable so the service is fully deterministic under test. A connection is
opened per operation (cheap at personal-memory scale and safe under FastAPI's
threadpool); the schema/recovery pass runs once at construction.

Soft deletion (status → ``forgotten``) removes a memory from the FTS index so it is
excluded from retrieval while remaining on disk; permanent deletion removes the row,
its revisions, and its FTS entry, leaving only a content-free audit record.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.memory import migrations
from app.memory.models import (
    ACTIVE,
    EV_PROPOSAL,
    FORGOTTEN,
    Memory,
    normalize,
)

log = get_logger(__name__)

_MEMORY_COLUMNS = (
    "id, owner, type, title, content, entities, tags, subject, source, "
    "created_at, updated_at, last_used_at, confidence, expires_at, status, "
    "content_hash, revision"
)


def _fts_blob(values: list[str]) -> str:
    return " ".join(values or [])


class MemoryRepository:
    def __init__(self, db_path, *, now_fn=None, id_factory=None):
        self.db_path = Path(db_path)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._id = id_factory or (lambda: "mem_" + uuid.uuid4().hex)
        # Ensure schema + recover a corrupt file exactly once, up front.
        migrations.open_database(self.db_path).close()

    # -- connection / clock --------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self):
        """A connection scoped to one operation: commit on success, and ALWAYS
        close so no file handle/WAL lock leaks (important on Windows)."""
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

    def new_id(self) -> str:
        return self._id()

    # -- FTS sync (private; caller holds the connection/transaction) ---------------

    def _fts_delete(self, conn: sqlite3.Connection, memory_id: str) -> None:
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))

    def _fts_insert(self, conn: sqlite3.Connection, memory: Memory) -> None:
        conn.execute(
            "INSERT INTO memories_fts(memory_id, title, content, entities, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory.id, memory.title, memory.content,
             _fts_blob(memory.entities), _fts_blob(memory.tags)),
        )

    def _write_revision(self, conn: sqlite3.Connection, memory: Memory, reason: str) -> None:
        conn.execute(
            "INSERT INTO memory_revisions(memory_id, revision, title, content, "
            "entities, tags, subject, confidence, content_hash, status, changed_at, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory.id, memory.revision, memory.title, memory.content,
             _fts_blob(memory.entities), _fts_blob(memory.tags), memory.subject,
             memory.confidence, memory.content_hash, memory.status,
             memory.updated_at, reason),
        )

    # -- writes --------------------------------------------------------------------

    def insert(self, memory: Memory, *, reason: str = "created") -> Memory:
        now = self.now_iso()
        memory.created_at = memory.created_at or now
        memory.updated_at = now
        memory.revision = 1
        memory.status = memory.status or ACTIVE
        row = memory.to_row()
        with self._cursor() as conn:
            conn.execute(
                f"INSERT INTO memories ({_MEMORY_COLUMNS}) VALUES "
                "(:id, :owner, :type, :title, :content, :entities, :tags, :subject, "
                ":source, :created_at, :updated_at, :last_used_at, :confidence, "
                ":expires_at, :status, :content_hash, :revision)",
                row,
            )
            if memory.status == ACTIVE:
                self._fts_insert(conn, memory)
            self._write_revision(conn, memory, reason)
        return memory

    def update(self, memory: Memory, *, reason: str = "updated") -> Memory:
        """Persist a new revision of an existing memory IN PLACE (stable id)."""
        memory.updated_at = self.now_iso()
        memory.revision = memory.revision + 1
        row = memory.to_row()
        with self._cursor() as conn:
            conn.execute(
                "UPDATE memories SET type=:type, title=:title, content=:content, "
                "entities=:entities, tags=:tags, subject=:subject, source=:source, "
                "updated_at=:updated_at, confidence=:confidence, expires_at=:expires_at, "
                "status=:status, content_hash=:content_hash, revision=:revision "
                "WHERE id=:id",
                row,
            )
            self._fts_delete(conn, memory.id)
            if memory.status == ACTIVE:
                self._fts_insert(conn, memory)
            self._write_revision(conn, memory, reason)
        return memory

    def soft_forget(self, memory_ids: list[str], *, reason: str = "forgotten") -> int:
        changed = 0
        with self._cursor() as conn:
            for mid in memory_ids:
                row = conn.execute(
                    f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id=? AND status=?",
                    (mid, ACTIVE),
                ).fetchone()
                if row is None:
                    continue
                memory = Memory.from_row(row)
                memory.status = FORGOTTEN
                memory.updated_at = self.now_iso()
                memory.revision += 1
                conn.execute(
                    "UPDATE memories SET status=?, updated_at=?, revision=? WHERE id=?",
                    (FORGOTTEN, memory.updated_at, memory.revision, mid),
                )
                self._fts_delete(conn, mid)            # excluded from retrieval
                self._write_revision(conn, memory, reason)
                changed += 1
        return changed

    def restore(self, memory_id: str) -> Memory | None:
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id=? AND status=?",
                (memory_id, FORGOTTEN),
            ).fetchone()
            if row is None:
                return None
            memory = Memory.from_row(row)
            memory.status = ACTIVE
            memory.updated_at = self.now_iso()
            memory.revision += 1
            conn.execute(
                "UPDATE memories SET status=?, updated_at=?, revision=? WHERE id=?",
                (ACTIVE, memory.updated_at, memory.revision, memory_id),
            )
            self._fts_delete(conn, memory_id)
            self._fts_insert(conn, memory)
            self._write_revision(conn, memory, "restored")
            return memory

    def permanent_delete(self, memory_ids: list[str]) -> int:
        """Remove rows, revisions, and FTS entries entirely (no content remains)."""
        removed = 0
        with self._cursor() as conn:
            for mid in memory_ids:
                cur = conn.execute("DELETE FROM memories WHERE id=?", (mid,))
                if cur.rowcount:
                    removed += 1
                conn.execute("DELETE FROM memory_revisions WHERE memory_id=?", (mid,))
                self._fts_delete(conn, mid)
        return removed

    def touch_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = self.now_iso()
        with self._cursor() as conn:
            conn.executemany(
                "UPDATE memories SET last_used_at=? WHERE id=?",
                [(now, mid) for mid in memory_ids],
            )

    # -- reads ---------------------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return Memory.from_row(row) if row else None

    def list(self, owner: str, *, statuses=(ACTIVE,), mem_type: str | None = None,
             limit: int = 100) -> list[Memory]:
        clause = ",".join("?" for _ in statuses)
        params: list = [owner, *statuses]
        sql = (f"SELECT {_MEMORY_COLUMNS} FROM memories "
               f"WHERE owner=? AND status IN ({clause})")
        if mem_type:
            sql += " AND type=?"
            params.append(mem_type)
        sql += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Memory.from_row(r) for r in rows]

    def find_duplicate(self, owner: str, mem_type: str, content_hash: str) -> Memory | None:
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories "
                "WHERE owner=? AND type=? AND content_hash=? AND status=? LIMIT 1",
                (owner, mem_type, content_hash, ACTIVE),
            ).fetchone()
        return Memory.from_row(row) if row else None

    def find_conflict(self, owner: str, mem_type: str, subject: str,
                      content_hash: str) -> Memory | None:
        """An active memory with the same subject but a DIFFERENT value — the one a
        new value would update. Returns None when there is no subject to key on."""
        subject = normalize(subject)
        if not subject:
            return None
        with self._cursor() as conn:
            row = conn.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories "
                "WHERE owner=? AND type=? AND subject=? AND status=? AND content_hash!=? "
                "ORDER BY datetime(updated_at) DESC LIMIT 1",
                (owner, mem_type, subject, ACTIVE, content_hash),
            ).fetchone()
        return Memory.from_row(row) if row else None

    def active(self, owner: str, *, limit: int = 500) -> list[Memory]:
        return self.list(owner, statuses=(ACTIVE,), limit=limit)

    def search_fts(self, owner: str, query: str, *, limit: int = 20) -> list[tuple[Memory, float]]:
        """Full-text candidates ordered by bm25 (lower = more relevant)."""
        if not (query or "").strip():
            return []
        sql = (
            f"SELECT m.id AS mid, bm25(memories_fts) AS score FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.memory_id "
            "WHERE memories_fts MATCH ? AND m.owner=? AND m.status=? "
            "ORDER BY score LIMIT ?"
        )
        out: list[tuple[Memory, float]] = []
        with self._cursor() as conn:
            try:
                rows = conn.execute(sql, (query, owner, ACTIVE, limit)).fetchall()
            except sqlite3.OperationalError:
                # A malformed FTS query (stray quotes/operators) must never crash
                # retrieval — fall back to no full-text hits.
                return []
            for r in rows:
                mem = conn.execute(
                    f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id=?", (r["mid"],)
                ).fetchone()
                if mem:
                    out.append((Memory.from_row(mem), float(r["score"])))
        return out

    # -- audit / revisions ---------------------------------------------------------

    def audit(self, event: str, *, owner: str | None = None, memory_id: str | None = None,
              memory_type: str | None = None, detail: str | None = None) -> None:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO memory_audit(ts, event, owner, memory_id, memory_type, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self.now_iso(), event, owner, memory_id, memory_type, detail),
            )

    def audit_recent(self, limit: int = 50) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT ts, event, owner, memory_id, memory_type, detail "
                "FROM memory_audit ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def revisions(self, memory_id: str) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT revision, title, content, subject, confidence, status, "
                "changed_at, reason FROM memory_revisions WHERE memory_id=? "
                "ORDER BY revision", (memory_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self, owner: str) -> dict:
        with self._cursor() as conn:
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM memories WHERE owner=? GROUP BY status",
                (owner,),
            ).fetchall()
            type_rows = conn.execute(
                "SELECT type, COUNT(*) AS n FROM memories WHERE owner=? AND status=? "
                "GROUP BY type", (owner, ACTIVE),
            ).fetchall()
        by_status = {r["status"]: r["n"] for r in status_rows}
        return {
            "active": by_status.get(ACTIVE, 0),
            "forgotten": by_status.get(FORGOTTEN, 0),
            "total": sum(by_status.values()),
            "by_type": {r["type"]: r["n"] for r in type_rows},
        }


# The proposal audit event is referenced by the service; re-export for convenience.
PROPOSAL = EV_PROPOSAL
