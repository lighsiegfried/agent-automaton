"""Schema creation, versioned migrations, and corruption recovery (Phase 5A).

The memory store is a local SQLite database in WAL mode with a schema version
tracked in ``PRAGMA user_version``. ``open_database`` is the single entry point:
it opens the file, verifies integrity, transparently recovers a corrupt database
(the bad file is moved aside and a fresh one is created — memory loss is preferred
to a crash loop), enables WAL, and applies any pending migrations.

No cloud dependency, no ORM — just stdlib ``sqlite3``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.logger import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1

# --- v1 schema ---------------------------------------------------------------------
_V1 = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    type          TEXT NOT NULL,
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    entities      TEXT NOT NULL DEFAULT '[]',
    tags          TEXT NOT NULL DEFAULT '[]',
    subject       TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'api',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    last_used_at  TEXT,
    confidence    REAL NOT NULL DEFAULT 0.8,
    expires_at    TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    content_hash  TEXT NOT NULL,
    revision      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_memories_owner_status ON memories(owner, status);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(owner, type, subject, status);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(owner, content_hash, status);

CREATE TABLE IF NOT EXISTS memory_revisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    entities      TEXT NOT NULL DEFAULT '[]',
    tags          TEXT NOT NULL DEFAULT '[]',
    subject       TEXT NOT NULL DEFAULT '',
    confidence    REAL,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    changed_at    TEXT NOT NULL,
    reason        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revisions_memory ON memory_revisions(memory_id, revision);

CREATE TABLE IF NOT EXISTS memory_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    event         TEXT NOT NULL,
    owner         TEXT,
    memory_id     TEXT,
    memory_type   TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON memory_audit(ts);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    title,
    content,
    entities,
    tags,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_V1)


# version -> migration callable. Applied in ascending order for versions above
# the database's current ``user_version``.
MIGRATIONS = {
    1: _migrate_to_v1,
}


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Bring the schema up to ``SCHEMA_VERSION``. Returns the resulting version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version in range(current + 1, SCHEMA_VERSION + 1):
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            continue
        log.info("memory: migrating schema to v%d", version)
        migrate(conn)
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    return conn.execute("PRAGMA user_version").fetchone()[0]


def integrity_ok(conn: sqlite3.Connection) -> bool:
    """True when the database passes SQLite's quick integrity check."""
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row) and row[0] == "ok"


def _backup_corrupt(db_path: Path) -> Path | None:
    """Move a corrupt database (and its WAL/SHM sidecars) aside so a fresh one can
    be created. Returns the backup path, or None if there was nothing to move."""
    if not db_path.exists():
        return None
    index = 0
    while True:
        candidate = db_path.with_suffix(db_path.suffix + f".corrupt.{index}")
        if not candidate.exists():
            break
        index += 1
    try:
        db_path.replace(candidate)
    except OSError:
        # Last resort: delete so the store can be rebuilt rather than crash-looping.
        try:
            db_path.unlink()
        except OSError:
            return None
        candidate = None
    for sidecar in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + sidecar)
        if side.exists():
            try:
                side.unlink()
            except OSError:
                pass
    log.warning("memory: corrupt database backed up to %s", candidate)
    return candidate


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def open_database(db_path: Path, *, recover: bool = True) -> sqlite3.Connection:
    """Open (or create) the memory database, WAL-enabled and migrated.

    If the existing file is corrupt and ``recover`` is set, it is moved aside and
    a new database is created in its place — a local personal store should degrade
    to empty rather than block Fifi from starting.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect(db_path)
    if recover and db_path.exists() and not integrity_ok(conn):
        conn.close()
        _backup_corrupt(db_path)
        conn = _connect(db_path)

    # WAL is a no-op for in-memory databases; ignore the returned mode.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass

    try:
        apply_migrations(conn)
    except sqlite3.DatabaseError:
        # A schema step failed on a damaged file — recover once, then re-migrate.
        if not recover:
            raise
        conn.close()
        _backup_corrupt(db_path)
        conn = _connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        apply_migrations(conn)
    return conn
