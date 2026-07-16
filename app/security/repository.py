"""SQLite persistence for security state (Phase 6B).

Stores the salted password verifier (never plaintext), failed-unlock/lockout counters
(so a lockout survives a restart), elevations, and a redacted audit trail — in a WAL
database with migrations, owner scope, and corruption recovery. Elevations never
survive a reboot: the service clears them at startup.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.security.models import Elevation, Verifier

log = get_logger(__name__)

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS security_config (
    owner          TEXT PRIMARY KEY,
    verifier       TEXT,
    failed_unlocks INTEGER NOT NULL DEFAULT 0,
    lockout_until  TEXT,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS elevations (
    elevation_id       TEXT PRIMARY KEY,
    owner              TEXT NOT NULL,
    from_profile       TEXT NOT NULL,
    to_profile         TEXT NOT NULL,
    capability         TEXT NOT NULL,
    granted_at         TEXT NOT NULL,
    expires_at_iso     TEXT NOT NULL,
    revoked            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_elev_owner ON elevations(owner, revoked);
CREATE TABLE IF NOT EXISTS security_audit (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT NOT NULL,
    owner  TEXT,
    event  TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_sec_audit ON security_audit(owner, id);
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
    log.warning("security: corrupt database backed up/reset (%s)", db_path)


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
        log.info("security: schema migrated to v%d", SCHEMA_VERSION)
    return conn


class SecurityRepository:
    def __init__(self, db_path, *, now_fn=None):
        self.db_path = Path(db_path)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
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

    # -- config / verifier ---------------------------------------------------------

    def get_config(self, owner: str) -> dict:
        with self._cursor() as conn:
            r = conn.execute("SELECT verifier, failed_unlocks, lockout_until FROM security_config "
                             "WHERE owner=?", (owner,)).fetchone()
        if r is None:
            return {"verifier": None, "failed_unlocks": 0, "lockout_until": None}
        return {"verifier": json.loads(r["verifier"]) if r["verifier"] else None,
                "failed_unlocks": r["failed_unlocks"], "lockout_until": r["lockout_until"]}

    def _save_config(self, owner, verifier_json, failed, lockout):
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO security_config(owner, verifier, failed_unlocks, lockout_until, updated_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(owner) DO UPDATE SET verifier=excluded.verifier, "
                "failed_unlocks=excluded.failed_unlocks, lockout_until=excluded.lockout_until, "
                "updated_at=excluded.updated_at",
                (owner, verifier_json, failed, lockout, self.now_iso()))

    def set_verifier(self, owner: str, verifier: Verifier) -> None:
        cfg = self.get_config(owner)
        self._save_config(owner, json.dumps(verifier.__dict__), cfg["failed_unlocks"], cfg["lockout_until"])

    def get_verifier(self, owner: str) -> Verifier | None:
        v = self.get_config(owner)["verifier"]
        return Verifier(**v) if v else None

    def set_lockout(self, owner: str, *, failed: int, lockout_until) -> None:
        cfg = self.get_config(owner)
        vj = json.dumps(cfg["verifier"]) if cfg["verifier"] else None
        self._save_config(owner, vj, failed, lockout_until)

    # -- elevations ----------------------------------------------------------------

    def create_elevation(self, elev: Elevation) -> Elevation:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO elevations(elevation_id, owner, from_profile, to_profile, capability, "
                "granted_at, expires_at_iso, revoked) VALUES (?,?,?,?,?,?,?,0)",
                (elev.elevation_id, elev.owner, elev.from_profile, elev.to_profile, elev.capability,
                 elev.granted_at, elev.expires_at_iso))
        return elev

    def active_elevations(self, owner: str) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute("SELECT * FROM elevations WHERE owner=? AND revoked=0", (owner,)).fetchall()
        return [dict(r) for r in rows]

    def revoke_all_elevations(self, owner: str) -> int:
        with self._cursor() as conn:
            cur = conn.execute("UPDATE elevations SET revoked=1 WHERE owner=? AND revoked=0", (owner,))
            return cur.rowcount

    # -- audit ---------------------------------------------------------------------

    def audit(self, event: str, *, owner=None, detail=None) -> None:
        with self._cursor() as conn:
            conn.execute("INSERT INTO security_audit(ts, owner, event, detail) VALUES (?,?,?,?)",
                         (self.now_iso(), owner, event, detail))

    def audit_recent(self, owner=None, limit=100) -> list[dict]:
        with self._cursor() as conn:
            if owner:
                rows = conn.execute("SELECT ts, owner, event, detail FROM security_audit "
                                    "WHERE owner=? ORDER BY id DESC LIMIT ?", (owner, limit)).fetchall()
            else:
                rows = conn.execute("SELECT ts, owner, event, detail FROM security_audit "
                                    "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def new_id(self) -> str:
        return "elev_" + uuid.uuid4().hex
