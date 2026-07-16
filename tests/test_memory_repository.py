"""Storage layer (Phase 5A): migrations, corruption recovery, owner isolation,
duplicates, conflict keys, revisions, soft/permanent delete, FTS sync."""

import itertools
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.memory import migrations
from app.memory.models import ACTIVE, FORGOTTEN, Memory, content_hash, normalize
from app.memory.repository import MemoryRepository


@pytest.fixture
def repo(tmp_path):
    ids = (f"mem_{i}" for i in itertools.count(1))
    clock = {"t": datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)}
    r = MemoryRepository(tmp_path / "mem.db", now_fn=lambda: clock["t"],
                         id_factory=lambda: next(ids))
    r._clock = clock
    return r


def _mem(repo, *, owner="local", type="preference", title="Editor preferido",
         content="Mi editor preferido es VS Code", subject="editor preferido",
         entities=("vs code",), tags=("editor",)):
    return Memory(id=repo.new_id(), owner=owner, type=type, title=title, content=content,
                  entities=list(entities), tags=list(tags), subject=normalize(subject),
                  content_hash=content_hash(owner, type, content))


# --- migrations + WAL --------------------------------------------------------------


def test_schema_version_and_wal(tmp_path):
    db = tmp_path / "m.db"
    conn = migrations.open_database(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memories", "memory_revisions", "memory_audit"} <= tables
    conn.close()


def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "m.db"
    migrations.open_database(db).close()
    # Re-opening applies nothing new and does not raise.
    conn = migrations.open_database(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
    conn.close()


def test_corruption_recovery(tmp_path):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"definitely not a sqlite database " * 40)
    # Opening a corrupt file recovers: the bad file is backed up, a fresh DB is made.
    r = MemoryRepository(db, now_fn=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                         id_factory=lambda: "mem_1")
    r.insert(_mem(r))
    assert r.counts("local")["active"] == 1
    backups = list(tmp_path.glob("corrupt.db.corrupt.*"))
    assert backups, "a corrupt-database backup should have been created"


def test_integrity_ok_detects_garbage(tmp_path):
    db = tmp_path / "g.db"
    db.write_bytes(b"garbage")
    conn = sqlite3.connect(db)
    assert migrations.integrity_ok(conn) is False
    conn.close()


# --- insert / duplicate / conflict -------------------------------------------------


def test_insert_creates_revision_and_fts(repo):
    m = _mem(repo)
    repo.insert(m)
    assert repo.get(m.id).revision == 1
    assert [r["reason"] for r in repo.revisions(m.id)] == ["created"]
    assert [mm.id for mm, _ in repo.search_fts("local", '"preferido"')] == [m.id]


def test_find_duplicate(repo):
    m = _mem(repo)
    repo.insert(m)
    h = content_hash("local", "preference", "Mi editor preferido es VS Code")
    assert repo.find_duplicate("local", "preference", h).id == m.id
    assert repo.find_duplicate("local", "preference", "nope") is None


def test_find_conflict_same_subject_different_value(repo):
    m = _mem(repo)
    repo.insert(m)
    new_hash = content_hash("local", "preference", "Mi editor preferido es Neovim")
    conflict = repo.find_conflict("local", "preference", "editor preferido", new_hash)
    assert conflict.id == m.id
    # Same value (same hash) is not a conflict.
    assert repo.find_conflict("local", "preference", "editor preferido", m.content_hash) is None
    # No subject → never a conflict.
    assert repo.find_conflict("local", "preference", "", new_hash) is None


# --- update / revisions ------------------------------------------------------------


def test_update_in_place_preserves_id_and_history(repo):
    m = _mem(repo)
    repo.insert(m)
    m.content = "Mi editor preferido es Neovim"
    m.content_hash = content_hash("local", "preference", m.content)
    repo.update(m)
    stored = repo.get(m.id)
    assert stored.id == m.id and stored.revision == 2
    assert stored.content == "Mi editor preferido es Neovim"
    revs = repo.revisions(m.id)
    assert [(r["revision"], r["reason"]) for r in revs] == [(1, "created"), (2, "updated")]
    assert revs[0]["content"] == "Mi editor preferido es VS Code"     # old value preserved


# --- soft forget / restore / permanent delete --------------------------------------


def test_soft_forget_excludes_from_fts_but_keeps_row(repo):
    m = _mem(repo)
    repo.insert(m)
    assert repo.soft_forget([m.id]) == 1
    assert repo.get(m.id).status == FORGOTTEN
    assert repo.search_fts("local", '"preferido"') == []       # excluded from retrieval
    assert repo.counts("local") == {"active": 0, "forgotten": 1, "total": 1, "by_type": {}}


def test_restore_reindexes(repo):
    m = _mem(repo)
    repo.insert(m)
    repo.soft_forget([m.id])
    assert repo.restore(m.id).status == ACTIVE
    assert [mm.id for mm, _ in repo.search_fts("local", '"preferido"')] == [m.id]


def test_permanent_delete_removes_everything(repo):
    m = _mem(repo)
    repo.insert(m)
    assert repo.permanent_delete([m.id]) == 1
    assert repo.get(m.id) is None
    assert repo.revisions(m.id) == []
    assert repo.search_fts("local", '"preferido"') == []


# --- owner isolation ---------------------------------------------------------------


def test_owner_isolation(repo):
    mine = _mem(repo, owner="local")
    theirs = _mem(repo, owner="intruder", content="Su editor es Emacs", subject="editor")
    repo.insert(mine)
    repo.insert(theirs)
    assert [m.id for m in repo.list("local")] == [mine.id]
    assert [m.id for m in repo.list("intruder")] == [theirs.id]
    # FTS is owner-scoped too.
    assert all(m.owner == "local" for m, _ in repo.search_fts("local", '"editor"'))


def test_expiry_helper_moves_forward(repo):
    base = repo.now_iso()
    later = repo.future_iso(120)
    assert datetime.fromisoformat(later) - datetime.fromisoformat(base) == timedelta(seconds=120)
