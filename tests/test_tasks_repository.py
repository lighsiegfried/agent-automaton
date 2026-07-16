"""Task storage (Phase 5B): migrations, corruption recovery, owner scope, crash-safe
persistence, audit, and the single-effect-slot query."""

import itertools
import sqlite3
from datetime import datetime, timezone

import pytest

from app.tasks import repository as repo_mod
from app.tasks.models import (
    AWAITING_CONFIRMATION, COMPLETED, PLANNED, STEP_AWAITING, Step, Task,
)
from app.tasks.repository import TaskRepository


@pytest.fixture
def repo(tmp_path):
    ids = (str(i) for i in itertools.count(1))
    return TaskRepository(tmp_path / "t.db",
                          now_fn=lambda: datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                          id_factory=lambda: next(ids))


def _task(repo, owner="local", status=PLANNED, steps=1):
    tid = repo.new_task_id()
    task = Task(task_id=tid, owner=owner, user_request="do things", title="T", status=status,
                max_steps=12)
    for i in range(steps):
        task.steps.append(Step(step_id=repo.new_step_id(), task_id=tid, position=i,
                               intent="memory_search", arguments={"query": "x"},
                               domain="memory", risk_level="read_only",
                               idempotency_key=f"idem_{tid}_{i}"))
    return repo.create_task(task)


def test_schema_and_wal(tmp_path):
    conn = repo_mod.open_database(tmp_path / "t.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == repo_mod.SCHEMA_VERSION
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "task_steps", "task_audit"} <= tables
    conn.close()


def test_corruption_recovery(tmp_path):
    db = tmp_path / "t.db"
    db.write_bytes(b"not a database at all " * 40)
    repo = TaskRepository(db, now_fn=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                          id_factory=lambda: "1")
    _task(repo)
    assert len(repo.list_tasks("local")) == 1
    assert list(tmp_path.glob("t.db.corrupt.*"))


def test_integrity_check_detects_garbage(tmp_path):
    db = tmp_path / "g.db"
    db.write_bytes(b"garbage")
    conn = sqlite3.connect(db)
    assert repo_mod._integrity_ok(conn) is False
    conn.close()


def test_create_and_reload_persists_steps(repo):
    task = _task(repo, steps=3)
    reloaded = repo.get_task(task.task_id)
    assert reloaded.status == PLANNED
    assert [s.position for s in reloaded.steps] == [0, 1, 2]
    assert reloaded.steps[0].intent == "memory_search"


def test_save_task_and_step_round_trip(repo):
    task = _task(repo)
    task.status = COMPLETED
    task.final_summary = "done"
    repo.save_task(task)
    step = task.steps[0]
    step.status = "completed"
    step.result_summary = "found 1"
    repo.save_step(step)
    reloaded = repo.get_task(task.task_id)
    assert reloaded.status == COMPLETED and reloaded.final_summary == "done"
    assert reloaded.steps[0].status == "completed"


def test_owner_isolation(repo):
    _task(repo, owner="local")
    _task(repo, owner="intruder")
    assert [t.owner for t in repo.list_tasks("local")] == ["local"]
    assert [t.owner for t in repo.list_tasks("intruder")] == ["intruder"]


def test_active_effect_task_query(repo):
    a = _task(repo, owner="local")
    a.status = AWAITING_CONFIRMATION
    repo.save_task(a)
    assert repo.active_effect_task("local") == a.task_id
    assert repo.active_effect_task("local", exclude=a.task_id) is None


def test_step_by_action_id(repo):
    task = _task(repo)
    step = task.steps[0]
    step.status = STEP_AWAITING
    step.action_id = "act_123"
    repo.save_step(step)
    found = repo.step_by_action_id("act_123")
    assert found is not None and found.step_id == step.step_id


def test_audit_trail(repo):
    task = _task(repo)
    repo.audit("plan_created", task_id=task.task_id, detail="x")
    repo.audit("step_started", task_id=task.task_id, step_id=task.steps[0].step_id)
    events = [e["event"] for e in repo.audit_recent(task.task_id)]
    assert events[:2] == ["step_started", "plan_created"]
