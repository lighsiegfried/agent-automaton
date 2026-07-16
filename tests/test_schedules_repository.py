"""Schedule storage (Phase 5C): migrations, corruption recovery, owner scope, the
atomic execution lease, runs history, and audit."""

import itertools
import sqlite3
from datetime import datetime, timezone

import pytest

from app.schedules import repository as repo_mod
from app.schedules.models import ACTIVE, ONE_TIME, REMINDER, Schedule, ScheduleRun
from app.schedules.repository import ScheduleRepository


@pytest.fixture
def repo(tmp_path):
    ids = (str(i) for i in itertools.count(1))
    return ScheduleRepository(tmp_path / "s.db",
                              now_fn=lambda: datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
                              id_factory=lambda: next(ids))


def _sched(repo, *, owner="local", next_run="2026-07-15T12:00:00+00:00", status=ACTIVE,
           trigger=REMINDER):
    s = Schedule(schedule_id=repo.new_schedule_id(), owner=owner, title="ping",
                 timezone="America/Guatemala", trigger_type=trigger, next_run_at=next_run,
                 status=status, task_template={"reminder_text": "ping"})
    return repo.create(s)


def test_schema_and_wal(tmp_path):
    conn = repo_mod.open_database(tmp_path / "s.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schedules", "schedule_runs", "schedule_audit"} <= tables
    conn.close()


def test_corruption_recovery(tmp_path):
    db = tmp_path / "s.db"
    db.write_bytes(b"garbage not a db " * 40)
    repo = ScheduleRepository(db, now_fn=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                              id_factory=lambda: "1")
    _sched(repo)
    assert len(repo.list("local")) == 1
    assert list(tmp_path.glob("s.db.corrupt.*"))


def test_create_reload_persists(repo):
    s = _sched(repo)
    reloaded = repo.get(s.schedule_id)
    assert reloaded.title == "ping" and reloaded.status == ACTIVE
    assert reloaded.reminder_text() == "ping"


def test_owner_isolation(repo):
    _sched(repo, owner="local")
    _sched(repo, owner="intruder")
    assert [s.owner for s in repo.list("local")] == ["local"]
    assert [s.owner for s in repo.list("intruder")] == ["intruder"]


def test_due_query(repo):
    _sched(repo, next_run="2026-07-15T11:00:00+00:00")     # past → due
    _sched(repo, next_run="2026-07-15T13:00:00+00:00")     # future → not due
    due = repo.due("2026-07-15T12:00:00+00:00")
    assert len(due) == 1 and due[0].next_run_at == "2026-07-15T11:00:00+00:00"


# --- atomic execution lease --------------------------------------------------------


def test_atomic_lease_single_winner(repo):
    s = _sched(repo)
    now = "2026-07-15T12:00:00+00:00"
    until = "2026-07-15T12:05:00+00:00"
    assert repo.acquire_lease(s.schedule_id, "workerA", now, until) is True
    # A second worker cannot claim a live lease.
    assert repo.acquire_lease(s.schedule_id, "workerB", now, until) is False
    # After release, it can be claimed again.
    repo.release_lease(s.schedule_id)
    assert repo.acquire_lease(s.schedule_id, "workerB", now, until) is True


def test_expired_lease_reclaimable(repo):
    s = _sched(repo)
    repo.acquire_lease(s.schedule_id, "workerA", "2026-07-15T12:00:00+00:00",
                       "2026-07-15T12:01:00+00:00")
    # Now is past the lease expiry → another worker may reclaim.
    assert repo.acquire_lease(s.schedule_id, "workerB", "2026-07-15T12:02:00+00:00",
                              "2026-07-15T12:07:00+00:00") is True


def test_lease_only_on_active(repo):
    from app.schedules.models import PAUSED
    s = _sched(repo, status=PAUSED)
    assert repo.acquire_lease(s.schedule_id, "w", "2026-07-15T12:00:00+00:00",
                              "2026-07-15T12:05:00+00:00") is False


# --- runs / audit ------------------------------------------------------------------


def test_runs_and_audit(repo):
    s = _sched(repo)
    repo.record_run(ScheduleRun(run_id=repo.new_run_id(), schedule_id=s.schedule_id,
                                scheduled_for=s.next_run_at, ran_at="2026-07-15T12:00:00+00:00",
                                status="delivered", summary="ok"))
    assert repo.runs(s.schedule_id)[0].status == "delivered"
    repo.audit("schedule_created", schedule_id=s.schedule_id, detail="x")
    assert repo.audit_recent(s.schedule_id)[0]["event"] == "schedule_created"
