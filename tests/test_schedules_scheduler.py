"""Scheduler policy (Phase 5C, item 6): firing, missed-run/backlog, autorun vs
effect-pause, duplicate-run prevention across restart, and max_runs — with a fake
task runner and a real repository."""

import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.schedules.models import (
    ACTIVE, COMPLETED, ONE_TIME, RECURRING, REMINDER, RUN_COMPLETED, RUN_PAUSED,
    RUN_SKIPPED, Schedule,
)
from app.schedules.repository import ScheduleRepository
from app.schedules.scheduler import Scheduler

UTC = timezone.utc
T0 = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "schedule_misfire_grace_seconds", 300)
    monkeypatch.setattr(s, "schedule_max_backlog_runs", 3)
    monkeypatch.setattr(s, "schedule_poll_seconds", 15)
    monkeypatch.setattr(s, "schedule_timezone", "America/Guatemala")
    clock = {"t": T0}
    ids = (str(i) for i in itertools.count(1))
    repo = ScheduleRepository(tmp_path / "s.db", now_fn=lambda: clock["t"], id_factory=lambda: next(ids))
    notes, calls = [], []
    results = {"default": {"status": RUN_COMPLETED, "task_id": "task_x", "summary": "ok"}}

    def runner(schedule, now, settings):
        calls.append(schedule.schedule_id)
        return results.get(schedule.schedule_id, results["default"])

    scheduler = Scheduler(repo, runner, now_fn=lambda: clock["t"],
                          notifier=lambda t, m, **k: notes.append((t, m)),
                          settings_provider=lambda: s, owner_token="tester")
    return SimpleNamespace(settings=s, repo=repo, scheduler=scheduler, clock=clock,
                           notes=notes, calls=calls, results=results, tmp=tmp_path)


def _make(env, *, trigger, next_run, recurrence=None, max_runs=None, steps=None):
    template = {"steps": steps} if steps is not None else {"reminder_text": "ping"}
    s = Schedule(schedule_id=env.repo.new_schedule_id(), owner="local", title="ping",
                 timezone="America/Guatemala", trigger_type=trigger, next_run_at=next_run,
                 recurrence=recurrence, task_template=template, max_runs=max_runs)
    return env.repo.create(s)


def _iso(dt):
    return dt.isoformat(timespec="seconds")


# --- reminders ---------------------------------------------------------------------


def test_reminder_fires_once_and_completes(env):
    s = _make(env, trigger=REMINDER, next_run=_iso(T0 - timedelta(minutes=1)))
    env.scheduler.poll_once()
    assert len(env.notes) == 1
    assert env.repo.get(s.schedule_id).status == COMPLETED
    # Re-polling does not re-fire.
    env.scheduler.poll_once()
    assert len(env.notes) == 1


def test_recurring_reminder_advances(env):
    s = _make(env, trigger=REMINDER, next_run=_iso(T0 - timedelta(minutes=1)),
              recurrence={"freq": "interval", "seconds": 60})
    env.scheduler.poll_once()
    reloaded = env.repo.get(s.schedule_id)
    assert reloaded.status == ACTIVE
    assert datetime.fromisoformat(reloaded.next_run_at) > T0        # advanced to the future


# --- task autorun vs pause ---------------------------------------------------------


def test_read_only_task_autoruns(env):
    s = _make(env, trigger=ONE_TIME, next_run=_iso(T0 - timedelta(minutes=1)),
              steps=[{"intent": "browser_read", "arguments": {}}])
    env.scheduler.poll_once()
    assert env.calls == [s.schedule_id]                             # runner was invoked
    assert env.repo.runs(s.schedule_id)[0].status == RUN_COMPLETED


def test_effect_task_pauses_not_sends(env):
    s = _make(env, trigger=ONE_TIME, next_run=_iso(T0 - timedelta(minutes=1)),
              steps=[{"intent": "email_prepare_send", "arguments": {}}])
    env.results[s.schedule_id] = {"status": RUN_PAUSED, "task_id": "t1", "summary": "paused (0/1)"}
    env.scheduler.poll_once()
    assert env.repo.runs(s.schedule_id)[0].status == RUN_PAUSED     # prepared, never sent


# --- missed-run / backlog ----------------------------------------------------------


def test_task_beyond_grace_is_skipped(env):
    s = _make(env, trigger=ONE_TIME, next_run=_iso(T0 - timedelta(hours=1)),   # 60 min late
              steps=[{"intent": "browser_read", "arguments": {}}])
    env.scheduler.poll_once()
    assert env.calls == []                                          # runner NOT invoked
    assert env.repo.runs(s.schedule_id)[0].status == RUN_SKIPPED


def test_backlog_collapsed_to_single_run(env):
    # A recurring reminder missed many minutes → delivered ONCE, jumped to the future.
    s = _make(env, trigger=REMINDER, next_run=_iso(T0 - timedelta(hours=2)),
              recurrence={"freq": "interval", "seconds": 60})
    env.scheduler.poll_once()
    assert len(env.notes) == 1                                      # not a burst of catch-ups
    assert datetime.fromisoformat(env.repo.get(s.schedule_id).next_run_at) > T0
    assert any(e["event"] == "backlog_collapsed" for e in env.repo.audit_recent(s.schedule_id))


# --- duplicate-run prevention across restart --------------------------------------


def test_no_double_run_across_restart(env):
    s = _make(env, trigger=ONE_TIME, next_run=_iso(T0 - timedelta(minutes=1)),
              steps=[{"intent": "browser_read", "arguments": {}}])
    env.scheduler.poll_once()
    assert env.calls == [s.schedule_id]
    # "Restart": a brand-new scheduler over the same DB must not re-run (next_run
    # is gone / completed).
    fresh = Scheduler(env.repo, lambda sch, now, st: env.calls.append(sch.schedule_id) or {},
                      now_fn=lambda: env.clock["t"], settings_provider=lambda: env.settings,
                      owner_token="tester2")
    fresh.poll_once()
    assert env.calls == [s.schedule_id]                            # still exactly one run


def test_concurrent_pollers_single_run(env):
    s = _make(env, trigger=ONE_TIME, next_run=_iso(T0 - timedelta(minutes=1)),
              steps=[{"intent": "browser_read", "arguments": {}}])
    # Second poller shares the repo; the atomic lease + next_run advancement means
    # only one run happens even if both see it due.
    second = Scheduler(env.repo, lambda sch, now, st: env.calls.append("B") or
                       {"status": RUN_COMPLETED, "summary": "ok"},
                       now_fn=lambda: env.clock["t"], settings_provider=lambda: env.settings,
                       owner_token="B")
    env.scheduler.poll_once()
    second.poll_once()
    assert env.calls.count(s.schedule_id) + env.calls.count("B") == 1


# --- max runs ----------------------------------------------------------------------


def test_max_runs_completes(env):
    s = _make(env, trigger=REMINDER, next_run=_iso(T0 - timedelta(seconds=1)),
              recurrence={"freq": "interval", "seconds": 60}, max_runs=2)
    env.scheduler.poll_once()                                       # run 1
    env.clock["t"] = datetime.fromisoformat(
        env.repo.get(s.schedule_id).next_run_at) + timedelta(seconds=1)
    env.scheduler.poll_once()                                       # run 2 → max reached
    assert len(env.notes) == 2
    assert env.repo.get(s.schedule_id).status == COMPLETED
