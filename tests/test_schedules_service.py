"""ScheduleService behaviour (Phase 5C): sensitive prepare→confirm (exact phrase,
wake blocked), ambiguous-date rejection, template validation, lifecycle, owner
isolation, and update-preserves-history — through the real dispatcher/broker."""

import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
from app.schedules import service as sched_service
from app.schedules.service import ScheduleService

UTC = timezone.utc


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_schedules", True)
    monkeypatch.setattr(s, "schedule_owner", "local")
    monkeypatch.setattr(s, "schedules_db_path", tmp_path / "s.db")
    monkeypatch.setattr(s, "schedule_timezone", "America/Guatemala")
    monkeypatch.setattr(s, "enable_multi_step_tasks", True)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": datetime(2026, 7, 15, 12, 0, tzinfo=UTC)}
    ids = (str(i) for i in itertools.count(1))
    svc = ScheduleService(db_path=tmp_path / "s.db", now_fn=lambda: clock["t"],
                          id_factory=lambda: next(ids))
    monkeypatch.setattr(sched_service, "_service", svc)
    return SimpleNamespace(settings=s, svc=svc, clock=clock, broker=pending.get_pending_broker())


def _prepare(env, **args):
    return env.svc.prepare(args, "es", env.settings, env.broker)


def _confirm(phrase, wake=False):
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
        language="es", wake=wake)


def _count(env, owner="local"):
    return len(env.svc.repo.list(owner))


# --- sensitive prepare → confirm ---------------------------------------------------


def test_prepare_reminder_needs_confirmation(env):
    r = _prepare(env, text="llamar a Ana mañana a las nueve", kind="reminder")
    assert r["status"] == "needs_confirmation"
    assert env.broker.summary()["domain"] == "schedule_write"
    assert "09:00" in r["data"]["when_local"]                # resolved absolute time shown
    assert _count(env) == 0                                   # not stored until confirmed


@pytest.mark.parametrize("phrase", ["confirmar programación", "crear recordatorio", "confirm schedule"])
def test_accepted_phrases_commit(env, phrase):
    _prepare(env, text="llamar a Ana mañana a las nueve", kind="reminder")
    assert _confirm(phrase)["status"] == "executed"
    assert _count(env) == 1


@pytest.mark.parametrize("phrase", ["sí", "si", "yes", "ok", "confirmar"])
def test_plain_yes_never_commits(env, phrase):
    _prepare(env, text="llamar a Ana mañana a las nueve", kind="reminder")
    assert _confirm(phrase)["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0


def test_wake_cannot_confirm(env):
    _prepare(env, text="llamar a Ana mañana a las nueve", kind="reminder")
    assert _confirm("confirmar programación", wake=True)["error_code"] == errors.CONFIRMATION_MISMATCH
    assert _count(env) == 0
    assert _confirm("confirmar programación")["status"] == "executed"   # deliberate still works


def test_expired_proposal_rejected(env):
    _prepare(env, text="mañana a las 9", kind="reminder")
    env.clock["t"] = env.clock["t"] + timedelta(seconds=env.settings.memory_action_expires_seconds + 10)
    assert _confirm("confirmar programación")["error_code"] == errors.ACTION_EXPIRED
    assert _count(env) == 0


# --- date resolution ---------------------------------------------------------------


def test_ambiguous_date_rejected(env):
    r = _prepare(env, text="llamar a Ana el viernes", kind="reminder")
    assert r["error_code"] == errors.AMBIGUOUS_DATE
    assert env.broker.summary() is None                       # no pending created


def test_invalid_timezone_rejected(env):
    assert _prepare(env, text="mañana a las 9", kind="reminder",
                    timezone="Mars/Phobos")["error_code"] == errors.INVALID_TIMEZONE


# --- task templates ----------------------------------------------------------------


def test_task_preview_lists_steps_needing_confirmation(env):
    r = _prepare(env, text="cada día a las 9", kind="task", title="resumen",
                 steps=[{"intent": "browser_search", "arguments": {"query": "x"}},
                        {"intent": "email_draft_new", "arguments": {"to": ["a@x.com"], "body": "h"},
                         "dependencies": [0]}])
    assert r["status"] == "needs_confirmation" and r["data"]["trigger_type"] == "recurring"
    needing = [p["intent"] for p in r["data"]["steps_requiring_confirmation"]]
    assert needing == ["email_draft_new"]                     # the read isn't sensitive


def test_invalid_template_rejected(env):
    r = _prepare(env, text="cada día a las 9", kind="task",
                 steps=[{"intent": "delete_everything", "arguments": {}}])
    assert r["error_code"] == errors.UNSUPPORTED_STEP and _count(env) == 0


# --- lifecycle ---------------------------------------------------------------------


def test_pause_resume_cancel(env):
    _prepare(env, text="cada día a las 9", kind="task",
             steps=[{"intent": "browser_read", "arguments": {}}])
    sid = _confirm("confirmar programación")["data"]["schedule"]["schedule_id"]
    assert env.svc.pause(sid, env.settings, "es")["status"] == "ok"
    assert env.svc.get(sid, env.settings)["status"] == "paused"
    assert env.svc.resume(sid, env.settings, "es")["status"] == "ok"
    assert env.svc.get(sid, env.settings)["status"] == "active"
    assert env.svc.cancel(sid, env.settings, "es")["status"] == "cancelled"
    assert env.svc.get(sid, env.settings)["status"] == "cancelled"


def test_resume_rolls_recurring_past_due_forward(env):
    _prepare(env, text="cada día a las 9", kind="task",
             steps=[{"intent": "browser_read", "arguments": {}}])
    sid = _confirm("confirmar programación")["data"]["schedule"]["schedule_id"]
    env.svc.pause(sid, env.settings, "es")
    env.clock["t"] = env.clock["t"] + timedelta(days=3)       # miss several occurrences
    env.svc.resume(sid, env.settings, "es")
    nxt = datetime.fromisoformat(env.svc.get(sid, env.settings)["next_run_at"])
    assert nxt > env.clock["t"]                                # rolled forward, no backlog


# --- owner isolation + update history ----------------------------------------------


def test_owner_isolation(env, monkeypatch):
    _prepare(env, text="mañana a las 9", kind="reminder")
    _confirm("confirmar programación")
    monkeypatch.setattr(env.settings, "schedule_owner", "someone_else")
    assert env.svc.list({}, "es", env.settings, env.broker)["data"]["count"] == 0
    monkeypatch.setattr(env.settings, "schedule_owner", "local")
    assert env.svc.list({}, "es", env.settings, env.broker)["data"]["count"] == 1


def test_update_preserves_run_history(env):
    _prepare(env, text="cada día a las 9", kind="task",
             steps=[{"intent": "browser_read", "arguments": {}}])
    sid = _confirm("confirmar programación")["data"]["schedule"]["schedule_id"]
    from app.schedules.models import ScheduleRun
    env.svc.repo.record_run(ScheduleRun(run_id="run_x", schedule_id=sid, scheduled_for=None,
                                        ran_at="2026-07-15T12:00:00+00:00", status="completed",
                                        summary="did a thing"))
    # Modify the schedule → same id, run history intact.
    env.svc.update_prepare(sid, {"text": "cada día a las 10", "kind": "task",
                                 "steps": [{"intent": "browser_read", "arguments": {}}]},
                           "es", env.settings, env.broker)
    _confirm("confirmar programación")
    assert env.svc.get(sid, env.settings)["title"]            # still there
    assert any(r["run_id"] == "run_x" for r in env.svc.runs(sid, env.settings))
