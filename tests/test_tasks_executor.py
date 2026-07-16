"""Executor mechanics (Phase 5B, items 5-8) with a scriptable fake dispatch:
approval gating, safe-step execution, pause at effects, confirmation reconciliation,
read-only retry vs no-effect-retry, timeouts, concurrency, cancellation, idempotency,
and restart inspection. The real domain-confirmation path is covered end-to-end in
test_tasks_pipeline.py."""

import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.tasks import service as task_service
from app.tasks.service import TaskService


def DONE():
    return {"status": "ok", "error_code": None, "state": errors.COMPLETED, "action_id": None,
            "spoken": "", "data": {}}


def PAUSE(action_id="act1"):
    return {"status": "needs_confirmation", "error_code": None,
            "state": errors.AWAITING_CONFIRMATION, "action_id": action_id, "spoken": "", "data": {}}


def FAIL(code="EXECUTION_FAILED"):
    return {"status": "rejected", "error_code": code, "state": errors.BLOCKED,
            "action_id": None, "spoken": "", "data": {}}


class FakeDispatch:
    """Stands in for conversation.dispatch: scripted per-intent results (a queue,
    then a sticky default read-only success). A ``needs_confirmation`` result also
    registers a pending in the real broker, so the executor sees realistic broker
    state (a domain service would do the same)."""

    def __init__(self):
        self.script: dict[str, list] = {}
        self.calls: list[str] = []

    def set(self, intent, *results):
        self.script[intent] = list(results)

    def __call__(self, command, *, language="es", wake=False, settings=None):
        self.calls.append(command.intent)
        if command.is_confirmation or command.intent == "confirm":
            pending.get_pending_broker().clear()
            return {"status": "executed", "error_code": None, "state": errors.COMPLETED,
                    "action_id": None, "spoken": "", "data": {}}
        queue = self.script.get(command.intent)
        result = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else DONE()
        if result["status"] == "needs_confirmation":
            pending.get_pending_broker().register(
                domain="fake", action_id=result["action_id"], target="",
                expires_at="", cancel=lambda: None)
        return result


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_multi_step_tasks", True)
    monkeypatch.setattr(settings, "enable_memory", False)
    monkeypatch.setattr(settings, "task_owner", "local")
    monkeypatch.setattr(settings, "tasks_db_path", tmp_path / "t.db")
    monkeypatch.setattr(settings, "task_max_step_retries", 1)
    monkeypatch.setattr(settings, "task_max_runtime_minutes", 20)
    monkeypatch.setattr(settings, "task_require_plan_approval", True)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    fake = FakeDispatch()
    monkeypatch.setattr(conversation, "dispatch", fake)
    clock = {"t": datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)}
    ids = (str(i) for i in itertools.count(1))
    svc = TaskService(db_path=tmp_path / "t.db", now_fn=lambda: clock["t"],
                      id_factory=lambda: next(ids))
    monkeypatch.setattr(task_service, "_service", svc)
    return SimpleNamespace(settings=settings, svc=svc, fake=fake, clock=clock, tmp=tmp_path)


def _plan(env, steps):
    r = env.svc.plan({"steps": steps}, "es", env.settings)
    return r["data"]["task"]["task_id"], r


def _status(env, tid):
    return env.svc.get(tid, env.settings)["status"]


# --- approval gating ---------------------------------------------------------------


def test_run_requires_approval(env):
    tid, _ = _plan(env, [{"intent": "memory_search", "arguments": {"query": "x"}}])
    assert env.svc.run(tid, env.settings, "es")["error_code"] == errors.PLAN_NOT_APPROVED
    assert env.fake.calls == []                      # nothing dispatched without approval


def test_approval_mismatch(env):
    tid, _ = _plan(env, [{"intent": "memory_search", "arguments": {"query": "x"}}])
    assert env.svc.approve(tid, "hazlo ya", env.settings, "es")["error_code"] == errors.APPROVAL_MISMATCH
    assert _status(env, tid) == "planned"


# --- safe steps run automatically; approval never dispatches an effect -------------


def test_safe_steps_execute_after_approval(env):
    tid, _ = _plan(env, [
        {"intent": "memory_search", "arguments": {"query": "x"}},
        {"intent": "browser_read", "arguments": {}, "dependencies": [0]},
    ])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    assert env.fake.calls == []                      # approval alone runs nothing
    env.svc.run(tid, env.settings, "es")
    assert _status(env, tid) == "completed"
    assert env.fake.calls == ["memory_search", "browser_read"]


# --- pause at effect; approval does not pre-confirm --------------------------------


def test_pause_at_effect_step(env):
    env.fake.set("email_draft_new", PAUSE("place_1"))
    tid, _ = _plan(env, [{"intent": "email_draft_new",
                          "arguments": {"to": ["ana@x.com"], "body": "hi"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    out = env.svc.run(tid, env.settings, "es")
    assert out["status"] == "needs_confirmation"
    task = env.svc.get(tid, env.settings)
    assert task["status"] == "awaiting_confirmation"
    assert task["steps"][0]["status"] == "awaiting_confirmation"


def test_confirmation_reconciles_then_resume_advances(env):
    env.fake.set("email_draft_new", PAUSE("place_1"))
    tid, _ = _plan(env, [
        {"intent": "email_draft_new", "arguments": {"to": ["ana@x.com"], "body": "hi"}},
        {"intent": "browser_read", "arguments": {}, "dependencies": [0]},
    ])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    # Simulate the domain confirmation completing the effect (the listener path).
    env.svc._executor.on_effect_complete({"status": "placed"}, "place_1")
    assert _status(env, tid) == "paused"
    env.svc.resume(tid, env.settings, "es")          # "continuar tarea"
    assert _status(env, tid) == "completed"


def test_bare_resume_cannot_confirm_effect(env):
    env.fake.set("email_prepare_send", PAUSE("send_1"))
    tid, _ = _plan(env, [{"intent": "email_prepare_send", "arguments": {}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    out = env.svc.resume(tid, env.settings, "es")    # no phrase
    assert out["error_code"] == errors.CONFIRMATION_REQUIRED
    assert _status(env, tid) == "awaiting_confirmation"


# --- retry policy ------------------------------------------------------------------


def test_read_only_retries_once_then_succeeds(env):
    env.fake.set("browser_search", FAIL(), DONE())
    tid, _ = _plan(env, [{"intent": "browser_search", "arguments": {"query": "x"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    assert _status(env, tid) == "completed"
    assert env.fake.calls.count("browser_search") == 2      # one retry
    assert env.svc.get(tid, env.settings)["steps"][0]["attempt_count"] == 2


def test_read_only_retry_exhausted_fails(env):
    env.fake.set("browser_search", FAIL(), FAIL())
    tid, _ = _plan(env, [{"intent": "browser_search", "arguments": {"query": "x"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    out = env.svc.run(tid, env.settings, "es")
    assert out["status"] == "error" and _status(env, tid) == "failed"
    assert env.fake.calls.count("browser_search") == 2      # 1 + 1 retry, then stop


def test_effect_never_retries(env):
    env.fake.set("email_draft_new", FAIL(errors.LOGIN_REQUIRED))
    tid, _ = _plan(env, [{"intent": "email_draft_new",
                          "arguments": {"to": ["ana@x.com"], "body": "hi"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    assert _status(env, tid) == "failed"
    assert env.fake.calls.count("email_draft_new") == 1     # no retry for an effect


# --- timeout -----------------------------------------------------------------------


def test_timeout_pauses_with_stable_error(env):
    env.fake.set("email_draft_new", PAUSE("p1"))
    tid, _ = _plan(env, [
        {"intent": "email_draft_new", "arguments": {"to": ["ana@x.com"], "body": "hi"}},
        {"intent": "browser_read", "arguments": {}, "dependencies": [0]},
    ])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")                    # started_at = now, pause
    env.svc._executor.on_effect_complete({"status": "placed"}, "p1")
    env.clock["t"] = env.clock["t"] + timedelta(minutes=21)  # blow the runtime budget
    out = env.svc.resume(tid, env.settings, "es")
    assert out["error_code"] == errors.TASK_TIMEOUT and _status(env, tid) == "failed"


# --- concurrency -------------------------------------------------------------------


def test_only_one_task_runs_effects_at_a_time(env):
    env.fake.set("email_draft_new", PAUSE("p1"))
    a, _ = _plan(env, [{"intent": "email_draft_new", "arguments": {"to": ["a@x.com"], "body": "h"}}])
    env.svc.approve(a, "aprobar plan", env.settings, "es")
    env.svc.run(a, env.settings, "es")                      # A now awaiting (holds the slot)
    b, _ = _plan(env, [{"intent": "memory_search", "arguments": {"query": "x"}}])
    env.svc.approve(b, "aprobar plan", env.settings, "es")
    out = env.svc.run(b, env.settings, "es")
    assert out["error_code"] == errors.TASK_BUSY
    assert _status(env, b) == "approved"                    # B did not start


# --- cancellation ------------------------------------------------------------------


def test_cancel_stops_future_but_keeps_done(env):
    env.fake.set("email_draft_new", PAUSE("p1"))
    tid, _ = _plan(env, [
        {"intent": "browser_read", "arguments": {}},                      # will complete
        {"intent": "email_draft_new", "arguments": {"to": ["a@x.com"], "body": "h"},
         "dependencies": [0]},                                            # will pause
    ])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    env.svc.cancel(tid, env.settings, "es")
    task = env.svc.get(tid, env.settings)
    assert task["status"] == "cancelled"
    assert task["steps"][0]["status"] == "completed"        # a done step is NOT undone
    assert task["steps"][1]["status"] == "cancelled"        # the pending effect is stopped


# --- idempotency / duplicate-effect prevention -------------------------------------


def test_completed_effect_never_dispatched_twice(env):
    env.fake.set("email_draft_new", PAUSE("p1"))
    tid, _ = _plan(env, [{"intent": "email_draft_new",
                          "arguments": {"to": ["a@x.com"], "body": "h"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    env.svc._executor.on_effect_complete({"status": "sent"}, "p1")        # step done
    env.svc.resume(tid, env.settings, "es")                               # completes task
    # Re-driving a completed task never re-dispatches the effect.
    env.svc.run(tid, env.settings, "es")
    env.svc.resume(tid, env.settings, "es")
    assert env.fake.calls.count("email_draft_new") == 1


# --- restart inspection ------------------------------------------------------------


def test_restart_inspection_without_autoresume(env, monkeypatch):
    env.fake.set("email_draft_new", PAUSE("p1"))
    tid, _ = _plan(env, [{"intent": "email_draft_new",
                          "arguments": {"to": ["a@x.com"], "body": "h"}}])
    env.svc.approve(tid, "aprobar plan", env.settings, "es")
    env.svc.run(tid, env.settings, "es")
    assert _status(env, tid) == "awaiting_confirmation"
    # "Restart": a brand-new service instance over the same DB inspects the task,
    # and nothing has auto-resumed.
    fresh = TaskService(db_path=env.tmp / "t.db")
    reloaded = fresh.get(tid, env.settings)
    assert reloaded["status"] == "awaiting_confirmation"
    assert reloaded["steps"][0]["status"] == "awaiting_confirmation"
    assert any(t["task_id"] == tid for t in fresh.list(env.settings))
