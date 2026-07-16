"""ActivityService (Phase 5D): ingestion, cross-domain aggregation, control routing
through existing services (no generic confirm), safe restart/unload, degraded
sections, retention, and redacted export."""

import itertools
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.activity import events as activity_events
from app.activity import service as activity_service
from app.activity.service import ActivityService
from app.config import get_settings
from app.core import eventbus, errors, pending

UTC = timezone.utc


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_activity_center", True)
    monkeypatch.setattr(s, "activity_owner", "local")
    monkeypatch.setattr(s, "activity_db_path", tmp_path / "a.db")
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": datetime(2026, 7, 15, 12, 0, tzinfo=UTC)}
    asvc = ActivityService(db_path=tmp_path / "a.db", now_fn=lambda: clock["t"],
                           settings_provider=lambda: s)
    monkeypatch.setattr(activity_service, "_service", asvc)
    activity_events.install()   # idempotent
    return SimpleNamespace(settings=s, svc=asvc, clock=clock, tmp=tmp_path)


# --- ingestion ---------------------------------------------------------------------


def test_ingest_normalizes_and_flattens_task(env):
    eventbus.emit(domain="tasks", event_type="task_plan", status="planned",
                  metadata={"task": {"task_id": "task_1", "title": "research", "status": "planned",
                                     "current_step": 0, "steps": [{}, {}]}})
    ev = env.svc.events(page=1)["events"][0]
    full = env.svc.event(ev["event_id"])
    assert ev["domain"] == "tasks"
    assert full["metadata"]["step_count"] == 2 and full["metadata"]["title"] == "research"


def test_ingest_respects_disabled_flag(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_activity_center", False)
    # The sink checks the flag; ingest directly is also a no-op guard via the sink,
    # but calling ingest() still writes — the flag is enforced at the bus sink.
    from app.activity.events import _sink
    _sink({"domain": "memory", "event_type": "x"})
    assert env.svc.events(page=1)["total"] == 0


def test_aggregation_across_domains(env):
    for dom in ("memory", "browser", "email", "tasks", "schedules"):
        eventbus.emit(domain=dom, event_type=f"{dom}_evt", dedup_key=f"{dom}-1")
    got = {e["domain"] for e in env.svc.events(page=1, page_size=50)["events"]}
    assert {"memory", "browser", "email", "tasks", "schedules"} <= got


# --- pending confirm routing (no generic confirm) ----------------------------------


def _with_memory(env, monkeypatch):
    from app.memory import service as memory_service
    from app.memory.service import MemoryService
    monkeypatch.setattr(env.settings, "enable_memory", True)
    monkeypatch.setattr(env.settings, "memory_owner", "local")
    monkeypatch.setattr(env.settings, "memory_db_path", env.tmp / "m.db")
    ids = (f"mem_{i}" for i in itertools.count(1))
    svc = MemoryService(db_path=env.tmp / "m.db",
                        now_fn=lambda: datetime(2026, 7, 15, tzinfo=UTC), id_factory=lambda: next(ids))
    monkeypatch.setattr(memory_service, "_service", svc)
    return svc


def _propose(intent="memory_propose", **args):
    from app.core import conversation
    from app.core.nl import ServiceCommand
    return conversation.dispatch(ServiceCommand(intent=intent, arguments=args), language="es")


def test_no_generic_confirm(env, monkeypatch):
    _with_memory(env, monkeypatch)
    _propose(content="el proyecto usa FastAPI", source="voice")
    # A blank confirm is refused (there is no "confirm everything").
    assert env.svc.confirm_action("")["error_code"] == errors.CONFIRMATION_REQUIRED
    # A wrong phrase is refused by the normal broker path.
    assert env.svc.confirm_action("sí")["error_code"] == errors.CONFIRMATION_MISMATCH


def test_confirm_routes_through_domain_phrase(env, monkeypatch):
    memsvc = _with_memory(env, monkeypatch)
    _propose(content="el proyecto usa FastAPI", source="voice")
    pend = env.svc.pending()["pending"]
    assert pend["domain"] == "memory_write"
    result = env.svc.confirm_action(pend["confirm_phrase"])
    assert result["status"] == "executed"
    assert memsvc.repo.counts("local")["active"] == 1


def test_cancel_action_clears_broker(env, monkeypatch):
    _with_memory(env, monkeypatch)
    _propose(content="algo", source="voice")
    assert env.svc.cancel_action()["had_pending"] is True
    assert pending.get_pending_broker().summary() is None


# --- task / schedule controls (through existing services) --------------------------


def test_task_control_cancel(env, monkeypatch):
    from app.tasks import service as task_service
    from app.tasks.service import TaskService
    monkeypatch.setattr(env.settings, "enable_multi_step_tasks", True)
    monkeypatch.setattr(env.settings, "task_owner", "local")
    monkeypatch.setattr(env.settings, "tasks_db_path", env.tmp / "t.db")
    tids = (str(i) for i in itertools.count(1))
    tsvc = TaskService(db_path=env.tmp / "t.db",
                       now_fn=lambda: datetime(2026, 7, 15, 12, tzinfo=UTC), id_factory=lambda: next(tids))
    monkeypatch.setattr(task_service, "_service", tsvc)
    tid = tsvc.plan({"steps": [{"intent": "memory_search", "arguments": {"query": "x"}}]},
                    "es", env.settings)["data"]["task"]["task_id"]
    assert env.svc.task_control(tid, "cancel")["status"] == "cancelled"


def test_schedule_control_pause(env, monkeypatch):
    from app.schedules import service as sched_service
    from app.schedules.service import ScheduleService
    from app.core import conversation
    from app.core.nl import ServiceCommand
    monkeypatch.setattr(env.settings, "enable_schedules", True)
    monkeypatch.setattr(env.settings, "enable_multi_step_tasks", True)
    monkeypatch.setattr(env.settings, "schedule_owner", "local")
    monkeypatch.setattr(env.settings, "schedule_timezone", "America/Guatemala")
    monkeypatch.setattr(env.settings, "schedules_db_path", env.tmp / "s.db")
    sids = (str(i) for i in itertools.count(1))
    ssvc = ScheduleService(db_path=env.tmp / "s.db",
                           now_fn=lambda: datetime(2026, 7, 15, 12, tzinfo=UTC), id_factory=lambda: next(sids))
    monkeypatch.setattr(sched_service, "_service", ssvc)
    ssvc.prepare({"text": "cada día a las 9", "kind": "task",
                  "steps": [{"intent": "browser_read", "arguments": {}}]}, "es",
                 env.settings, pending.get_pending_broker())
    sid = conversation.dispatch(ServiceCommand(intent="confirm", is_confirmation=True,
                                               confirmation_phrase="confirmar programación"),
                                language="es")["data"]["schedule"]["schedule_id"]
    assert env.svc.schedule_control(sid, "pause")["status"] == "ok"


# --- safe runtime recovery ---------------------------------------------------------


def test_restart_rejects_unknown_service(env):
    assert env.svc.restart_service("rm -rf /")["error_code"] == errors.UNSUPPORTED_SERVICE
    # An OWNED service is delegated to the tray/runtime, never killed here.
    assert env.svc.restart_service("ollama")["status"] == "delegated"


def test_unload_never_raises(env):
    result = env.svc.unload()      # Ollama down in tests → unavailable, no exception
    assert "status" in result


# --- degraded sections + reliability -----------------------------------------------


def test_disabled_sections_are_graceful(env):
    assert env.svc.tasks().get("status") == "disabled"
    assert env.svc.schedules().get("status") == "disabled"
    svcs = env.svc.services()["services"]
    assert svcs["ollama"]["status"] in ("unavailable", "ok")   # never raises
    assert env.svc.overview()["fifi"] in ("ready", "degraded")


# --- export + retention ------------------------------------------------------------


def test_export_is_redacted(env):
    eventbus.emit(domain="text", event_type="draft_text",
                  title="draft with sk-LIVE1234567890ABCDEFghij",
                  metadata={"content": "mi contraseña es hunter2", "count": 1})
    exp = env.svc.export()
    blob = str(exp["events"])
    assert exp["redacted"] is True
    assert "hunter2" not in blob and "sk-LIVE" not in blob and "contraseña es" not in blob


def test_export_can_be_disabled(env, monkeypatch):
    monkeypatch.setattr(env.settings, "activity_enable_export", False)
    assert env.svc.export()["error_code"] == errors.EXPORT_DISABLED


def test_retention_prune_runs(env):
    for i in range(5):
        eventbus.emit(domain="runtime", event_type="probe", dedup_key=f"old-{i}")
    assert env.svc.prune_now() >= 0     # incremental, never raises
