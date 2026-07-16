"""End-to-end schedules (Phase 5C): NL routing, the /schedules API, the scheduler
firing reminders, a recurring read-only task run, a scheduled email that prepares
but never sends until a fresh recipient confirmation, and restart no-double-fire."""

import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.browser import service as browser_service
from app.browser.service import BrowserService
from app.config import get_settings
from app.core import pending
from app.core.router import handle_command
from app.integrations.email_web import service as email_service
from app.integrations.email_web.service import EmailService
from app.main import app
from app.schedules import service as sched_service
from app.schedules.service import ScheduleService
from app.schemas.commands import CommandRequest, Intent
from app.tasks import service as task_service
from app.tasks.service import TaskService

client = TestClient(app)
UTC = timezone.utc


class FakeEmailAdapter:
    def __init__(self):
        self._logged_in = True
        self.compose = {"to": [], "cc": [], "bcc": [], "subject": "", "body": "",
                        "attachment_count": 0, "mode": "new"}
        self.sent = 0

    def provider(self): return "gmail"
    def logged_in(self): return self._logged_in
    def search(self, q): return []
    def start_new(self): self.compose["mode"] = "new"
    def set_compose(self, to, subject, body, cc=None):
        self.compose.update(to=list(to), subject=subject, body=body, cc=list(cc or []))
    def compose_state(self): return dict(self.compose)
    def click_send(self): self.sent += 1
    def close(self): pass


class FakeSession:
    def __init__(self):
        self.raw = {"url": "https://ex.com", "title": "Novedades", "text": "News.",
                    "headings": [{"level": 2, "text": "News"}], "links": [], "buttons": [],
                    "form_controls": [], "landmarks": ["main"]}
        self.reads = 0

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.raw["url"] = url
    def scroll(self, dy): pass
    def snapshot_raw(self): self.reads += 1; return self.raw
    def close(self): pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    for flag in ("enable_schedules", "enable_multi_step_tasks", "enable_browser_automation",
                 "enable_email_web_automation"):
        monkeypatch.setattr(s, flag, True)
    monkeypatch.setattr(s, "enable_real_email_send", False)
    monkeypatch.setattr(s, "enable_memory", False)
    monkeypatch.setattr(s, "schedule_owner", "local")
    monkeypatch.setattr(s, "task_owner", "local")
    monkeypatch.setattr(s, "schedule_timezone", "America/Guatemala")
    monkeypatch.setattr(s, "schedules_db_path", tmp_path / "s.db")
    monkeypatch.setattr(s, "tasks_db_path", tmp_path / "t.db")
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": datetime(2026, 7, 15, 12, 0, tzinfo=UTC)}

    adapter = FakeEmailAdapter()
    eids = (f"em{i}" for i in itertools.count(1))
    monkeypatch.setattr(email_service, "_service",
                        EmailService(adapter_factory=lambda p, ss: adapter, clock=lambda: 1000.0,
                                     now_utc=lambda: datetime(2026, 7, 15, tzinfo=UTC),
                                     id_factory=lambda: next(eids)))
    monkeypatch.setattr(browser_service, "_service",
                        BrowserService(session_factory=lambda ss: FakeSession(), id_factory=lambda: "f1"))
    tids = (str(i) for i in itertools.count(1))
    tsvc = TaskService(db_path=tmp_path / "t.db", now_fn=lambda: clock["t"], id_factory=lambda: next(tids))
    monkeypatch.setattr(task_service, "_service", tsvc)
    notes = []
    sids = (str(i) for i in itertools.count(1))
    svc = ScheduleService(db_path=tmp_path / "s.db", now_fn=lambda: clock["t"],
                          id_factory=lambda: next(sids), task_service_provider=lambda: tsvc,
                          notifier=lambda t, m, **k: notes.append((t, m)))
    monkeypatch.setattr(sched_service, "_service", svc)
    return SimpleNamespace(settings=s, svc=svc, adapter=adapter, clock=clock, notes=notes, tmp=tmp_path)


def _cmd(text, language="es"):
    return handle_command(CommandRequest(text=text, language=language))


# --- NL routing --------------------------------------------------------------------


def test_nl_routing(env):
    assert _cmd("Fifi, recuérdame llamar a Ana mañana a las nueve").intent == Intent.SCHEDULE_PREPARE
    assert _cmd("cada lunes busca novedades del proyecto").intent == Intent.SCHEDULE_PREPARE
    assert _cmd("¿qué tengo programado?").intent == Intent.SCHEDULE_LIST


# --- reminder fires once -----------------------------------------------------------


def test_reminder_fires_once_via_scheduler(env):
    prep = client.post("/schedules/prepare",
                       json={"text": "llamar a Ana en 1 minuto", "kind": "reminder"}).json()
    assert prep["status"] == "needs_confirmation"
    sid = client.post("/schedules/confirm", json={"phrase": "confirmar programación"}
                      ).json()["data"]["schedule"]["schedule_id"]
    env.clock["t"] = env.clock["t"] + timedelta(minutes=2)
    env.svc.scheduler.poll_once()
    assert len(env.notes) == 1
    env.svc.scheduler.poll_once()                                  # no second fire
    assert len(env.notes) == 1
    runs = client.get(f"/schedules/{sid}/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["status"] == "delivered"


# --- recurring read-only task ------------------------------------------------------


def test_recurring_readonly_task_runs(env):
    client.post("/schedules/prepare", json={
        "text": "cada día a las 9", "kind": "task", "title": "novedades",
        "steps": [{"intent": "browser_search", "arguments": {"query": "novedades"}},
                  {"intent": "browser_summarize", "arguments": {}, "dependencies": [0]}]})
    sid = client.post("/schedules/confirm", json={"phrase": "confirmar programación"}
                      ).json()["data"]["schedule"]["schedule_id"]
    run = client.post(f"/schedules/{sid}/run").json()
    detail = client.get(f"/schedules/{sid}").json()["schedule"]
    assert detail["runs"][0]["status"] == "completed"
    assert detail["status"] == "active"                            # recurring stays active


# --- scheduled email prepares but never sends --------------------------------------


EMAIL_STEPS = [
    {"intent": "browser_search", "arguments": {"query": "novedades"}},
    {"intent": "email_draft_new",
     "arguments": {"to": ["ana@example.com"], "subject": "Resumen", "body": "El resumen."},
     "dependencies": [0]},
    {"intent": "email_prepare_send", "arguments": {}, "dependencies": [1]},
]


def test_scheduled_email_prepares_not_sends_then_fresh_confirmation(env):
    client.post("/schedules/prepare",
                json={"text": "cada día a las 9", "kind": "task", "title": "resumen", "steps": EMAIL_STEPS})
    sid = client.post("/schedules/confirm", json={"phrase": "confirmar programación"}
                      ).json()["data"]["schedule"]["schedule_id"]
    # Scheduling + confirming the schedule never sends anything.
    assert env.adapter.sent == 0 and env.adapter.compose["to"] == []

    client.post(f"/schedules/{sid}/run")
    run = client.get(f"/schedules/{sid}/runs").json()["runs"][0]
    assert run["status"] == "paused"                               # draft prepared, not sent
    assert env.adapter.sent == 0

    tid = run["task_id"]
    # A fresh domain confirmation is required to place, then to send.
    _cmd("colocar borrador")
    assert env.adapter.compose["to"] == ["ana@example.com"] and env.adapter.sent == 0
    task_service.get_task_service().resume(tid, env.settings, "es")   # advance to send
    _cmd("confirmar envío a ana")
    task_service.get_task_service().resume(tid, env.settings, "es")
    assert task_service.get_task_service().get(tid, env.settings)["status"] == "completed"
    assert env.adapter.sent == 0                                    # simulated by default (one effect)


# --- restart no double-fire --------------------------------------------------------


def test_restart_no_double_fire(env, monkeypatch):
    client.post("/schedules/prepare", json={"text": "nota en 1 minuto", "kind": "reminder"})
    sid = client.post("/schedules/confirm", json={"phrase": "confirmar programación"}
                      ).json()["data"]["schedule"]["schedule_id"]
    env.clock["t"] = env.clock["t"] + timedelta(minutes=2)
    env.svc.scheduler.poll_once()
    assert len(env.notes) == 1
    # "Restart": fresh service + scheduler over the same on-disk DB.
    fresh = ScheduleService(db_path=env.tmp / "s.db", now_fn=lambda: env.clock["t"],
                            notifier=lambda t, m, **k: env.notes.append((t, m)))
    monkeypatch.setattr(sched_service, "_service", fresh)
    fresh.scheduler.poll_once()
    assert len(env.notes) == 1                                      # still exactly one
    assert client.get(f"/schedules/{sid}").json()["schedule"]["status"] == "completed"
