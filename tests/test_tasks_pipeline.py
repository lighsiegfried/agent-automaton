"""End-to-end tasks (Phase 5B): NL planning, the /tasks API, real domain
confirmations through the shared broker + effect listener, plan-approval-cannot-send,
bounded memory context, restart inspection, and cancellation."""

import itertools
from datetime import datetime, timezone
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
from app.memory import service as memory_service
from app.memory.service import MemoryService
from app.schemas.commands import CommandRequest, Intent
from app.tasks import service as task_service
from app.tasks.service import TaskService

client = TestClient(app)


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


def _browser_raw():
    return {"url": "https://example.com", "title": "Playwright docs",
            "text": "Playwright is a browser automation library.",
            "headings": [{"level": 2, "text": "Docs"}], "links": [], "buttons": [],
            "form_controls": [], "landmarks": ["main"]}


class FakeSession:
    def __init__(self, raw):
        self.raw = raw
        self.navs = 0
        self.reads = 0

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.navs += 1; self.raw["url"] = url
    def back(self): pass
    def forward(self): pass
    def scroll(self, dy): pass
    def snapshot_raw(self): self.reads += 1; return self.raw
    def fill(self, n, v): pass
    def close(self): pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_multi_step_tasks", True)
    monkeypatch.setattr(s, "task_owner", "local")
    monkeypatch.setattr(s, "tasks_db_path", tmp_path / "t.db")
    monkeypatch.setattr(s, "enable_email_web_automation", True)
    monkeypatch.setattr(s, "enable_real_email_send", False)
    monkeypatch.setattr(s, "enable_browser_automation", True)
    monkeypatch.setattr(s, "enable_memory", True)
    monkeypatch.setattr(s, "memory_owner", "local")
    monkeypatch.setattr(s, "memory_db_path", tmp_path / "m.db")
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())

    msvc = MemoryService(db_path=tmp_path / "m.db",
                         now_fn=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                         id_factory=lambda: "mem_" + uuid4())
    monkeypatch.setattr(memory_service, "_service", msvc)

    adapter = FakeEmailAdapter()
    eids = (f"em{i}" for i in itertools.count(1))
    esvc = EmailService(adapter_factory=lambda p, ss: adapter, clock=lambda: 1000.0,
                        now_utc=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                        id_factory=lambda: next(eids))
    monkeypatch.setattr(email_service, "_service", esvc)

    session = FakeSession(_browser_raw())
    bs = BrowserService(session_factory=lambda ss: session, id_factory=lambda: "form1")
    monkeypatch.setattr(browser_service, "_service", bs)

    tids = (str(i) for i in itertools.count(1))
    tsvc = TaskService(db_path=tmp_path / "t.db",
                       now_fn=lambda: datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                       id_factory=lambda: next(tids))
    monkeypatch.setattr(task_service, "_service", tsvc)
    return SimpleNamespace(settings=s, adapter=adapter, session=session, msvc=msvc,
                           tsvc=tsvc, tmp=tmp_path)


_uuid_counter = itertools.count(1)


def uuid4():
    return str(next(_uuid_counter))


def _cmd(text, language="es"):
    return handle_command(CommandRequest(text=text, language=language))


RESEARCH_PLAN = [
    {"intent": "browser_search", "arguments": {"query": "Playwright"}},
    {"intent": "browser_summarize", "arguments": {}, "dependencies": [0]},
    {"intent": "email_draft_new",
     "arguments": {"to": ["ana@example.com"], "subject": "Resumen", "body": "Aquí el resumen"},
     "dependencies": [1]},
    {"intent": "email_prepare_send", "arguments": {}, "dependencies": [2]},
]


# --- natural-language planning -----------------------------------------------------


def test_nl_plan_routing(env):
    resp = _cmd("Fifi, prepara un plan para investigar Playwright")
    assert resp.intent == Intent.TASK_PLAN
    # The read-only research heuristic produced at least a browser step.
    tasks = env.tsvc.list(env.settings)
    assert tasks and tasks[0]["status"] == "planned"


def test_nl_approve_and_cancel_route(env):
    env.tsvc.plan({"steps": [{"intent": "memory_search", "arguments": {"query": "x"}}]},
                  "es", env.settings)
    assert _cmd("aprobar plan").intent == Intent.TASK_APPROVE
    assert _cmd("cancelar tarea").intent == Intent.TASK_CANCEL


# --- full API flow with REAL domain confirmations ----------------------------------


def _plan_via_api():
    return client.post("/tasks/plan", json={"request": "research and email", "steps": RESEARCH_PLAN}).json()


def test_full_flow_reads_auto_then_effects_pause_for_domain_phrase(env):
    plan = _plan_via_api()
    tid = plan["data"]["task"]["task_id"]
    assert [st["classification"] for st in plan["data"]["plan"]] == \
        ["read_only", "read_only", "local", "effect"]

    assert client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"}).json()["status"] == "ok"
    assert env.adapter.sent == 0 and env.adapter.compose["to"] == []   # approval sent nothing

    run = client.post(f"/tasks/{tid}/run").json()
    assert run["status"] == "needs_confirmation"                       # paused at the draft
    task = client.get(f"/tasks/{tid}").json()["task"]
    assert [s["status"] for s in task["steps"]][:2] == ["completed", "completed"]  # reads ran
    assert task["status"] == "awaiting_confirmation" and env.adapter.sent == 0

    # A plain "sí" cannot confirm the effect — it isn't even a confirmation, so the
    # task stays paused and nothing is placed.
    _cmd("sí")
    assert client.get(f"/tasks/{tid}").json()["task"]["status"] == "awaiting_confirmation"
    assert env.adapter.compose["to"] == []

    # The real place confirmation (voice) advances the task via the effect listener.
    assert _cmd("colocar borrador").result["status"] in ("placed", "executed", "filled")
    assert env.adapter.compose["to"] == ["ana@example.com"] and env.adapter.sent == 0
    assert client.get(f"/tasks/{tid}").json()["task"]["status"] == "paused"

    resume = client.post(f"/tasks/{tid}/resume", json={}).json()
    assert resume["status"] == "needs_confirmation"                    # now paused at send
    assert "envío" in resume["required_phrase"].lower()

    # Sending needs its own recipient-specific phrase; a wrong one won't send.
    assert env.adapter.sent == 0
    _cmd("confirmar envío a ana")
    assert client.get(f"/tasks/{tid}").json()["task"]["status"] == "paused"
    final = client.post(f"/tasks/{tid}/resume", json={}).json()
    assert final["data"]["task"]["status"] == "completed"
    assert env.adapter.sent == 0                                       # simulated by default


def test_plan_approval_never_sends(env):
    plan = _plan_via_api()
    tid = plan["data"]["task"]["task_id"]
    client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"})
    # Approval, even repeated, never places or sends.
    client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"})
    assert env.adapter.sent == 0 and env.adapter.compose["to"] == []


def test_resume_with_phrase_confirms_via_api(env):
    plan = _plan_via_api()
    tid = plan["data"]["task"]["task_id"]
    client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"})
    client.post(f"/tasks/{tid}/run")
    # /resume can carry the domain phrase to confirm + advance in one call.
    client.post(f"/tasks/{tid}/resume", json={"phrase": "colocar borrador"})
    assert env.adapter.compose["to"] == ["ana@example.com"]
    r = client.post(f"/tasks/{tid}/resume", json={"phrase": "confirmar envío a ana"}).json()
    assert r["data"]["task"]["status"] == "completed" and env.adapter.sent == 0


# --- bounded memory context --------------------------------------------------------


def test_bounded_memory_context(env, monkeypatch):
    monkeypatch.setattr(env.settings, "task_context_max_memories", 2)
    # Store several memories the request will match.
    from app.memory.models import Memory, content_hash
    for i in range(5):
        env.msvc.repo.insert(Memory(
            id=f"mem_stored_{i}", owner="local", type="project",
            title=f"Playwright note {i}", content=f"Playwright detail number {i}",
            tags=["playwright"], content_hash=content_hash("local", "project", f"d{i}")))
    plan = client.post("/tasks/plan", json={"request": "resumir Playwright",
                                            "steps": RESEARCH_PLAN[:2]}).json()
    refs = plan["data"]["task"]["memory_refs"]
    assert 0 < len(refs) <= 2                       # bounded to TASK_CONTEXT_MAX_MEMORIES


# --- restart inspection + resume without repeating completed steps -----------------


def test_restart_inspection_and_resume(env, monkeypatch):
    # Plan through the first effect (draft placement) only.
    steps = RESEARCH_PLAN[:3]
    tid = client.post("/tasks/plan", json={"request": "r", "steps": steps}).json()["data"]["task"]["task_id"]
    client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"})
    client.post(f"/tasks/{tid}/run")
    assert client.get(f"/tasks/{tid}").json()["task"]["status"] == "awaiting_confirmation"
    reads_before = env.session.reads

    # "Restart": fresh broker + fresh domain/task services (in-memory state lost,
    # the task DB persists).
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    fresh_email = EmailService(adapter_factory=lambda p, ss: env.adapter, clock=lambda: 2000.0,
                               now_utc=lambda: datetime(2026, 7, 15, tzinfo=timezone.utc),
                               id_factory=lambda: "em_new")
    monkeypatch.setattr(email_service, "_service", fresh_email)
    # A consistent clock 5 minutes on (still within the 20-minute runtime budget).
    fresh_tasks = TaskService(db_path=env.tmp / "t.db",
                              now_fn=lambda: datetime(2026, 7, 15, 12, 5, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(task_service, "_service", fresh_tasks)

    # Inspect recovery: the paused task and its completed reads survive.
    reloaded = fresh_tasks.get(tid, env.settings)
    assert reloaded["status"] == "awaiting_confirmation"
    assert [s["status"] for s in reloaded["steps"]][:2] == ["completed", "completed"]

    # Resume: the lost pending is re-prepared (draft re-created), completed reads
    # are NOT re-run, then the confirmation finishes the step.
    fresh_tasks.resume(tid, env.settings, "es")
    assert env.session.reads == reads_before        # research/summary never repeat
    fresh_tasks.resume(tid, env.settings, "es", phrase="colocar borrador")
    assert env.adapter.compose["to"] == ["ana@example.com"]
    assert fresh_tasks.get(tid, env.settings)["status"] == "completed"


# --- cancellation midway -----------------------------------------------------------


def test_cancel_midway(env):
    tid = _plan_via_api()["data"]["task"]["task_id"]
    client.post(f"/tasks/{tid}/approve", json={"phrase": "aprobar plan"})
    client.post(f"/tasks/{tid}/run")               # pauses at the draft
    assert client.post(f"/tasks/{tid}/cancel").json()["data"]["task"]["status"] == "cancelled"
    assert env.adapter.sent == 0
    # A cancelled task can't be resumed.
    assert client.post(f"/tasks/{tid}/resume", json={}).json()["status"] == "rejected"
