"""End-to-end Activity Center (Phase 5D): the localhost UI + API, events flowing in
once from real dispatch across domains, a pending confirmed through the UI path,
restart persistence, and no sensitive content in the timeline or export."""

import itertools
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.activity import events as activity_events
from app.activity import service as activity_service
from app.activity.api import ACTIVITY_HTML, is_loopback
from app.activity.service import ActivityService
from app.browser import service as browser_service
from app.browser.service import BrowserService
from app.config import get_settings
from app.core import pending
from app.core.router import handle_command
from app.main import app
from app.memory import service as memory_service
from app.memory.service import MemoryService
from app.schemas.commands import CommandRequest
from app.text import service as text_service

client = TestClient(app)
UTC = timezone.utc


def _notepad():
    return {"title": "Untitled - Notepad", "process_name": "notepad.exe",
            "exe_path": r"C:\Windows\notepad.exe", "control_type": "Edit",
            "is_password": False, "is_elevated": False, "editable": True, "error": None}


class FakeSession:
    def __init__(self):
        self.raw = {"url": "https://ex.com", "title": "Example", "text": "Pricing.",
                    "headings": [{"level": 2, "text": "Pricing"}], "links": [], "buttons": [],
                    "form_controls": [], "landmarks": ["main"]}

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.raw["url"] = url
    def scroll(self, dy): pass
    def snapshot_raw(self): return self.raw
    def fill(self, n, v): pass
    def close(self): pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_activity_center", True)
    monkeypatch.setattr(s, "activity_owner", "local")
    monkeypatch.setattr(s, "activity_db_path", tmp_path / "a.db")
    monkeypatch.setattr(s, "enable_memory", True)
    monkeypatch.setattr(s, "memory_owner", "local")
    monkeypatch.setattr(s, "memory_db_path", tmp_path / "m.db")
    monkeypatch.setattr(s, "enable_browser_automation", True)
    monkeypatch.setattr(s, "enable_real_text_input", False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    mids = (f"mem_{i}" for i in itertools.count(1))
    monkeypatch.setattr(memory_service, "_service",
                        MemoryService(db_path=tmp_path / "m.db",
                                      now_fn=lambda: datetime(2026, 7, 15, tzinfo=UTC),
                                      id_factory=lambda: next(mids)))
    tids = (f"t{i}" for i in itertools.count(1))
    monkeypatch.setattr(text_service, "_service",
                        text_service.TextActionService(target_probe=_notepad,
                                                       id_factory=lambda: next(tids),
                                                       inserter=lambda p, s2: {"ok": True, "method": "uia"}))
    monkeypatch.setattr(browser_service, "_service",
                        BrowserService(session_factory=lambda ss: FakeSession(), id_factory=lambda: "f1"))
    asvc = ActivityService(db_path=tmp_path / "a.db",
                           now_fn=lambda: datetime(2026, 7, 15, 12, tzinfo=UTC), settings_provider=lambda: s)
    monkeypatch.setattr(activity_service, "_service", asvc)
    activity_events.install()
    return SimpleNamespace(settings=s, svc=asvc, tmp=tmp_path)


def _cmd(text, language="es"):
    return handle_command(CommandRequest(text=text, language=language))


# --- UI + localhost ----------------------------------------------------------------


def test_ui_served_without_external_resources(env):
    html = client.get("/activity").text
    assert "<title>Fifi Activity Center" in html
    # No external CDNs / remote resources (only same-origin fetches).
    assert "cdn" not in html.lower()
    assert "http://" not in ACTIVITY_HTML and "https://" not in ACTIVITY_HTML


def test_localhost_binding():
    assert is_loopback(get_settings().activity_ui_host) is True
    assert is_loopback("0.0.0.0") is False and is_loopback("8.8.8.8") is False


# --- ingestion across domains, once each -------------------------------------------


def test_events_flow_in_once_across_domains(env):
    _cmd("¿qué recuerdas de proyecto?")             # memory
    _cmd("abre el navegador y busca documentación")  # browser
    events = client.get("/activity/events?page=1&page_size=50").json()["events"]
    domains = [e["domain"] for e in events]
    assert "memory" in domains and "browser" in domains
    # A single search produced exactly one memory event (no duplication from hooks).
    assert sum(1 for e in events if e["domain"] == "memory") == 1


def test_pending_and_confirm_through_ui(env):
    _cmd("redacta un saludo")                        # text draft → pending
    p = client.get("/activity/pending").json()["pending"]
    assert p and p["domain"] == "text" and p["confirm_phrase"]
    # Confirm through the Activity Center action endpoint (routes via the broker).
    res = client.post("/activity/actions/confirm", json={"phrase": p["confirm_phrase"]}).json()
    assert res["status"] in ("executed", "simulated")
    assert client.get("/activity/pending").json()["pending"] is None


def test_cancel_through_ui(env):
    _cmd("redacta un saludo")
    assert client.post("/activity/actions/cancel").json()["had_pending"] is True
    assert client.get("/activity/pending").json()["pending"] is None


def test_overview_consistent_for_tray_and_ui(env):
    _cmd("redacta un saludo")
    ov = client.get("/activity/overview").json()
    # Both the tray summary and the UI read this same overview.
    assert ov["pending"]["domain"] == "text"
    assert set(ov) >= {"fifi", "services", "recent_errors", "pending", "active_task", "next_schedule"}


# --- restart persistence -----------------------------------------------------------


def test_history_persists_across_restart(env, monkeypatch):
    _cmd("¿qué recuerdas de x?")
    before = client.get("/activity/events").json()["total"]
    assert before >= 1
    fresh = ActivityService(db_path=env.tmp / "a.db",
                            now_fn=lambda: datetime(2026, 7, 15, 12, 30, tzinfo=UTC),
                            settings_provider=lambda: env.settings)
    monkeypatch.setattr(activity_service, "_service", fresh)
    assert client.get("/activity/events").json()["total"] == before   # same on-disk history


# --- redaction end-to-end ----------------------------------------------------------


def test_no_sensitive_content_in_timeline_or_export(env):
    # A draft whose instruction contains a secret — the secret must not surface.
    _cmd("redacta que mi contraseña es hunter2 y el token sk-LIVE1234567890ABCDEFghij")
    timeline = str(client.get("/activity/events?page=1&page_size=50").json())
    export = str(client.get("/activity/export").json())
    for blob in (timeline, export):
        assert "hunter2" not in blob and "sk-LIVE" not in blob


def test_disabled_returns_guard(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_activity_center", False)
    assert client.get("/activity/overview").json().get("error_code") == "ACTIVITY_DISABLED"
