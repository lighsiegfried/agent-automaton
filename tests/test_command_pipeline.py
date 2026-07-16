"""End-to-end: transcribed command -> handle_command -> service result (Phase 4B.1).

Exercises the real router with injected fake services, plus the /pending endpoint
and that legacy tool routing (the PTT/voice path) is unchanged.
"""

import itertools
import types

import pytest
from fastapi.testclient import TestClient

from app.browser import service as browser_service
from app.config import get_settings
from app.core import pending
from app.core.router import handle_command
from app.main import app
from app.schemas.commands import CommandRequest, ExecutionStatus, Intent
from app.text import service as text_service

client = TestClient(app)


def _notepad_probe():
    return {"title": "Untitled - Notepad", "process_name": "notepad.exe",
            "exe_path": r"C:\Windows\notepad.exe", "control_type": "Edit",
            "is_password": False, "is_elevated": False, "editable": True, "error": None}


def _browser_raw():
    return {"url": "https://example.com", "title": "Example", "text": "Pricing info here.",
            "headings": [{"level": 2, "text": "Pricing"}], "links": [], "buttons": [],
            "form_controls": [], "landmarks": ["main"]}


class FakeSession:
    def __init__(self, raw):
        self.raw = raw

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.raw["url"] = url
    def back(self): pass
    def forward(self): pass
    def scroll(self, dy): pass
    def snapshot_raw(self): return self.raw
    def fill(self, name, value): pass
    def close(self): pass


@pytest.fixture(autouse=True)
def env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_browser_automation", True)
    monkeypatch.setattr(settings, "enable_real_text_input", False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    ids = (f"t{i}" for i in itertools.count(1))
    ts = text_service.TextActionService(target_probe=_notepad_probe, id_factory=lambda: next(ids),
                                        inserter=lambda p, s: {"ok": True, "method": "uia"})
    monkeypatch.setattr(text_service, "_service", ts)
    bs = browser_service.BrowserService(session_factory=lambda s: FakeSession(_browser_raw()),
                                        id_factory=lambda: "form1")
    monkeypatch.setattr(browser_service, "_service", bs)
    return types.SimpleNamespace(settings=settings)


def _cmd(text, language=None):
    return handle_command(CommandRequest(text=text, language=language))


# --- Spanish + English service routing ---------------------------------------------


def test_spanish_draft_needs_confirmation():
    resp = _cmd("Fifi, redacta un correo para Ana", language="es")
    assert resp.intent == Intent.DRAFT_TEXT
    assert resp.status == ExecutionStatus.NEEDS_CONFIRMATION
    assert resp.assistant_message                     # a safe spoken line exists
    assert "un correo para Ana" not in resp.assistant_message


def test_english_draft_needs_confirmation():
    resp = _cmd("draft an email to the team", language="en")
    assert resp.intent == Intent.DRAFT_TEXT
    assert resp.status == ExecutionStatus.NEEDS_CONFIRMATION


def test_text_confirmation_flow_simulated():
    _cmd("escribe esto en la ventana: Hola", language="es")
    resp = _cmd("confirmar escritura", language="es")
    assert resp.status == ExecutionStatus.SIMULATED
    assert resp.result["domain"] == "text"


def test_cross_domain_confirmation_rejected_in_pipeline():
    _cmd("redacta un saludo", language="es")               # text pending
    resp = _cmd("confirmar formulario", language="es")     # browser phrase
    assert resp.result["error_code"] == "CONFIRMATION_MISMATCH"


def test_browser_search_executes():
    resp = _cmd("abre el navegador y busca documentación de Playwright", language="es")
    assert resp.intent == Intent.BROWSER_SEARCH
    assert resp.status == ExecutionStatus.EXECUTED


# --- legacy tool routing (PTT/voice path) is unchanged -----------------------------


def test_legacy_open_app_still_routes_to_tool():
    resp = _cmd("abre notepad", language="es")
    assert resp.intent == Intent.OPEN_APP
    assert resp.tool == "open_app"                    # legacy tool path intact


def test_legacy_search_web_still_routes_to_tool():
    resp = _cmd("busca gatos en internet", language="es")
    assert resp.intent == Intent.SEARCH_WEB
    assert resp.tool == "search_web"


# --- /pending endpoint (tray pending-state) ----------------------------------------


def test_pending_endpoint_reflects_awaiting_confirmation():
    assert client.get("/pending").json()["pending"] is None
    client.post("/command", json={"text": "redacta un correo", "language": "es"})
    pending_body = client.get("/pending").json()["pending"]
    assert pending_body["domain"] == "text"
    assert pending_body["status"] == "awaiting_confirmation"
    assert "confirmar escritura" in pending_body["required_confirmation_phrase"]


def test_pending_cancel_endpoint():
    client.post("/command", json={"text": "redacta un correo", "language": "es"})
    assert client.post("/pending/cancel").json()["had_pending"] is True
    assert client.get("/pending").json()["pending"] is None
