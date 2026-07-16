"""Deterministic dispatcher + unified pending broker end-to-end (Phase 4B.1).

Services are the real TextActionService / BrowserService with injected fakes
(notepad target probe, fake browser session, fake inserter) so the whole
command→service path is exercised without a desktop or a browser.
"""

import itertools
import types

import pytest

from app.browser import service as browser_service
from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
from app.text import service as text_service


def _notepad_probe():
    return {"title": "Untitled - Notepad", "process_name": "notepad.exe",
            "exe_path": r"C:\Windows\notepad.exe", "control_type": "Edit",
            "is_password": False, "is_elevated": False, "editable": True, "error": None}


def _browser_raw():
    return {
        "url": "https://example.com/form", "title": "Sign up",
        "text": "Create your account. Pricing details here.",
        "headings": [{"level": 1, "text": "Sign up"}, {"level": 2, "text": "Pricing"}],
        "links": [{"text": "Docs", "href": "https://example.com/docs"}],
        "buttons": [{"text": "Submit", "role": "button"}],
        "form_controls": [{"name": "full_name", "type": "text", "label": "Full name"},
                          {"name": "password", "type": "password", "label": "Password"}],
        "landmarks": ["main"],
    }


class FakeSession:
    def __init__(self, raw):
        self.raw, self.filled = raw, []

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.raw["url"] = url
    def back(self): pass
    def forward(self): pass
    def scroll(self, dy): pass
    def snapshot_raw(self): return self.raw
    def fill(self, name, value): self.filled.append((name, value))
    def close(self): pass


@pytest.fixture
def env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_browser_automation", True)
    monkeypatch.setattr(settings, "enable_real_text_input", False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())

    clock = {"t": 1000.0}
    inserted = []
    ids = (f"text{i}" for i in itertools.count(1))
    ts = text_service.TextActionService(
        clock=lambda: clock["t"], target_probe=_notepad_probe, id_factory=lambda: next(ids),
        inserter=lambda p, s: inserted.append(p) or {"ok": True, "method": "uia",
                                                     "chars": len(p.text), "pressed_enter": False},
    )
    monkeypatch.setattr(text_service, "_service", ts)

    session = FakeSession(_browser_raw())
    bs = browser_service.BrowserService(session_factory=lambda s: session,
                                        clock=lambda: clock["t"], id_factory=lambda: "form1")
    monkeypatch.setattr(browser_service, "_service", bs)

    return types.SimpleNamespace(settings=settings, clock=clock, inserted=inserted,
                                 session=session, broker=pending.get_pending_broker())


def cmd(intent, **args):
    return ServiceCommand(intent=intent, arguments=args)


def confirm(phrase):
    return ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase)


# --- text drafting + confirmation --------------------------------------------------


def test_draft_text_creates_pending_without_inserting(env):
    result = conversation.dispatch(cmd("draft_text", instruction="un correo para Ana"), language="es")
    assert result["status"] == "needs_confirmation"
    assert result["state"] == errors.AWAITING_CONFIRMATION
    assert result["domain"] == "text"
    assert env.inserted == []
    assert env.broker.summary()["domain"] == "text"
    # Spoken line is safe: mentions count + app, never the draft text.
    assert "un correo para Ana" not in result["spoken"]


def test_text_confirmation_simulated_by_default(env):
    aid = conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")["action_id"]
    result = conversation.dispatch(confirm("confirmar escritura"), language="es")
    assert result["status"] == "simulated" and result["state"] == errors.COMPLETED
    assert env.inserted == []
    assert env.broker.summary() is None       # consumed


def test_text_confirmation_executes_when_enabled(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_real_text_input", True)
    conversation.dispatch(cmd("type_text", text="Hola mundo"), language="es")
    result = conversation.dispatch(confirm("insert text"), language="en")
    assert result["status"] == "executed"
    assert len(env.inserted) == 1
    assert env.inserted[0].text == "Hola mundo"


def test_duplicate_confirmation_prevented(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_real_text_input", True)
    conversation.dispatch(cmd("type_text", text="Hola"), language="es")
    conversation.dispatch(confirm("confirmar escritura"), language="es")
    again = conversation.dispatch(confirm("confirmar escritura"), language="es")
    assert again["error_code"] == errors.CONFIRMATION_REQUIRED   # nothing pending
    assert len(env.inserted) == 1


def test_secret_draft_blocked_and_not_spoken(env):
    result = conversation.dispatch(cmd("type_text", text="card 4111 1111 1111 1111"), language="es")
    assert result["status"] == "rejected"
    assert result["error_code"] == errors.SENSITIVE_FIELD_BLOCKED
    assert "4111" not in result["spoken"]


# --- cross-domain confirmation routing ---------------------------------------------


def test_cross_domain_confirmation_rejected(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")   # text pending
    result = conversation.dispatch(confirm("confirmar formulario"), language="es")  # browser phrase
    assert result["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.inserted == []                 # text not inserted


def test_plain_si_never_confirms(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")
    result = conversation.dispatch(confirm("sí"), language="es")
    assert result["error_code"] == errors.CONFIRMATION_MISMATCH


def test_wake_utterance_never_confirms(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")
    result = conversation.dispatch(confirm("Fifi"), language="es", wake=True)
    assert result["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.inserted == []


# --- one pending across domains (supersede) ----------------------------------------


def test_new_pending_supersedes_previous_domain(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")   # text pending
    conversation.dispatch(cmd("browser_prepare_form", fields={"full_name": "Ada"}), language="es")
    assert env.broker.summary()["domain"] == "browser"     # browser superseded text
    # The stale text confirmation is now a mismatch, and does nothing.
    result = conversation.dispatch(confirm("confirmar escritura"), language="es")
    assert result["error_code"] == errors.CONFIRMATION_MISMATCH


# --- browser reading / navigation --------------------------------------------------


def test_browser_search(env):
    result = conversation.dispatch(cmd("browser_search", query="playwright docs"), language="es")
    assert result["status"] == "ok" and result["state"] == errors.COMPLETED
    assert result["data"]["title"] == "Sign up"


def test_browser_summarize(env):
    result = conversation.dispatch(cmd("browser_summarize"), language="es")
    assert result["status"] == "ok"
    assert result["data"]["heading_count"] == 2


def test_browser_find(env):
    result = conversation.dispatch(cmd("browser_find", query="pricing"), language="es")
    assert result["status"] == "ok"
    assert result["data"]["count"] >= 1


def test_browser_disabled_reports_feature_disabled(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_browser_automation", False)
    result = conversation.dispatch(cmd("browser_search", query="x"), language="es")
    assert result["error_code"] == errors.FEATURE_DISABLED


# --- browser form prepare + confirm ------------------------------------------------


def test_browser_form_prepare_and_confirm(env):
    prepared = conversation.dispatch(
        cmd("browser_prepare_form", fields={"full_name": "Ada", "password": "x"}), language="es")
    assert prepared["status"] == "needs_confirmation"
    assert prepared["data"]["skipped_sensitive"] == 1     # password skipped
    result = conversation.dispatch(confirm("confirmar formulario"), language="es")
    assert result["status"] == "filled"
    assert env.session.filled == [("full_name", "Ada")]   # filled, never submitted


def test_browser_form_confirm_rejected_after_page_change(env):
    conversation.dispatch(cmd("browser_prepare_form", fields={"full_name": "Ada"}), language="es")
    env.session.raw["title"] = "Different page"
    result = conversation.dispatch(confirm("confirmar formulario"), language="es")
    assert result["error_code"] == errors.TARGET_CHANGED
    assert env.session.filled == []


# --- expiry + cancel ---------------------------------------------------------------


def test_expired_text_action_reports_action_expired(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")
    env.clock["t"] += 100      # past text_action_expires_seconds
    result = conversation.dispatch(confirm("confirmar escritura"), language="es")
    assert result["error_code"] == errors.ACTION_EXPIRED


def test_cancel_clears_pending(env):
    conversation.dispatch(cmd("draft_text", instruction="hola"), language="es")
    result = conversation.dispatch(cmd("cancel"), language="es")
    assert result["status"] == "cancelled"
    assert env.broker.summary() is None
