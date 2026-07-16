"""/whatsapp/* endpoints (Phase 4C) — fake session, simulated send."""

import itertools

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core import pending
from app.integrations.whatsapp import service as wa_service
from app.integrations.whatsapp.service import WhatsAppService
from app.main import app

client = TestClient(app)


class FakeSession:
    def __init__(self):
        self._chat = "Ana"
        self.composer = ""
        self.sent = 0

    def logged_in(self): return True
    def search_contacts(self, name): return ["Ana", "Beto"]
    def open_chat(self, name): self._chat = name; return True
    def chat_title(self): return self._chat
    def composer_text(self): return self.composer
    def set_composer(self, text): self.composer = text
    def click_send(self): self.sent += 1
    def close(self): pass


@pytest.fixture(autouse=True)
def env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_whatsapp_automation", True)
    monkeypatch.setattr(settings, "enable_real_whatsapp_send", False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    session = FakeSession()
    ids = (f"wa{i}" for i in itertools.count(1))
    monkeypatch.setattr(wa_service, "_service",
                        WhatsAppService(session_provider=lambda s: session, id_factory=lambda: next(ids)))
    return session


def test_status_logged_in():
    body = client.get("/whatsapp/status").json()
    assert body["status"] == "ok" and body["data"]["logged_in"] is True


def test_find_contact_exact():
    body = client.post("/whatsapp/contact/find", json={"name": "Ana"}).json()
    assert body["data"]["contact"] == "Ana"


def test_draft_place_send_flow_simulated(env):
    draft = client.post("/whatsapp/draft", json={"text": "Hola Ana"}).json()
    assert draft["status"] == "needs_confirmation"
    placed = client.post("/whatsapp/draft/place", json={"phrase": "colocar borrador"}).json()
    assert placed["status"] == "placed"
    assert env.composer == "Hola Ana" and env.sent == 0
    sent = client.post("/whatsapp/send/confirm", json={"phrase": "confirmar envío a Ana"}).json()
    assert sent["status"] == "simulated" and env.sent == 0


def test_send_confirm_wrong_recipient_rejected(env):
    client.post("/whatsapp/draft", json={"text": "Hola"})
    client.post("/whatsapp/draft/place", json={"phrase": "colocar borrador"})
    body = client.post("/whatsapp/send/confirm", json={"phrase": "confirmar envío a Beto"}).json()
    assert body["error_code"] == "CONFIRMATION_MISMATCH"
    assert env.sent == 0


def test_disabled_reports_feature_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "enable_whatsapp_automation", False)
    assert client.get("/whatsapp/status").json()["error_code"] == "FEATURE_DISABLED"
