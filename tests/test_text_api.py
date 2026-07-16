"""/text/* + /windows/active-target endpoints (Phase 4A).

The foreground-window probe is stubbed to a Notepad target so the flow is
deterministic. Real input stays disabled (default), so confirm simulates.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.text import service as text_service
from app.text import targets

client = TestClient(app)


def _notepad_probe():
    return {
        "title": "Untitled - Notepad", "process_name": "notepad.exe",
        "exe_path": r"C:\Windows\System32\notepad.exe", "control_type": "Edit",
        "is_password": False, "is_elevated": False, "editable": True, "error": None,
    }


@pytest.fixture(autouse=True)
def fresh_service_and_probe(monkeypatch):
    text_service._service = None
    monkeypatch.setattr(targets, "_default_probe", _notepad_probe)
    yield
    text_service._service = None


def test_active_target_is_safe_summary():
    r = client.get("/windows/active-target")
    assert r.status_code == 200
    body = r.json()
    assert body["application"] == "notepad"
    assert body["executable"] == "notepad.exe"      # basename only
    assert "System32" not in body["executable"]     # no filesystem path
    assert "exe_path" not in body                     # raw path never exposed
    assert body["blocked"] is False


def test_draft_pending_confirm_flow_simulates():
    drafted = client.post("/text/draft", json={"action": "type_text", "text": "Hola mundo"}).json()
    assert drafted["status"] == "needs_confirmation"
    action_id = drafted["preview"]["action_id"]
    assert "confirmar escritura" in drafted["confirmation_phrases"]

    pending = client.get("/text/pending").json()
    assert pending["pending"]["action_id"] == action_id

    confirmed = client.post("/text/confirm",
                            json={"action_id": action_id, "phrase": "insert text"}).json()
    assert confirmed["status"] == "simulated" and confirmed["inserted"] is False

    # No duplicate: the pending is consumed.
    again = client.post("/text/confirm",
                        json={"action_id": action_id, "phrase": "insert text"}).json()
    assert again["status"] == "rejected"


def test_plain_si_does_not_confirm():
    drafted = client.post("/text/draft", json={"action": "type_text", "text": "Hola"}).json()
    action_id = drafted["preview"]["action_id"]
    r = client.post("/text/confirm", json={"action_id": action_id, "phrase": "sí"}).json()
    assert r["status"] == "needs_confirmation"


def test_cancel_clears_pending():
    client.post("/text/draft", json={"action": "type_text", "text": "Hola"})
    assert client.post("/text/cancel").json()["status"] == "cancelled"
    assert client.get("/text/pending").json()["pending"] is None


def test_draft_into_password_field_is_rejected(monkeypatch):
    monkeypatch.setattr(targets, "_default_probe", lambda: {
        "title": "Login", "process_name": "chrome.exe", "exe_path": "chrome.exe",
        "control_type": "Edit", "is_password": True, "is_elevated": False,
        "editable": True, "error": None,
    })
    r = client.post("/text/draft", json={"action": "type_text", "text": "hunter2"}).json()
    assert r["status"] == "rejected" and "password" in r["reason"]


def test_draft_secret_text_is_rejected():
    r = client.post("/text/draft",
                    json={"action": "type_text", "text": "my card 4111 1111 1111 1111"}).json()
    assert r["status"] == "rejected"
