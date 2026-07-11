import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "real_windows_tools" in data


def test_sensitive_command_needs_then_accepts_confirmation(client, settings):
    settings.require_confirmation = True
    settings.enable_real_windows_tools = False

    first = client.post("/command", json={"text": "open app notepad"}).json()
    assert first["status"] == "needs_confirmation"

    second = client.post(
        "/command", json={"text": "open app notepad", "confirm": True}
    ).json()
    assert second["status"] == "simulated"
    assert second["result"]["simulated"] is True


def test_disallowed_app_is_rejected(client, settings):
    response = client.post(
        "/command", json={"text": "open app photoshop", "confirm": True}
    ).json()
    assert response["status"] == "rejected"


def test_unknown_command_is_not_handled(client):
    response = client.post("/command", json={"text": "hazme un sandwich"}).json()
    assert response["status"] == "not_handled"
