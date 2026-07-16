"""/security/* endpoints (Phase 6B) over HTTP.

Confirms the router is mounted, the disabled-by-default path is transparent, an enabled
session locks/unlocks/elevates correctly, and NO response body ever leaks a salt, hash,
or plaintext credential.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core import pending
from app.main import app
from app.security import service as sec_mod
from app.security.service import SecurityService

client = TestClient(app)


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A throwaway security service wired to the singleton the API resolves."""
    s = get_settings()
    monkeypatch.setattr(s, "security_owner", "local")
    monkeypatch.setattr(s, "security_default_profile", "standard")
    monkeypatch.setattr(s, "security_lock_on_start", True)
    monkeypatch.setattr(s, "security_max_failed_unlocks", 5)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    svc = SecurityService(db_path=tmp_path / "sec.db",
                          settings_provider=lambda: s, hello_probe=lambda: False)
    monkeypatch.setattr(sec_mod, "_service", svc)
    return s


def test_status_transparent_when_disabled(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", False)
    r = client.get("/security/status")
    assert r.status_code == 200
    assert r.json()["data"]["enabled"] is False


def test_status_locked_when_enabled(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    body = client.get("/security/status").json()["data"]
    assert body["enabled"] is True and body["locked"] is True and body["profile"] == "locked"


def test_unlock_flow_over_http(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    assert client.post("/security/password", json={"password": "hunter2!"}).json()["status"] == "ok"
    r = client.post("/security/unlock", json={"method": "password", "secret": "hunter2!"}).json()
    assert r["status"] == "ok" and r["data"]["profile"] == "standard"
    prof = client.get("/security/profile").json()["data"]
    assert prof["base_profile"] == "standard"


def test_permissions_matrix_endpoint(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    client.post("/security/password", json={"password": "pw"})
    client.post("/security/unlock", json={"method": "password", "secret": "pw"})
    body = client.get("/security/permissions").json()["data"]
    assert "permissions" in body and "memory_search" in body["permissions"]
    assert body["permissions"]["memory_search"]["allowed"] is True


def test_elevation_over_http(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    monkeypatch.setattr(fresh, "enable_real_email_send", True, raising=False)
    client.post("/security/password", json={"password": "pw"})
    client.post("/security/unlock", json={"method": "password", "secret": "pw"})
    r = client.post("/security/elevation/prepare", json={"capability": "email"}).json()
    assert r["status"] == "ok"
    client.post("/security/elevation/revoke")


def test_no_response_ever_leaks_secret_material(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    client.post("/security/password", json={"password": "S3cr3t-Passphrase!"})
    client.post("/security/unlock", json={"method": "password", "secret": "S3cr3t-Passphrase!"})
    for path in ("/security/status", "/security/profile", "/security/permissions",
                 "/security/audit"):
        blob = json.dumps(client.get(path).json())
        assert "S3cr3t-Passphrase!" not in blob, path
        assert '"salt"' not in blob and '"hash"' not in blob, path
        assert "pbkdf2" not in blob or "created_at" in blob, path   # only safe verifier meta


def test_lock_endpoint(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "enable_security_profiles", True)
    client.post("/security/password", json={"password": "pw"})
    client.post("/security/unlock", json={"method": "password", "secret": "pw"})
    assert client.post("/security/lock").json()["status"] == "ok"
    assert client.get("/security/status").json()["data"]["locked"] is True
