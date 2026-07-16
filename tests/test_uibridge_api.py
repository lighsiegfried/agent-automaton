"""Phase 6C — the /ui/* bridge endpoints over HTTP.

Flag-gating (off by default), loopback-only, the mint→token→overview handshake, and
token gating on the protected endpoints. The infinite SSE stream is only checked on its
UNAUTHORIZED path (a valid-token stream would block a TestClient by design)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core import errors
from app.main import app
from app.uibridge import api as ui_api
from app.uibridge import session as ui_session

client = TestClient(app)


@pytest.fixture
def bridge(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_desktop_bridge", True)
    ui_session.reset_session_manager()
    yield s
    ui_session.reset_session_manager()


def test_disabled_by_default(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_desktop_bridge", False)
    r = client.post("/ui/session")
    assert r.status_code == 403 and r.json()["error_code"] == errors.DESKTOP_DISABLED


def test_mint_session(bridge):
    r = client.post("/ui/session")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["token"] and data["ttl_seconds"] > 0
    # the minted token verifies against the same singleton
    assert ui_session.get_session_manager().verify(data["token"])["ok"] is True


def test_overview_requires_token(bridge):
    r = client.get("/ui/overview")
    assert r.status_code == 401 and r.json()["error_code"] == errors.UI_TOKEN_REQUIRED


def test_overview_rejects_bad_token(bridge):
    r = client.get("/ui/overview", headers={"Authorization": "Bearer not.a.token"})
    assert r.status_code == 403 and r.json()["error_code"] == errors.UI_TOKEN_INVALID


def test_overview_with_token_returns_safe_snapshot(bridge):
    token = client.post("/ui/session").json()["data"]["token"]
    r = client.get("/ui/overview", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert "pending" in data and "security" in data
    # security disabled by default => the snapshot says so, no secrets
    assert data["security"] == {"enabled": False}


def test_revoke_session(bridge):
    token = client.post("/ui/session").json()["data"]["token"]
    client.post("/ui/session/revoke", headers={"Authorization": f"Bearer {token}"})
    r = client.get("/ui/overview", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403           # revoked token no longer authorizes


def test_stream_unauthorized_is_refused_without_streaming(bridge):
    # no token => the endpoint returns a denial BEFORE opening the event stream
    r = client.get("/ui/stream")
    assert r.status_code == 401 and r.json()["error_code"] == errors.UI_TOKEN_REQUIRED


def test_non_local_caller_blocked():
    # _is_local is the loopback guard; a remote client host is refused
    remote = SimpleNamespace(client=SimpleNamespace(host="10.0.0.9"), headers={},
                             query_params={})
    assert ui_api._is_local(remote) is False
    local = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={},
                            query_params={})
    assert ui_api._is_local(local) is True
