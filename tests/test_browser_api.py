"""/browser/* endpoints (Phase 4B). Fake session; browser enabled via settings."""

import pytest
from fastapi.testclient import TestClient

from app.browser import service as bsvc
from app.config import get_settings
from app.main import app

client = TestClient(app)


def _raw():
    return {
        "url": "https://example.com/form", "title": "Sign up",
        "text": "Create your account. Pricing details below.",
        "headings": [{"level": 2, "text": "Pricing"}],
        "links": [{"text": "Docs", "href": "https://example.com/docs"}],
        "buttons": [{"text": "Submit", "role": "button"}],
        "form_controls": [
            {"name": "full_name", "type": "text", "label": "Full name"},
            {"name": "password", "type": "password", "label": "Password"},
            {"name": "csrf", "type": "hidden", "label": ""},
        ],
        "landmarks": ["main"],
    }


class FakeSession:
    def __init__(self, raw):
        self.raw, self.filled, self.goto_urls = raw, [], []
        self.last_activity = 0.0

    def current_url(self): return self.raw["url"]
    def title(self): return self.raw["title"]
    def goto(self, url): self.goto_urls.append(url); self.raw["url"] = url
    def back(self): pass
    def forward(self): pass
    def scroll(self, dy): pass
    def snapshot_raw(self): return self.raw
    def fill(self, name, value): self.filled.append((name, value))
    def close(self): pass


@pytest.fixture(autouse=True)
def browser_env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_browser_automation", True)
    fake = FakeSession(_raw())
    svc = bsvc.BrowserService(session_factory=lambda s: fake, id_factory=lambda: "form1")
    monkeypatch.setattr(bsvc, "_service", svc)
    return fake


def test_session_start_and_status():
    assert client.post("/browser/session/start").json()["status"] == "ok"
    assert client.get("/browser/session/status").json()["active"] is True


def test_navigate_and_safe_page_snapshot():
    client.post("/browser/session/start")
    nav = client.post("/browser/navigate", json={"url": "example.com/form"}).json()
    assert nav["status"] == "ok"
    page = client.get("/browser/page").json()["page"]
    # Safe snapshot: no raw HTML/cookies/storage; hidden field dropped; no values.
    assert "html" not in page and "cookies" not in page
    names = [c["name"] for c in page["form_controls"]]
    assert "csrf" not in names
    assert all("value" not in c for c in page["form_controls"])


def test_navigate_blocks_unsafe_scheme():
    client.post("/browser/session/start")
    assert client.post("/browser/navigate", json={"url": "file:///etc/passwd"}).json()["status"] == "rejected"


def test_find_and_scroll_action():
    client.post("/browser/session/start")
    assert client.post("/browser/find", json={"query": "pricing"}).json()["found"] is True
    assert client.post("/browser/action", json={"type": "scroll", "direction": "down"}).json()["status"] == "ok"


def test_form_prepare_confirm_never_submits(browser_env):
    client.post("/browser/session/start")
    prepared = client.post("/browser/form/prepare",
                           json={"fields": {"full_name": "Ada", "password": "hunter2"}}).json()
    assert prepared["status"] == "needs_confirmation"
    assert [f["name"] for f in prepared["plan"]["fields"]] == ["full_name"]   # password skipped
    action_id = prepared["plan"]["action_id"]

    confirmed = client.post("/browser/form/confirm",
                            json={"action_id": action_id, "phrase": "confirmar formulario"}).json()
    assert confirmed["status"] == "filled" and confirmed["submitted"] is False
    assert browser_env.filled == [("full_name", "Ada")]

    # No duplicate: confirming again is rejected.
    again = client.post("/browser/form/confirm",
                        json={"action_id": action_id, "phrase": "confirmar formulario"}).json()
    assert again["status"] == "rejected"


def test_form_confirm_rejects_plain_si():
    client.post("/browser/session/start")
    prepared = client.post("/browser/form/prepare", json={"fields": {"full_name": "Ada"}}).json()
    r = client.post("/browser/form/confirm",
                    json={"action_id": prepared["plan"]["action_id"], "phrase": "sí"}).json()
    assert r["status"] == "needs_confirmation"


def test_disabled_when_flag_off(monkeypatch):
    monkeypatch.setattr(get_settings(), "enable_browser_automation", False)
    assert client.post("/browser/session/start").json()["status"] == "disabled"
