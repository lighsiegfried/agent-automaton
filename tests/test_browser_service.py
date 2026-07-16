"""BrowserService orchestration with a fake session (Phase 4B). No real browser."""

import itertools
import types

import pytest

from app.browser.service import BrowserService


def make_settings(**over):
    base = dict(
        enable_browser_automation=True,
        browser_engine="chromium", browser_headless=True, browser_profile_mode="isolated",
        browser_allowed_schemes_list=["https", "http"],
        browser_block_private_networks=True,
        browser_max_page_text_chars=50000,
        browser_action_timeout_seconds=30,
        browser_session_idle_seconds=600,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def make_raw(**over):
    raw = {
        "url": "https://example.com/form", "title": "Sign up",
        "text": "Create your account. Pricing details below.",
        "headings": [{"level": 1, "text": "Sign up"}, {"level": 2, "text": "Pricing"}],
        "links": [{"text": "Docs", "href": "https://example.com/docs"}],
        "buttons": [{"text": "Submit", "role": "button"}],
        "form_controls": [
            {"name": "full_name", "type": "text", "label": "Full name"},
            {"name": "email", "type": "email", "label": "Email"},
            {"name": "password", "type": "password", "label": "Password"},
        ],
        "landmarks": ["main"],
    }
    raw.update(over)
    return raw


class FakeSession:
    def __init__(self, raw):
        self.raw = raw
        self.goto_urls, self.scrolls, self.filled = [], [], []
        self.closed = False
        self.last_activity = 0.0

    def current_url(self):
        return self.raw["url"]

    def title(self):
        return self.raw["title"]

    def goto(self, url):
        self.goto_urls.append(url)
        self.raw["url"] = url

    def back(self):
        self.goto_urls.append("<back>")

    def forward(self):
        self.goto_urls.append("<forward>")

    def scroll(self, dy):
        self.scrolls.append(dy)

    def snapshot_raw(self):
        return self.raw

    def fill(self, name, value):
        self.filled.append((name, value))

    def close(self):
        self.closed = True


@pytest.fixture
def clock():
    return {"t": 1000.0}


@pytest.fixture
def service(clock):
    ids = (f"form{i}" for i in itertools.count(1))
    session = FakeSession(make_raw())
    svc = BrowserService(
        session_factory=lambda settings: session,
        clock=lambda: clock["t"],
        id_factory=lambda: next(ids),
    )
    svc.fake = session
    return svc


# --- session lifecycle -------------------------------------------------------------


def test_disabled_by_default():
    svc = BrowserService(session_factory=lambda s: None)
    assert svc.start_session(make_settings(enable_browser_automation=False))["status"] == "disabled"


def test_start_and_status(service):
    s = make_settings()
    assert service.start_session(s)["status"] == "ok"
    st = service.status(s)
    assert st["active"] is True and st["url"] == "https://example.com/form"


def test_actions_require_a_session(service):
    r = service.navigate("https://example.com", make_settings())
    assert r["status"] == "error" and "no active" in r["reason"]


# --- navigation + URL policy -------------------------------------------------------


def test_navigate_ok(service):
    s = make_settings()
    service.start_session(s)
    r = service.navigate("example.com/docs", s)
    assert r["status"] == "ok"
    assert service.fake.goto_urls[-1] == "https://example.com/docs"
    assert "page" in r and r["page"]["title"] == "Sign up"


def test_navigate_blocks_unsafe_scheme(service):
    s = make_settings()
    service.start_session(s)
    assert service.navigate("javascript:alert(1)", s)["status"] == "rejected"


def test_navigate_blocks_private(service):
    s = make_settings()
    service.start_session(s)
    assert service.navigate("http://localhost:8000", s)["status"] == "rejected"


def test_search_builds_query_url(service):
    s = make_settings()
    service.start_session(s)
    service.search("playwright testing", s)
    assert "duckduckgo.com/?q=playwright" in service.fake.goto_urls[-1]


def test_find_and_scroll(service):
    s = make_settings()
    service.start_session(s)
    assert service.find("pricing", s)["found"] is True
    service.scroll("down", s)
    assert service.fake.scrolls == [600]


def test_open_link_validates_and_navigates(service):
    s = make_settings()
    service.start_session(s)
    r = service.open_link("Docs", s)
    assert r["status"] == "ok"
    assert service.fake.goto_urls[-1] == "https://example.com/docs"


# --- form preparation + confirmation ----------------------------------------------


def test_prepare_form_skips_sensitive_fields(service):
    s = make_settings()
    service.start_session(s)
    r = service.prepare_form({"full_name": "Ada", "email": "a@x.com", "password": "hunter2"}, s)
    assert r["status"] == "needs_confirmation"
    names = [f["name"] for f in r["plan"]["fields"]]
    assert names == ["full_name", "email"]
    skipped = [f["name"] for f in r["plan"]["skipped_sensitive_fields"]]
    assert skipped == ["password"]


def test_prepare_form_all_sensitive_rejected(service):
    s = make_settings()
    service.start_session(s)
    r = service.prepare_form({"password": "x"}, s)
    assert r["status"] == "rejected"


def test_confirm_requires_exact_phrase(service):
    s = make_settings()
    service.start_session(s)
    aid = service.prepare_form({"full_name": "Ada"}, s)["plan"]["action_id"]
    r = service.confirm_form(aid, "sí", s)
    assert r["status"] == "needs_confirmation"
    assert service.fake.filled == []            # nothing filled


def test_confirm_fills_without_submitting(service):
    s = make_settings()
    service.start_session(s)
    aid = service.prepare_form({"full_name": "Ada", "email": "a@x.com"}, s)["plan"]["action_id"]
    r = service.confirm_form(aid, "confirmar formulario", s)
    assert r["status"] == "filled" and r["submitted"] is False
    assert service.fake.filled == [("full_name", "Ada"), ("email", "a@x.com")]
    # No duplicate fill: the pending plan is consumed.
    again = service.confirm_form(aid, "confirmar formulario", s)
    assert again["status"] == "rejected"


def test_confirm_rejected_when_page_changed(service):
    s = make_settings()
    service.start_session(s)
    aid = service.prepare_form({"full_name": "Ada"}, s)["plan"]["action_id"]
    service.fake.raw["title"] = "Different page"      # page changed after prepare
    r = service.confirm_form(aid, "confirm form fill", s)
    assert r["status"] == "rejected" and "changed" in r["reason"]
    assert service.fake.filled == []


def test_confirm_rejected_after_expiry(service, clock):
    s = make_settings(browser_action_timeout_seconds=30)
    service.start_session(s)
    aid = service.prepare_form({"full_name": "Ada"}, s)["plan"]["action_id"]
    clock["t"] += 31
    r = service.confirm_form(aid, "confirmar formulario", s)
    assert r["status"] == "rejected" and "expired" in r["reason"]


def test_cancel_form(service):
    s = make_settings()
    service.start_session(s)
    service.prepare_form({"full_name": "Ada"}, s)
    assert service.cancel_form()["had_pending"] is True
    assert service.pending_form(s) is None


def test_close_session(service):
    s = make_settings()
    service.start_session(s)
    service.close_session(s)
    assert service.fake.closed is True
    assert service.status(s)["active"] is False
