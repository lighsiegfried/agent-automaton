"""Tray Activity Center controls (Phase 5D): open the localhost UI, and a status
summary built from the SAME overview the web UI renders (consistent state)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_tray  # noqa: E402
import tray_state as ts  # noqa: E402


class RecordingNotifier:
    def __init__(self):
        self.sent = []

    def notify(self, title, message, key=None):
        self.sent.append({"title": title, "message": message, "key": key})
        return True


class FakeActivityApi:
    def __init__(self, overview=None, url="http://127.0.0.1:8000/activity"):
        self._overview = overview
        self._url = url

    def url(self):
        return self._url

    def overview(self):
        return self._overview


def _overview(**over):
    d = {"fifi": "ready", "interaction_mode": "wake", "active_voice": "fifi_warm",
         "pending": {"domain": "text"}, "active_task": {"title": "research", "progress": "1/3"},
         "next_schedule": {"title": "daily summary", "next_run_at": "2026-07-20T15:00:00+00:00"},
         "recent_errors": 2, "services": {}, "resources": {"ram": None, "vram": None}}
    d.update(over)
    return d


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, activity_api=api)


def test_activity_menu_items_present_and_enabled():
    keys = {k for k, _ in ts.MENU_ITEMS}
    assert {ts.ACTIVITY_OPEN, ts.ACTIVITY_STATUS} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    assert enabled[ts.ACTIVITY_OPEN] and enabled[ts.ACTIVITY_STATUS]


def test_open_activity_center_opens_localhost_url():
    api = FakeActivityApi()
    ctrl = make_controller(api)
    opened = {}
    url = ctrl.open_activity_center(browser=lambda u: opened.setdefault("u", u))
    assert url.endswith("/activity") and opened["u"] == url
    assert "127.0.0.1" in url


def test_activity_status_summarizes_overview():
    ctrl = make_controller(FakeActivityApi(_overview()))
    ctrl.activity_status()
    msg = ctrl.notifier.sent[-1]["message"]
    assert "1 pending" in msg and "1/3" in msg and "daily summary" in msg and "2 issue" in msg


def test_activity_status_no_pending_no_errors():
    ctrl = make_controller(FakeActivityApi(_overview(pending=None, active_task=None,
                                                     next_schedule=None, recent_errors=0)))
    ctrl.activity_status()
    msg = ctrl.notifier.sent[-1]["message"]
    assert "0 pending" in msg and "next: none" in msg and "0 issue" in msg


def test_activity_status_unavailable_is_graceful():
    ctrl = make_controller(FakeActivityApi(None))
    assert ctrl.activity_status() is None
    assert "unavailable" in ctrl.notifier.sent[-1]["message"].lower()
