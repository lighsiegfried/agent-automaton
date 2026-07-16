"""Tray memory controls (Phase 5A): status/pending/confirm/cancel/search/viewer.

The tray is UI-agnostic here — a fake /memory API stands in for the running server,
so the controller behaviour is fully testable without pystray or a live API. A menu
click IS the explicit confirmation for a pending memory; wake never reaches it."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_tray  # noqa: E402
import tray_state as ts  # noqa: E402


class RecordingNotifier:
    def __init__(self):
        self.sent: list = []

    def notify(self, title, message, key=None):
        self.sent.append({"title": title, "message": message, "key": key})
        return True


class FakeMemoryApi:
    """Stands in for the /memory + /pending endpoints."""

    def __init__(self, *, pending=None, status=None, memories=None):
        self._pending = pending
        self._status = status or {"enabled": True, "owner": "local",
                                  "counts": {"active": 2, "forgotten": 1}}
        self._memories = memories or [
            {"type": "preference", "title": "Editor", "content": "VS Code", "date": "2026-07-15"}]
        self.confirmed: list = []
        self.cancelled = 0

    def status(self):
        return self._status

    def pending(self):
        return self._pending

    def confirm(self, phrase):
        self.confirmed.append(phrase)
        return {"status": "executed"}

    def cancel(self):
        self.cancelled += 1
        return {"status": "cancelled"}

    def search(self, query):
        return {"count": len(self._memories), "results": self._memories}

    def list(self):
        return {"count": len(self._memories), "memories": self._memories}


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, memory_api=api)


# --- menu model --------------------------------------------------------------------


def test_memory_menu_items_present_and_enabled():
    keys = {k for k, _label in ts.MENU_ITEMS}
    assert {ts.MEMORY_STATUS, ts.MEMORY_PENDING, ts.MEMORY_CONFIRM,
            ts.MEMORY_CANCEL, ts.MEMORY_SEARCH, ts.MEMORY_VIEWER} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    # Memory controls are API-backed and available even when the stack is stopped.
    assert enabled[ts.MEMORY_STATUS] and enabled[ts.MEMORY_CONFIRM] and enabled[ts.MEMORY_VIEWER]


# --- controller behaviour ----------------------------------------------------------


def test_memory_status_reports_counts():
    api = FakeMemoryApi()
    ctrl = make_controller(api)
    ctrl.memory_status()
    msg = ctrl.notifier.sent[-1]["message"]
    assert "2 active" in msg and "1 forgotten" in msg


def test_view_pending_memory_none():
    ctrl = make_controller(FakeMemoryApi(pending=None))
    assert ctrl.view_pending_memory() is None


def test_confirm_pending_memory_uses_exact_phrase():
    api = FakeMemoryApi(pending={"domain": "memory_write",
                                 "required_confirmation_phrase": "confirmar memoria / confirm memory"})
    ctrl = make_controller(api)
    assert ctrl.confirm_memory() is True
    assert api.confirmed == ["confirmar memoria"]     # the EXACT domain phrase, never a bare yes


def test_confirm_pending_forget_uses_forget_phrase():
    api = FakeMemoryApi(pending={"domain": "memory_forget"})
    ctrl = make_controller(api)
    ctrl.confirm_memory()
    assert api.confirmed == ["confirmar olvido"]


def test_confirm_permanent_delete_phrase():
    api = FakeMemoryApi(pending={"domain": "memory_delete"})
    ctrl = make_controller(api)
    ctrl.confirm_memory()
    assert api.confirmed == ["eliminar memoria permanentemente"]


def test_confirm_with_nothing_pending_is_noop():
    api = FakeMemoryApi(pending=None)
    ctrl = make_controller(api)
    assert ctrl.confirm_memory() is False
    assert api.confirmed == []


def test_cancel_memory():
    api = FakeMemoryApi()
    ctrl = make_controller(api)
    assert ctrl.cancel_memory() is True and api.cancelled == 1


def test_search_memory_notifies_count():
    ctrl = make_controller(FakeMemoryApi())
    ctrl.search_memory("editor")
    assert "1 match" in ctrl.notifier.sent[-1]["message"]


# --- read-only viewer --------------------------------------------------------------


def test_memory_viewer_html_lists_memories_without_secrets():
    html = fifi_tray.render_memory_html(
        {"memories": [{"type": "preference", "title": "Editor", "content": "VS Code",
                       "date": "2026-07-15"}]})
    assert "Editor" in html and "VS Code" in html
    assert "Read-only" in html


def test_open_memory_viewer_serves_page():
    api = FakeMemoryApi()
    ctrl = make_controller(api)
    opened = {}

    class FakeServer:
        def __init__(self, html): self.html = html
        def start(self): return "http://127.0.0.1:0"
        def set_html(self, html): self.html = html

    url = ctrl.open_memory_viewer(server_factory=FakeServer, browser=lambda u: opened.setdefault("u", u))
    assert url == "http://127.0.0.1:0" and opened["u"] == url
