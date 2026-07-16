"""Tray Knowledge Vault controls (Phase 6A): open, import, search, index status."""

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


class FakeKnowledgeApi:
    def __init__(self, status=None, search=None, upload=None):
        self._status = status or {"enabled": True, "counts": {"ready": 3}, "embedder": "hash-local-v1"}
        self._search = search or {"count": 2}
        self._upload = upload or {"status": "ok"}
        self.uploaded = []

    def url(self):
        return "http://127.0.0.1:8000/activity"

    def status(self):
        return self._status

    def search(self, query):
        return self._search

    def upload(self, path):
        self.uploaded.append(path)
        return self._upload


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, knowledge_api=api)


def test_knowledge_menu_items_present_and_enabled():
    keys = {k for k, _ in ts.MENU_ITEMS}
    assert {ts.KNOWLEDGE_OPEN, ts.KNOWLEDGE_IMPORT, ts.KNOWLEDGE_SEARCH, ts.KNOWLEDGE_STATUS} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    assert enabled[ts.KNOWLEDGE_OPEN] and enabled[ts.KNOWLEDGE_STATUS]


def test_open_knowledge_vault_localhost():
    ctrl = make_controller(FakeKnowledgeApi())
    opened = {}
    url = ctrl.open_knowledge_vault(browser=lambda u: opened.setdefault("u", u))
    assert "127.0.0.1" in url and opened["u"] == url


def test_index_status_summary():
    ctrl = make_controller(FakeKnowledgeApi())
    ctrl.knowledge_status()
    assert "3 indexed" in ctrl.notifier.sent[-1]["message"]


def test_search_documents_notifies_count():
    ctrl = make_controller(FakeKnowledgeApi())
    ctrl.search_documents("BOFASA")
    assert "2 passage" in ctrl.notifier.sent[-1]["message"]


def test_import_uses_explicit_selection():
    api = FakeKnowledgeApi()
    ctrl = make_controller(api)
    ctrl.import_document(chooser=lambda: "C:/docs/policy.pdf")
    assert api.uploaded == ["C:/docs/policy.pdf"]


def test_import_no_selection_is_graceful():
    ctrl = make_controller(FakeKnowledgeApi())
    assert ctrl.import_document(chooser=lambda: None) is None
