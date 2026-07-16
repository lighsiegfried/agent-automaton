"""Tray task controls (Phase 5B): current task, view plan, approve, resume, cancel,
pending confirmation, audit summary — driven by a fake /tasks API. The tray never
confirms an effect on its own; resume only surfaces the exact domain phrase."""

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


class FakeTaskApi:
    def __init__(self, task=None):
        self._task = task
        self.approved: list = []
        self.resumed: list = []
        self.cancelled: list = []

    def list(self):
        return [self._task] if self._task else []

    def current(self):
        return self._task

    def approve(self, task_id):
        self.approved.append(task_id)
        return {"status": "ok"}

    def resume(self, task_id, phrase=""):
        self.resumed.append((task_id, phrase))
        return {"status": "needs_confirmation", "required_phrase": "confirmar envío a ana",
                "data": {"task": {"status": "awaiting_confirmation"}}}

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        return {"status": "cancelled"}

    def audit(self, task_id):
        return {"events": [{"event": "plan_created"}, {"event": "step_completed"}]}


def _task(status="awaiting_confirmation"):
    return {
        "task_id": "task_1", "title": "research and email", "status": status,
        "current_step": 2, "steps": [
            {"position": 0, "intent": "browser_search", "risk_level": "read_only",
             "status": "completed", "confirmation_phrase": ""},
            {"position": 1, "intent": "email_prepare_send", "risk_level": "effect",
             "status": "awaiting_confirmation",
             "confirmation_phrase": "confirmar envío a ana"},
        ],
    }


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, task_api=api)


# --- menu model --------------------------------------------------------------------


def test_task_menu_items_present_and_enabled():
    keys = {k for k, _ in ts.MENU_ITEMS}
    assert {ts.TASK_CURRENT, ts.TASK_PLAN_VIEW, ts.TASK_APPROVE, ts.TASK_RESUME,
            ts.TASK_CANCEL, ts.TASK_PENDING, ts.TASK_AUDIT} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    assert enabled[ts.TASK_CURRENT] and enabled[ts.TASK_APPROVE] and enabled[ts.TASK_AUDIT]


# --- controller behaviour ----------------------------------------------------------


def test_current_task_reports_status():
    ctrl = make_controller(FakeTaskApi(_task()))
    ctrl.task_current()
    assert "awaiting_confirmation" in ctrl.notifier.sent[-1]["message"]


def test_view_plan_lists_steps():
    ctrl = make_controller(FakeTaskApi(_task()))
    ctrl.view_plan()
    msg = ctrl.notifier.sent[-1]["message"]
    assert "browser_search" in msg and "email_prepare_send" in msg


def test_approve_uses_canonical_phrase():
    api = FakeTaskApi(_task("planned"))
    ctrl = make_controller(api)
    assert ctrl.approve_task() is True
    assert api.approved == ["task_1"]


def test_resume_surfaces_required_phrase_not_confirmation():
    api = FakeTaskApi(_task())
    ctrl = make_controller(api)
    ctrl.resume_task()
    # The tray asked to resume WITHOUT a phrase — it never confirms an effect itself.
    assert api.resumed == [("task_1", "")]
    assert "confirmar envío a ana" in ctrl.notifier.sent[-1]["message"]


def test_cancel_task():
    api = FakeTaskApi(_task())
    ctrl = make_controller(api)
    assert ctrl.cancel_task() is True and api.cancelled == ["task_1"]


def test_pending_confirmation_shows_step_phrase():
    ctrl = make_controller(FakeTaskApi(_task()))
    step = ctrl.task_pending_confirmation()
    assert step["intent"] == "email_prepare_send"
    assert "confirmar envío a ana" in ctrl.notifier.sent[-1]["message"]


def test_pending_confirmation_none_when_not_awaiting():
    ctrl = make_controller(FakeTaskApi(_task("running")))
    assert ctrl.task_pending_confirmation() is None


def test_audit_summary():
    ctrl = make_controller(FakeTaskApi(_task()))
    events = ctrl.task_audit_summary()
    assert [e["event"] for e in events] == ["plan_created", "step_completed"]


def test_no_task_is_graceful():
    ctrl = make_controller(FakeTaskApi(None))
    assert ctrl.task_current() is None
    assert ctrl.approve_task() is False
    assert ctrl.cancel_task() is False
