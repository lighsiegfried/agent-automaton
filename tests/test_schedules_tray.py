"""Tray schedule controls (Phase 5C): upcoming, create-reminder guidance, pause/
resume, run now, pending scheduled effect, recent results — via a fake /schedules
API. The tray never confirms a scheduled effect itself; it only surfaces the phrase."""

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


class FakeScheduleApi:
    def __init__(self, schedules=None, pending=None):
        self._schedules = schedules or []
        self._pending = pending
        self.paused, self.resumed, self.ran = [], [], []

    def list(self):
        return self._schedules

    def pending(self):
        return self._pending

    def pause(self, sid):
        self.paused.append(sid)
        return {"status": "ok"}

    def resume(self, sid):
        self.resumed.append(sid)
        return {"status": "ok"}

    def run_now(self, sid):
        self.ran.append(sid)
        return {"status": "ok"}


def _sched(status="active", **over):
    d = {"schedule_id": "sch_1", "title": "novedades", "status": status,
         "next_run_at": "2026-07-20T15:00:00+00:00", "last_run_at": "2026-07-15T12:00:00+00:00",
         "last_result": "completed (2/2 steps)"}
    d.update(over)
    return d


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, schedule_api=api)


def test_schedule_menu_items_present_and_enabled():
    keys = {k for k, _ in ts.MENU_ITEMS}
    assert {ts.SCHEDULE_UPCOMING, ts.SCHEDULE_CREATE, ts.SCHEDULE_PAUSE, ts.SCHEDULE_RESUME,
            ts.SCHEDULE_RUN_NOW, ts.SCHEDULE_PENDING, ts.SCHEDULE_RESULTS} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    assert enabled[ts.SCHEDULE_UPCOMING] and enabled[ts.SCHEDULE_RUN_NOW]


def test_upcoming_lists_schedules():
    ctrl = make_controller(FakeScheduleApi([_sched()]))
    ctrl.schedule_upcoming()
    assert "novedades" in ctrl.notifier.sent[-1]["message"]


def test_upcoming_none():
    ctrl = make_controller(FakeScheduleApi([]))
    assert ctrl.schedule_upcoming() == []


def test_create_reminder_guides_to_voice():
    ctrl = make_controller(FakeScheduleApi())
    ctrl.create_reminder()
    assert "recuérdame" in ctrl.notifier.sent[-1]["message"]


def test_pause_and_run_now():
    api = FakeScheduleApi([_sched()])
    ctrl = make_controller(api)
    assert ctrl.pause_schedule() is True and api.paused == ["sch_1"]
    ctrl.run_schedule_now()
    assert api.ran == ["sch_1"]


def test_resume_finds_paused():
    api = FakeScheduleApi([_sched(status="paused")])
    ctrl = make_controller(api)
    assert ctrl.resume_schedule() is True and api.resumed == ["sch_1"]


def test_pending_scheduled_effect_surfaces_phrase():
    api = FakeScheduleApi(pending={"domain": "email_send",
                                   "required_confirmation_phrase": "confirmar envío a ana"})
    ctrl = make_controller(api)
    ctrl.pending_scheduled_effect()
    assert "confirmar envío a ana" in ctrl.notifier.sent[-1]["message"]


def test_pending_scheduled_effect_none():
    ctrl = make_controller(FakeScheduleApi(pending=None))
    assert ctrl.pending_scheduled_effect() is None


def test_recent_results():
    ctrl = make_controller(FakeScheduleApi([_sched()]))
    results = ctrl.schedule_recent_results()
    assert results and "completed" in ctrl.notifier.sent[-1]["message"]
