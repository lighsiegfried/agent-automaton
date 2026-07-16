"""Tray security controls (Phase 6B): status, unlock (Windows Hello), lock, request a
capability-scoped elevation, revoke. The tray never sends a password and never changes
the base profile — those stay on the localhost/desktop page and the spoken confirm flow."""

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


class FakeSecurityApi:
    def __init__(self, status=None, unlock=None, elevation=None):
        self._status = status if status is not None else {
            "enabled": True, "locked": False, "profile": "standard", "elevation": None}
        self._unlock = unlock if unlock is not None else {"status": "ok", "data": {"profile": "standard"}}
        self._elevation = elevation if elevation is not None else {
            "status": "ok", "data": {"elevation": {"capability": "email", "expires_at": "2026-07-15T12:10:00+00:00"}}}
        self.calls = []

    def status(self):
        return self._status

    def unlock(self, method="windows_hello", secret=""):
        self.calls.append(("unlock", method, secret))
        return self._unlock

    def lock(self):
        self.calls.append(("lock",))
        return {"status": "ok"}

    def request_elevation(self, capability):
        self.calls.append(("elevate", capability))
        return self._elevation

    def revoke_elevation(self):
        self.calls.append(("revoke",))
        return {"status": "ok"}


def make_controller(api):
    return fifi_tray.TrayController(
        policy=fifi_tray.StartupPolicy(get=lambda n, d="": d),
        run_runtime=lambda argv: 0, notifier=RecordingNotifier(),
        toggle_mute_fn=lambda: True, select_voice_fn=lambda p: True,
        sleep=lambda s: None, security_api=api)


def test_security_menu_items_present_and_enabled():
    keys = {k for k, _ in ts.MENU_ITEMS}
    assert {ts.SECURITY_STATUS, ts.SECURITY_UNLOCK, ts.SECURITY_LOCK,
            ts.SECURITY_ELEVATE, ts.SECURITY_REVOKE} <= keys
    enabled = ts.menu_enabled(ts.STOPPED)
    assert enabled[ts.SECURITY_STATUS] and enabled[ts.SECURITY_LOCK]


def test_elevation_capabilities_are_scoped_never_global():
    assert "*" not in ts.ELEVATION_CAPABILITIES
    assert "email" in ts.ELEVATION_CAPABILITIES and "whatsapp" in ts.ELEVATION_CAPABILITIES


def test_status_reports_profile():
    ctrl = make_controller(FakeSecurityApi())
    ctrl.security_status()
    assert "profile standard" in ctrl.notifier.sent[-1]["message"]


def test_status_reports_locked():
    ctrl = make_controller(FakeSecurityApi(status={"enabled": True, "locked": True, "profile": "locked"}))
    ctrl.security_status()
    assert "locked" in ctrl.notifier.sent[-1]["message"]


def test_status_reports_active_elevation():
    ctrl = make_controller(FakeSecurityApi(status={
        "enabled": True, "locked": False, "profile": "standard",
        "elevation": {"capability": "email"}}))
    ctrl.security_status()
    assert "elevated: email" in ctrl.notifier.sent[-1]["message"]


def test_unlock_uses_windows_hello_only():
    api = FakeSecurityApi()
    ctrl = make_controller(api)
    assert ctrl.unlock_security() is True
    # the tray only ever calls unlock with the hello method + an EMPTY secret
    assert api.calls == [("unlock", "windows_hello", "")]


def test_unlock_failure_guides_to_localhost():
    api = FakeSecurityApi(unlock={"status": "rejected", "error_code": "UNLOCK_FAILED"})
    ctrl = make_controller(api)
    assert ctrl.unlock_security() is False
    assert "localhost" in ctrl.notifier.sent[-1]["message"]


def test_lock_now():
    api = FakeSecurityApi()
    ctrl = make_controller(api)
    assert ctrl.lock_security() is True
    assert ("lock",) in api.calls


def test_request_elevation_is_capability_scoped():
    api = FakeSecurityApi()
    ctrl = make_controller(api)
    ctrl.request_elevation("email")
    assert ("elevate", "email") in api.calls
    msg = ctrl.notifier.sent[-1]["message"]
    assert "email" in msg and "confirmation is still required" in msg


def test_request_elevation_rejected_is_graceful():
    api = FakeSecurityApi(elevation={"status": "rejected", "error_code": "UNLOCK_REQUIRED"})
    ctrl = make_controller(api)
    ctrl.request_elevation("email")
    assert "rejected" in ctrl.notifier.sent[-1]["message"]


def test_revoke_elevation():
    api = FakeSecurityApi()
    ctrl = make_controller(api)
    assert ctrl.revoke_elevation() is True
    assert ("revoke",) in api.calls


def test_tray_never_changes_base_profile():
    """The tray exposes NO profile-change action — that stays behind the spoken/
    localhost 'confirmar cambio de perfil' flow."""
    assert not hasattr(fifi_tray.TrayController, "change_profile")
    api = FakeSecurityApi()
    # the security api client used by the tray offers no profile-change method
    assert not hasattr(api, "change_profile")
    assert not any("profile" in name for name in dir(fifi_tray.DefaultSecurityApi)
                   if not name.startswith("_"))
