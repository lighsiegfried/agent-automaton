"""Task Scheduler auto-start command building + enable/disable/status (Phase 3E)."""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_autostart as auto  # noqa: E402


class FakeRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.calls: list = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(
            returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def test_enable_command_is_per_user_delayed_onlogon():
    cmd = auto.build_enable_command(delay_seconds=20, task_name="T")
    assert cmd[0] == "schtasks" and "/Create" in cmd
    assert cmd[cmd.index("/TN") + 1] == "T"
    assert cmd[cmd.index("/SC") + 1] == "ONLOGON"
    assert cmd[cmd.index("/DELAY") + 1] == "0000:20"
    assert cmd[cmd.index("/RL") + 1] == "LIMITED"   # least privilege, no admin
    assert "/F" in cmd


def test_delay_field_formatting():
    assert auto._delay_field(20) == "0000:20"
    assert auto._delay_field(125) == "0002:05"
    assert auto._delay_field(0) == "0000:00"


def test_task_action_stores_no_secrets():
    tr = auto._tr_string()
    # Only the interpreter + the tray script — no tokens, env, or credentials.
    assert "fifi_tray.py" in tr
    for secret_marker in ("token", "password", "secret", "key=", "--env"):
        assert secret_marker not in tr.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="auto-start is Windows-only")
def test_enable_ok(monkeypatch):
    runner = FakeRun(returncode=0)
    result = auto.enable(delay_seconds=15, runner=runner, task_name="T")
    assert result["status"] == "ok"
    assert result["delay_seconds"] == 15
    assert result["launches"][-1].endswith("fifi_tray.py")
    assert runner.calls[0][0] == "schtasks"


@pytest.mark.skipif(sys.platform != "win32", reason="auto-start is Windows-only")
def test_enable_failure_reports_message():
    result = auto.enable(runner=FakeRun(returncode=1, stderr="access denied"), task_name="T")
    assert result["status"] == "error"
    assert "access denied" in result["message"]


@pytest.mark.skipif(sys.platform != "win32", reason="auto-start is Windows-only")
def test_disable_absent_task_is_ok():
    result = auto.disable(runner=FakeRun(returncode=1, stderr="ERROR: cannot find the task"),
                          task_name="T")
    assert result["status"] == "ok"
    assert "was not enabled" in result["detail"]


@pytest.mark.skipif(sys.platform != "win32", reason="auto-start is Windows-only")
def test_status_reflects_presence():
    on = auto.status(runner=FakeRun(returncode=0, stdout="Folder: \\\nTaskName  T  Ready"),
                     task_name="T")
    assert on["enabled"] is True
    off = auto.status(runner=FakeRun(returncode=1, stderr="cannot find"), task_name="T")
    assert off["enabled"] is False
