"""local_runtime wiring for the tray + auto-start commands (Phase 3E)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_runtime  # noqa: E402


def test_parser_registers_autostart_and_tray():
    parser = local_runtime.build_parser()
    assert parser.parse_args(["tray"]).func is local_runtime.cmd_tray
    assert parser.parse_args(["autostart-enable"]).func is local_runtime.cmd_autostart_enable
    assert parser.parse_args(["autostart-disable"]).func is local_runtime.cmd_autostart_disable
    assert parser.parse_args(["autostart-status"]).func is local_runtime.cmd_autostart_status


def test_autostart_enable_dispatches(monkeypatch, capsys):
    captured = {}

    def fake_enable(delay_seconds, **kw):
        captured["delay"] = delay_seconds
        return {"status": "ok", "task_name": "FifiAssistantTray", "delay_seconds": delay_seconds,
                "launches": ["pythonw", "fifi_tray.py"], "detail": "created"}

    monkeypatch.setattr(local_runtime.fifi_autostart, "enable", fake_enable)
    args = local_runtime.build_parser().parse_args(["autostart-enable", "--delay", "25"])
    assert local_runtime.cmd_autostart_enable(args) == 0
    assert captured["delay"] == 25
    assert "FifiAssistantTray" in capsys.readouterr().out


def test_autostart_enable_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime.fifi_autostart, "enable",
                        lambda delay_seconds, **kw: {"status": "error", "message": "nope"})
    args = local_runtime.build_parser().parse_args(["autostart-enable"])
    assert local_runtime.cmd_autostart_enable(args) == 1
    assert "nope" in capsys.readouterr().out


def test_autostart_status_dispatches(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime.fifi_autostart, "status",
                        lambda: {"enabled": True, "task_name": "T", "detail": "scheduled"})
    args = local_runtime.build_parser().parse_args(["autostart-status"])
    assert local_runtime.cmd_autostart_status(args) == 0
    assert "Enabled         : True" in capsys.readouterr().out


def test_autostart_disable_dispatches(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime.fifi_autostart, "disable",
                        lambda: {"status": "ok", "task_name": "T", "detail": "auto-start disabled"})
    args = local_runtime.build_parser().parse_args(["autostart-disable"])
    assert local_runtime.cmd_autostart_disable(args) == 0
    assert "disabled" in capsys.readouterr().out
