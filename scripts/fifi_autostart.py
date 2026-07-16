"""Controlled auto-start for Fifi via Windows Task Scheduler (Phase 3E).

    python scripts/local_runtime.py autostart-enable
    python scripts/local_runtime.py autostart-disable
    python scripts/local_runtime.py autostart-status

Design / safety:
- Prefers a per-user **Task Scheduler** ONLOGON task with a bounded startup delay
  (FIFI_STARTUP_DELAY_SECONDS). A per-user task needs NO administrator rights.
- The task action only ever launches the tray app (``pythonw scripts/fifi_tray.py``)
  — it stores NO secrets, tokens, or credentials in the task arguments.
- ``enable`` is idempotent (/F overwrites); ``disable`` removes only OUR task and
  never touches other scheduled tasks. A Startup-folder fallback is provided for
  environments where schtasks is unavailable.

The subprocess runner is injectable so every command is unit-testable without
actually touching the Task Scheduler.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
TRAY_SCRIPT = SCRIPTS_DIR / "fifi_tray.py"

# Stable, unambiguous task name — we only ever create/delete THIS one.
TASK_NAME = "FifiAssistantTray"


def pythonw() -> str:
    """The windowless interpreter (pythonw.exe) so no console flashes at logon."""
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return str(candidate if candidate.exists() else exe)


def tray_launch_command() -> list[str]:
    """argv that starts the tray app — the ONLY thing autostart launches."""
    return [pythonw(), str(TRAY_SCRIPT)]


def _tr_string() -> str:
    """The /TR command string (quoted). Contains no secrets — just interpreter + script."""
    return " ".join(f'"{part}"' for part in tray_launch_command())


def _delay_field(delay_seconds: int) -> str:
    """schtasks /DELAY takes mmmm:ss."""
    delay_seconds = max(0, int(delay_seconds))
    return f"{delay_seconds // 60:04d}:{delay_seconds % 60:02d}"


def build_enable_command(delay_seconds: int = 20, task_name: str = TASK_NAME) -> list[str]:
    """The full schtasks /Create argv for a per-user, delayed ONLOGON task."""
    return [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", _tr_string(),
        "/SC", "ONLOGON",
        "/DELAY", _delay_field(delay_seconds),
        "/RL", "LIMITED",   # least privilege — never elevates
        "/F",               # overwrite if it already exists (idempotent)
    ]


def build_disable_command(task_name: str = TASK_NAME) -> list[str]:
    return ["schtasks", "/Delete", "/TN", task_name, "/F"]


def build_status_command(task_name: str = TASK_NAME) -> list[str]:
    return ["schtasks", "/Query", "/TN", task_name]


def _run(cmd: list[str], runner) -> subprocess.CompletedProcess:
    return runner(cmd, capture_output=True, text=True, timeout=30)


def enable(delay_seconds: int = 20, runner=subprocess.run, task_name: str = TASK_NAME) -> dict:
    if sys.platform != "win32":
        return {"status": "error", "message": "auto-start is Windows-only"}
    if not TRAY_SCRIPT.is_file():
        return {"status": "error", "message": f"tray script missing: {TRAY_SCRIPT}"}
    cmd = build_enable_command(delay_seconds, task_name)
    try:
        result = _run(cmd, runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "message": f"schtasks failed to run: {exc}"}
    if result.returncode != 0:
        return {"status": "error", "message": (result.stderr or result.stdout or "").strip(),
                "command": cmd}
    return {
        "status": "ok",
        "task_name": task_name,
        "delay_seconds": delay_seconds,
        "launches": tray_launch_command(),
        "detail": "per-user ONLOGON task created (no admin, delayed, no secrets stored)",
    }


def disable(runner=subprocess.run, task_name: str = TASK_NAME) -> dict:
    if sys.platform != "win32":
        return {"status": "error", "message": "auto-start is Windows-only"}
    try:
        result = _run(build_disable_command(task_name), runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "message": f"schtasks failed to run: {exc}"}
    if result.returncode != 0:
        out = (result.stderr or result.stdout or "").lower()
        if "cannot find" in out or "does not exist" in out:
            return {"status": "ok", "task_name": task_name, "detail": "was not enabled"}
        return {"status": "error", "message": (result.stderr or result.stdout or "").strip()}
    return {"status": "ok", "task_name": task_name, "detail": "auto-start disabled"}


def status(runner=subprocess.run, task_name: str = TASK_NAME) -> dict:
    if sys.platform != "win32":
        return {"enabled": False, "task_name": task_name, "detail": "Windows-only"}
    try:
        result = _run(build_status_command(task_name), runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"enabled": False, "task_name": task_name, "detail": f"query failed: {exc}"}
    enabled = result.returncode == 0 and task_name in (result.stdout or "")
    return {
        "enabled": enabled,
        "task_name": task_name,
        "detail": "scheduled (ONLOGON, delayed)" if enabled else "not scheduled",
    }
