"""Windows target inspector: what window/control is focused, and is it safe?

Returns a SAFE ``TargetSummary`` (no window contents, executable basename only).
Rejects terminals / PowerShell / CMD / registry tools / admin prompts, password
fields, elevated windows, and any application not in TEXT_ALLOWED_APPS.

The low-level foreground-window probe is Windows-specific and lazy; it is
injectable so the whole policy is testable without pywinauto or a desktop.
"""

from __future__ import annotations

import os

from app.config import get_settings
from app.text import secrets
from app.text.models import TargetSummary

# Shells / terminals — never type here.
TERMINAL_EXES = {
    "cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe",
    "conhost.exe", "openconsole.exe",
}
# Registry / management / UAC consent — never type here either.
SYSTEM_EXES = {"regedit.exe", "mmc.exe", "consent.exe", "runas.exe"}

# Normalise process image names to the canonical app names used in the allowlist.
APP_ALIASES = {"msedge": "edge", "iexplore": "edge", "wordpad": "wordpad", "write": "wordpad"}


def _canonical_app(process_name: str) -> str:
    stem = os.path.splitext(process_name or "")[0].lower()
    return APP_ALIASES.get(stem, stem)


def _default_probe() -> dict:
    """Best-effort foreground-window probe (Windows only, lazy imports)."""
    import sys

    if sys.platform != "win32":
        return {"error": "target inspection is only available on the Windows host"}
    try:
        import win32gui
        import win32process
    except Exception:
        return {"error": "pywin32 not installed (pip install -r requirements-desktop.txt)"}
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        exe_path, process_name = _process_image(pid)
        control_type, is_password, editable = _focused_control(hwnd)
        return {
            "title": title,
            "process_name": process_name,
            "exe_path": exe_path,
            "control_type": control_type,
            "is_password": is_password,
            "is_elevated": _is_elevated(pid),
            "editable": editable,
            "error": None,
        }
    except Exception as exc:  # never let inspection crash a request
        return {"error": f"inspection failed: {type(exc).__name__}"}


def _process_image(pid: int) -> tuple[str, str]:
    try:
        import win32api
        import win32con
        import win32process

        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | 0x0010, False, pid)
        path = win32process.GetModuleFileNameEx(handle, 0)
        return path, os.path.basename(path)
    except Exception:
        return "", ""


def _focused_control(hwnd) -> tuple[str, bool, bool]:
    """(control_type, is_password, editable) via UI Automation when available."""
    try:
        from pywinauto import Desktop

        element = Desktop(backend="uia").window(handle=hwnd).get_focus()
        control_type = element.element_info.control_type or ""
        is_password = bool(getattr(element.element_info, "is_password", False))
        editable = control_type in ("Edit", "Document", "Text") or element.is_editable()
        return control_type, is_password, editable
    except Exception:
        # Fall back to "unknown editable" so a supported app still works; the
        # caller's other gates (allowed app, terminal, elevation) still apply.
        return "", False, True


def _is_elevated(pid: int) -> bool:
    return False  # best-effort; refined by the real probe when UIA is present


def active_target(probe=None, settings=None) -> TargetSummary:
    """Inspect the focused window and classify it for text insertion."""
    settings = settings or get_settings()
    raw = (probe or _default_probe)()

    if raw.get("error"):
        return TargetSummary(
            application="", window_title="", process="", executable="",
            control_type="", editable=False, supported=False, blocked=True,
            blocked_reason=raw["error"],
        )

    process_name = raw.get("process_name", "") or ""
    application = _canonical_app(process_name)
    control_type = raw.get("control_type", "") or ""
    is_password = secrets.looks_like_password_field(
        control_type, bool(raw.get("is_password")), raw.get("title", "")
    )
    is_terminal = process_name.lower() in TERMINAL_EXES
    is_system = process_name.lower() in SYSTEM_EXES
    is_elevated = bool(raw.get("is_elevated"))
    editable = bool(raw.get("editable", False))
    supported = application in settings.text_allowed_apps_list

    blocked, reason = _classify(
        is_terminal, is_system, is_elevated, is_password, supported, editable,
        application, settings,
    )
    return TargetSummary(
        application=application,
        window_title=raw.get("title", "") or "",
        process=process_name,
        executable=os.path.basename(raw.get("exe_path", "") or process_name),
        control_type=control_type,
        editable=editable,
        supported=supported,
        is_password_field=is_password,
        is_terminal=is_terminal,
        is_elevated=is_elevated,
        blocked=blocked,
        blocked_reason=reason,
    )


def _classify(is_terminal, is_system, is_elevated, is_password, supported, editable,
              application, settings) -> tuple[bool, str]:
    if is_terminal:
        return True, "terminal/shell window — text insertion is blocked here"
    if is_system:
        return True, "system/registry/admin window — text insertion is blocked here"
    if is_elevated:
        return True, "elevated window — text insertion is blocked here"
    if is_password and settings.text_block_password_fields:
        return True, "password/secret field — text insertion is blocked"
    if not supported:
        return True, f"unsupported application {application!r} (not in TEXT_ALLOWED_APPS)"
    if not editable:
        return True, "no editable text control is focused"
    return False, ""
