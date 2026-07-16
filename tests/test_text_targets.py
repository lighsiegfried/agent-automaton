"""Windows target inspector classification (Phase 4A) — injected raw probe."""

import types

from app.text import targets


def _settings(**over):
    base = dict(
        text_allowed_apps_list=["notepad", "wordpad", "winword", "chrome", "edge"],
        text_block_password_fields=True,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _probe(**over):
    raw = {
        "title": "Untitled - Notepad", "process_name": "notepad.exe",
        "exe_path": r"C:\Windows\System32\notepad.exe", "control_type": "Edit",
        "is_password": False, "is_elevated": False, "editable": True, "error": None,
    }
    raw.update(over)
    return lambda: raw


def test_supported_editable_target_is_allowed():
    t = targets.active_target(_probe(), _settings())
    assert t.application == "notepad"
    assert t.supported is True and t.blocked is False
    assert t.executable == "notepad.exe"          # basename only, no full path
    assert "System32" not in t.executable


def test_terminal_is_blocked():
    t = targets.active_target(_probe(process_name="powershell.exe", title="Windows PowerShell"),
                              _settings())
    assert t.blocked is True and "terminal" in t.blocked_reason


def test_registry_tool_is_blocked():
    t = targets.active_target(_probe(process_name="regedit.exe"), _settings())
    assert t.blocked is True and "system" in t.blocked_reason.lower()


def test_password_field_is_blocked():
    t = targets.active_target(
        _probe(process_name="chrome.exe", is_password=True, control_type="Edit"), _settings()
    )
    assert t.is_password_field is True
    assert t.blocked is True and "password" in t.blocked_reason


def test_elevated_window_is_blocked():
    t = targets.active_target(_probe(is_elevated=True), _settings())
    assert t.blocked is True and "elevated" in t.blocked_reason


def test_unsupported_app_is_blocked():
    t = targets.active_target(_probe(process_name="secretapp.exe"), _settings())
    assert t.blocked is True and "unsupported" in t.blocked_reason


def test_non_editable_control_is_blocked():
    t = targets.active_target(_probe(editable=False, control_type="Button"), _settings())
    assert t.blocked is True and "editable" in t.blocked_reason


def test_edge_alias_resolves_to_allowed():
    t = targets.active_target(_probe(process_name="msedge.exe", title="Edge"), _settings())
    assert t.application == "edge" and t.supported is True


def test_probe_error_is_blocked_and_safe():
    t = targets.active_target(lambda: {"error": "pywin32 not installed"}, _settings())
    assert t.blocked is True and "pywin32" in t.blocked_reason
    assert t.window_title == "" and t.executable == ""


def test_signature_changes_with_window():
    a = targets.active_target(_probe(title="A - Notepad"), _settings())
    b = targets.active_target(_probe(title="B - Notepad"), _settings())
    assert a.signature() != b.signature()
