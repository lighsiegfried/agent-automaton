"""Insertion backend policy: UIA/keyboard/clipboard, no Enter, restore (Phase 4A)."""

import types

from app.text import insertion
from app.text.models import TargetSummary


def _pending(text="Hola ñ áé", action_type="type_text"):
    return types.SimpleNamespace(
        text=text, action_type=action_type,
        target=TargetSummary(application="notepad", window_title="t", process="notepad.exe",
                             executable="notepad.exe", control_type="Edit", editable=True,
                             supported=True),
    )


def _settings(clipboard=False):
    return types.SimpleNamespace(text_allow_clipboard_fallback=clipboard)


class FakeBackend:
    def __init__(self, uia=True, keyboard=True, clip="OLD"):
        self.uia, self.kbd = uia, keyboard
        self.typed = []
        self.clip_history = [clip]
        self.pasted = False

    def uia_set_text(self, target, text, *, append):
        if not self.uia:
            raise RuntimeError("no uia")
        self.typed.append(("uia", text, append))
        return True

    def keyboard_type(self, text):
        if not self.kbd:
            raise RuntimeError("no keyboard")
        self.typed.append(("kbd", text))
        return True

    def get_clipboard(self):
        return self.clip_history[-1]

    def set_clipboard(self, value):
        self.clip_history.append(value)

    def paste(self):
        self.pasted = True


def test_forbidden_keys_include_enter_and_tab():
    assert "enter" in insertion.FORBIDDEN_KEYS
    assert "tab" in insertion.FORBIDDEN_KEYS
    assert "ctrl+enter" in insertion.FORBIDDEN_KEYS


def test_uia_path_preferred_no_enter():
    backend = FakeBackend(uia=True)
    result = insertion.perform_insertion(_pending(), _settings(), backend=backend)
    assert result["ok"] and result["method"] == "uia"
    assert result["pressed_enter"] is False
    assert backend.typed == [("uia", "Hola ñ áé", False)]


def test_keyboard_fallback_preserves_unicode_and_sends_no_keys():
    backend = FakeBackend(uia=False, keyboard=True)
    result = insertion.perform_insertion(_pending(text="café ¡hola!"), _settings(), backend=backend)
    assert result["ok"] and result["method"] == "keyboard"
    assert backend.typed == [("kbd", "café ¡hola!")]   # typed as text, no Enter/Tab
    assert result["pressed_enter"] is False


def test_clipboard_used_only_when_allowed_and_restores_previous():
    backend = FakeBackend(uia=False, keyboard=False, clip="ORIGINAL")
    result = insertion.perform_insertion(
        _pending(text="pega esto"), _settings(clipboard=True), backend=backend
    )
    assert result["ok"] and result["method"] == "clipboard"
    assert result["used_clipboard"] is True
    assert result["clipboard_restored"] is True
    assert backend.pasted is True
    # clipboard was set to our text, then restored to the original.
    assert backend.clip_history[-2:] == ["pega esto", "ORIGINAL"]


def test_no_clipboard_when_disabled_fails_cleanly():
    backend = FakeBackend(uia=False, keyboard=False)
    result = insertion.perform_insertion(_pending(), _settings(clipboard=False), backend=backend)
    assert result["ok"] is False
    assert result["used_clipboard"] is False
    assert "no insertion backend" in result["error"]


def test_append_action_sets_append_flag():
    backend = FakeBackend(uia=True)
    result = insertion.perform_insertion(_pending(action_type="append_text"), _settings(),
                                         backend=backend)
    assert result["append"] is True
    assert backend.typed[0][2] is True   # uia_set_text called with append=True
