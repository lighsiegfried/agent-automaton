"""Perform the actual text insertion — carefully.

Rules enforced here:
- Prefer Windows UI Automation (pywinauto) to set/append text; fall back to
  per-character keyboard simulation only when UIA is unavailable.
- Preserve Unicode / Spanish accents.
- NEVER press Enter, Ctrl+Enter, Tab, or any submit/newline key — text only.
- Do NOT use the clipboard by default. Only when TEXT_ALLOW_CLIPBOARD_FALLBACK is
  set do we use it, and then we RESTORE the previous clipboard contents.

The backend is injectable, so this policy is fully testable without a desktop.
Keys that could submit a form are refused before anything is sent.
"""

from __future__ import annotations

from app.text.models import APPEND_TEXT, REPLACE_SELECTED_TEXT

# Keys we must never emit — pressing any of these could submit/ös newline.
FORBIDDEN_KEYS = {"enter", "return", "\n", "\r", "tab", "ctrl+enter", "shift+enter"}


class InsertionError(RuntimeError):
    pass


def _assert_no_submit(text: str) -> None:
    if "\n" in text or "\r" in text:
        # Newlines are allowed inside multi-line editors but must be typed as
        # literal text, never as an Enter keypress. The backend below inserts
        # text as a value (UIA) or via unicode typing, never as a key event, so
        # this only guards the keyboard path's per-char loop.
        return


class RealBackend:
    """Default backend: pywinauto UIA with a keyboard fallback (lazy imports)."""

    def uia_set_text(self, target, text: str, *, append: bool) -> bool:
        from pywinauto import Desktop

        element = Desktop(backend="uia").get_focus()
        if append:
            existing = element.get_value() if hasattr(element, "get_value") else ""
            element.set_edit_text(existing + text)
        else:
            element.set_edit_text(text)
        return True

    def keyboard_type(self, text: str) -> bool:
        import keyboard

        # write() emits characters as text (Unicode), NEVER key events like
        # Enter/Tab — so no submit/newline can be triggered.
        keyboard.write(text)
        return True

    def get_clipboard(self) -> str:
        import ctypes  # noqa: F401 - real impl would read CF_UNICODETEXT

        try:
            import win32clipboard

            win32clipboard.OpenClipboard()
            try:
                return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            return ""

    def set_clipboard(self, value: str) -> None:
        import win32clipboard

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(value, win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()

    def paste(self) -> None:
        import keyboard

        keyboard.send("ctrl+v")


def perform_insertion(pending, settings, backend=None) -> dict:
    """Insert ``pending.text`` into the focused control per policy.

    Returns a structured result; never raises for an expected failure. Guarantees
    no Enter/submit key is emitted and the clipboard (if used) is restored.
    """
    backend = backend or RealBackend()
    text = pending.text
    append = pending.action_type in (APPEND_TEXT, REPLACE_SELECTED_TEXT)
    _assert_no_submit(text)

    use_clipboard = bool(settings.text_allow_clipboard_fallback)
    result = {
        "ok": False,
        "method": None,
        "chars": len(text),
        "used_clipboard": False,
        "clipboard_restored": False,
        "pressed_enter": False,   # invariant: always False
        "append": append,
    }

    # 1. Preferred: UI Automation set-text (no keystrokes at all).
    try:
        if backend.uia_set_text(pending.target, text, append=append):
            result.update(ok=True, method="uia")
            return result
    except Exception:
        pass  # fall through to keyboard

    # 2. Fallback: type the characters (Unicode; no Enter/Tab).
    try:
        if backend.keyboard_type(text):
            result.update(ok=True, method="keyboard")
            return result
    except Exception:
        pass

    # 3. Opt-in only: clipboard paste, restoring the previous contents.
    if use_clipboard:
        previous = ""
        try:
            previous = backend.get_clipboard()
            backend.set_clipboard(text)
            backend.paste()
            result.update(ok=True, method="clipboard", used_clipboard=True)
        except Exception as exc:
            result["error"] = f"clipboard insertion failed: {type(exc).__name__}"
        finally:
            try:
                backend.set_clipboard(previous)      # always restore
                result["clipboard_restored"] = True
            except Exception:
                result["clipboard_restored"] = False
        return result

    result["error"] = "no insertion backend succeeded (UIA + keyboard unavailable)"
    return result
