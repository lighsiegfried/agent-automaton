"""Phase 6C — desktop client-side safety: sanitize (no raw HTML), loopback-only base
URL, and the no-generic-confirm phrase rule."""

import pytest

from desktop import safety


def test_sanitize_escapes_markup():
    out = safety.sanitize("<script>alert('x')</script>")
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_sanitize_strips_control_chars():
    out = safety.sanitize("a\x00b\x07c")
    assert "\x00" not in out and "\x07" not in out and "abc" in out


def test_sanitize_truncates():
    assert safety.sanitize("x" * 10000, max_len=100).endswith("…")


def test_sanitize_none_is_empty():
    assert safety.sanitize(None) == ""


def test_sanitize_line_collapses_newlines():
    assert "\n" not in safety.sanitize_line("a\nb\nc")


@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:8000", True),
    ("http://localhost:8000", True),
    ("https://[::1]:8000", True),
    ("http://evil.example.com", False),
    ("http://10.0.0.5:8000", False),
    ("ftp://127.0.0.1", False),
    ("not a url", False),
])
def test_is_local_base(url, ok):
    assert safety.is_local_base(url) is ok


def test_require_local_base_raises_on_remote():
    with pytest.raises(ValueError):
        safety.require_local_base("http://evil.example.com")
    assert safety.require_local_base("http://127.0.0.1:8000") == "http://127.0.0.1:8000"


def test_confirmable_phrase_present_and_absent():
    assert safety.confirmable_phrase({"required_confirmation_phrase": "confirmar envío"}) == "confirmar envío"
    assert safety.confirmable_phrase({"required_confirmation_phrase": ""}) is None
    assert safety.confirmable_phrase({}) is None
    assert safety.confirmable_phrase(None) is None
