"""URL policy + form-field safety + confirmation phrases (Phase 4B)."""

import pytest

from app.browser import safety

ALLOWED = ["https", "http"]


def test_normalize_adds_https_and_lowercases_host():
    assert safety.normalize_url("Example.COM/Path") == "https://example.com/Path"
    assert safety.normalize_url("  http://x.com/a?b=1 ") == "http://x.com/a?b=1"


def test_public_https_is_allowed():
    v = safety.validate_url("example.com/docs", ALLOWED)
    assert v["ok"] is True and v["scheme"] == "https" and v["host"] == "example.com"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "javascript:alert(1)", "data:text/html,<b>x",
    "chrome://settings", "edge://flags", "about:blank", "view-source:https://x.com",
    "blob:https://x", "ftp://host/f", "ws://host",
])
def test_blocked_schemes_are_rejected(url):
    v = safety.validate_url(url, ALLOWED)
    assert v["ok"] is False


def test_scheme_not_in_allowlist_rejected():
    v = safety.validate_url("http://x.com", ["https"])   # only https allowed
    assert v["ok"] is False and "allowlist" in v["reason"]


@pytest.mark.parametrize("url", [
    "http://localhost:8000", "https://127.0.0.1", "https://10.0.0.5",
    "https://192.168.1.10", "https://172.16.0.1", "https://169.254.1.1",
    "https://[::1]/", "https://printer.local", "https://0.0.0.0",
])
def test_private_networks_blocked(url):
    v = safety.validate_url(url, ALLOWED, block_private=True)
    assert v["ok"] is False and ("private" in v["reason"] or "loopback" in v["reason"])


def test_private_allowed_when_disabled():
    assert safety.validate_url("http://localhost:8000", ALLOWED, block_private=False)["ok"] is True


def test_is_private_host_helper():
    assert safety.is_private_host("127.0.0.1") is True
    assert safety.is_private_host("example.com") is False
    assert safety.is_private_host("foo.local") is True


@pytest.mark.parametrize("kwargs, expected", [
    ({"field_type": "password"}, True),
    ({"name": "cardNumber"}, True),
    ({"name": "cc-csc", "autocomplete": "cc-csc"}, True),
    ({"autocomplete": "one-time-code"}, True),
    ({"label": "CVV"}, True),
    ({"name": "api_key"}, True),
    ({"label": "Número de tarjeta"}, True),
    ({"name": "full_name", "field_type": "text"}, False),
    ({"name": "comment", "label": "Your message"}, False),
])
def test_sensitive_field_detection(kwargs, expected):
    sensitive, _ = safety.is_sensitive_field(**kwargs)
    assert sensitive is expected


def test_confirmation_phrases():
    for p in ("confirmar formulario", "Llenar Formulario", "confirm form fill"):
        assert safety.is_confirmation_phrase(p) is True
    for p in ("sí", "ok", "confirmar", "fill it"):
        assert safety.is_confirmation_phrase(p) is False
