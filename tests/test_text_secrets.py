"""Confirmation phrases + secret / password-field detection (Phase 4A)."""

from app.text import secrets


def test_only_exact_phrases_confirm():
    for phrase in ("confirmar escritura", "Escribirlo", "INSERT TEXT", " confirm typing "):
        assert secrets.is_confirmation_phrase(phrase) is True
    for nope in ("sí", "si", "yes", "ok", "hazlo", "vale", ""):
        assert secrets.is_confirmation_phrase(nope) is False


def test_detects_payment_card():
    ok, reasons = secrets.contains_likely_secret("pay with 4111 1111 1111 1111 today")
    assert ok and "payment_card_number" in reasons


def test_detects_api_keys_and_tokens():
    assert secrets.contains_likely_secret("key sk-ABCDEF0123456789ABCDEF")[0]
    assert secrets.contains_likely_secret("AKIAIOSFODNN7EXAMPLE here")[0]
    assert secrets.contains_likely_secret("Authorization: Bearer abcdef1234567890xyz")[0]


def test_detects_private_key_and_password_label():
    assert secrets.contains_likely_secret("-----BEGIN RSA PRIVATE KEY-----")[0]
    assert secrets.contains_likely_secret("password: hunter2")[0]
    assert secrets.contains_likely_secret("contraseña = miClaveSecreta")[0]


def test_plain_spanish_paragraph_is_not_secret():
    text = ("Hola, quería contarte que hoy fui al parque y vi muchos árboles "
            "y pájaros. Fue una tarde muy tranquila y agradable.")
    ok, reasons = secrets.contains_likely_secret(text)
    assert ok is False and reasons == []


def test_password_field_detection():
    assert secrets.looks_like_password_field("Edit", True) is True
    assert secrets.looks_like_password_field("PasswordBox", False) is True
    assert secrets.looks_like_password_field("Edit", False, field_name="OTP code") is True
    assert secrets.looks_like_password_field("Edit", False) is False
