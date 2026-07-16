"""Consent phrases, sensitive-data blocking, and deterministic content parsing (5A)."""

import pytest

from app.memory import safety


# --- confirmation-phrase predicates ------------------------------------------------


def test_confirmation_phrase_predicates():
    assert safety.is_write_confirm("confirmar memoria")
    assert safety.is_write_confirm("Confirm Memory")
    assert safety.is_forget_confirm("confirmar olvido")
    assert safety.is_delete_confirm("eliminar memoria permanentemente")
    # A plain yes is never a confirmation.
    for junk in ("sí", "si", "yes", "ok", "confirmar", "olvido"):
        assert not safety.is_write_confirm(junk)
        assert not safety.is_forget_confirm(junk)
        assert not safety.is_delete_confirm(junk)


# --- sensitive-data blocking (item 4) ----------------------------------------------


@pytest.mark.parametrize("content", [
    "mi contraseña es hunter2",
    "el OTP es 482913",
    "usa esta api key sk-ABCDEF1234567890ABCDEF12",
    "token ghp_ABCDEFGHIJKLMNOP0123456789abcdef",
    "mi tarjeta es 4111 1111 1111 1111",
    "-----BEGIN RSA PRIVATE KEY-----",
    "session_id=abc123DEF456ghi789",
    "IBAN: ES9121000418450200051332",
    "mi dirección es 1234 Calle Falsa, Springfield 90210",
    "estoy tomando fluoxetina para la depresión",
    "voto por el partido verde en cada elección",
])
def test_sensitive_content_blocked(content):
    ok, category, reasons = safety.check_storable("nota", content)
    assert ok is False, f"should block: {content!r}"
    assert category and reasons


@pytest.mark.parametrize("content", [
    "mi editor preferido es VS Code",
    "el proyecto agent-automaton usa Python y FastAPI",
    "decidimos usar SQLite para la memoria local",
    "recuérdame revisar el PR de los viernes",
])
def test_harmless_content_allowed(content):
    ok, category, _ = safety.check_storable("nota", content)
    assert ok is True and category == ""


def test_empty_is_not_storable():
    assert safety.check_storable("", "")[0] is False


# --- deterministic parsing ---------------------------------------------------------


def test_infer_type():
    assert safety.infer_type("mi editor preferido es VS Code") == "preference"
    assert safety.infer_type("decidimos usar SQLite") == "decision"
    assert safety.infer_type("el proyecto usa FastAPI") == "project"
    assert safety.infer_type("siempre revisa el lint antes de commitear") == "instruction"
    assert safety.infer_type("pasos para desplegar: build, test, ship") == "workflow"
    assert safety.infer_type("la capital de Francia es París") == "fact"


def test_infer_subject_stable_across_values():
    a = safety.infer_subject("mi editor preferido es VS Code")
    b = safety.infer_subject("mi editor preferido es Neovim")
    assert a == b == "editor preferido"
    assert safety.infer_subject("hoy llueve") == ""   # no copula → no subject key


def test_infer_entities_and_title():
    entities = safety.infer_entities("mi editor preferido es VS Code")
    assert "vs code" in entities
    assert safety.infer_title("mi editor preferido es VS Code", "editor preferido") == "Editor preferido"


def test_untrusted_sources_are_named():
    # The service refuses these channels; keep the set explicit for the guarantee.
    from app.memory.service import UNTRUSTED_SOURCES

    assert {"email", "web", "browser", "wake"} <= UNTRUSTED_SOURCES
