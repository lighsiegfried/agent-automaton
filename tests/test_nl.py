"""Bilingual NL parsing into service intents (Phase 4B.1)."""

import pytest

from app.core import nl
from app.schemas.commands import Intent


@pytest.mark.parametrize("text, intent, args", [
    ("redacta un correo para Ana", Intent.DRAFT_TEXT.value, {"instruction": "un correo para Ana"}),
    ("draft an email to the team", Intent.DRAFT_TEXT.value, {"instruction": "email to the team"}),
    ("reescribe este párrafo", Intent.REWRITE_TEXT.value, {"instruction": "este párrafo"}),
    ("abre el navegador y busca documentación de Playwright", Intent.BROWSER_SEARCH.value,
     {"query": "documentación de Playwright"}),
    ("open the browser and search for python docs", Intent.BROWSER_SEARCH.value,
     {"query": "python docs"}),
    ("resume esta página", Intent.BROWSER_SUMMARIZE.value, {}),
    ("summarize this page", Intent.BROWSER_SUMMARIZE.value, {}),
    ("encuentra la sección de precios", Intent.BROWSER_FIND.value, {"query": "precios"}),
    ("find the pricing section", Intent.BROWSER_FIND.value, {"query": "pricing"}),
    ("abre microsoft.com", Intent.BROWSER_OPEN.value, {"url": "microsoft.com"}),
    ("cierra el navegador", Intent.BROWSER_CLOSE.value, {}),
    ("prepara el formulario", Intent.BROWSER_PREPARE_FORM.value, {"fields": {}}),
])
def test_intent_routing(text, intent, args):
    command = nl.parse(text)
    assert command is not None
    assert command.intent == intent
    for key, value in args.items():
        assert command.arguments.get(key) == value


def test_type_text_insertion_with_content():
    command = nl.parse("escribe esto en la ventana: Hola mundo")
    assert command.intent == Intent.TYPE_TEXT.value
    assert command.arguments["text"] == "Hola mundo"
    assert command.arguments["mode"] == "insert"


def test_confirmation_phrases_are_detected():
    for phrase in ("confirmar escritura", "insert text", "confirmar formulario", "confirm form fill"):
        command = nl.parse(phrase)
        assert command is not None and command.is_confirmation is True
        assert command.confirmation_phrase == phrase


def test_plain_si_is_not_a_service_command():
    assert nl.parse("sí") is None
    assert nl.parse("yes") is None
    assert nl.parse("ok") is None


def test_cancel_detected():
    assert nl.parse("cancelar").intent == "cancel"
    assert nl.parse("cancel that").intent == "cancel"


def test_legacy_commands_fall_through():
    # Plain "busca X" and "abre <app>" stay with the legacy tool router.
    assert nl.parse("busca gatos") is None
    assert nl.parse("abre notepad") is None
    assert nl.parse("open downloads") is None
