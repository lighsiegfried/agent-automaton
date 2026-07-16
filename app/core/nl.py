"""Bilingual (ES/EN) natural-language parser for conversational service intents.

Maps spoken/typed commands to a structured ``ServiceCommand(intent, arguments)``
for the deterministic dispatcher. Returns None when nothing matches, so the
command falls through to the existing rule/tool router unchanged (legacy
"busca X" / "abre <app>" behaviour is preserved).

Confirmation phrases are recognised via the pending broker's domain registry so
routing stays correct as new domains (WhatsApp) are added. A bare "sí" never
matches a confirmation here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core import pending
from app.schemas.commands import Intent


@dataclass
class ServiceCommand:
    intent: str
    arguments: dict = field(default_factory=dict)
    is_confirmation: bool = False
    confirmation_phrase: str = ""


def _clean(value: str) -> str:
    return value.strip().strip('"').strip("«»").strip().rstrip(".").strip()


# (pattern, intent, arg builder). First match wins; checked after confirm/cancel.
_PATTERNS: list[tuple[re.Pattern[str], str, object]] = [
    # --- browser: open browser + search combined -----------------------------------
    (re.compile(r"\babre?\s+(?:el\s+)?navegador\s+y\s+busca(?:r)?\s+(.+)", re.I),
     Intent.BROWSER_SEARCH.value, lambda m: {"query": _clean(m.group(1))}),
    (re.compile(r"\bopen\s+(?:the\s+)?browser\s+and\s+search(?:\s+for)?\s+(.+)", re.I),
     Intent.BROWSER_SEARCH.value, lambda m: {"query": _clean(m.group(1))}),
    (re.compile(r"\bbusca(?:r)?\s+(.+?)\s+en\s+el\s+navegador\b", re.I),
     Intent.BROWSER_SEARCH.value, lambda m: {"query": _clean(m.group(1))}),
    (re.compile(r"\bsearch\s+(?:for\s+)?(.+?)\s+in\s+the\s+browser\b", re.I),
     Intent.BROWSER_SEARCH.value, lambda m: {"query": _clean(m.group(1))}),

    # --- browser: summarize / read -------------------------------------------------
    (re.compile(r"\b(?:resume|resumir|res[uú]mstage?)\s+(?:esta|la)\s+p[aá]gina\b", re.I),
     Intent.BROWSER_SUMMARIZE.value, lambda m: {}),
    (re.compile(r"\bsummari[sz]e\s+(?:this|the)\s+page\b", re.I),
     Intent.BROWSER_SUMMARIZE.value, lambda m: {}),
    (re.compile(r"\b(?:lee|leer)\s+(?:esta|la)\s+p[aá]gina\b", re.I),
     Intent.BROWSER_READ.value, lambda m: {}),
    (re.compile(r"\bread\s+(?:this|the)\s+page\b", re.I),
     Intent.BROWSER_READ.value, lambda m: {}),

    # --- browser: find a section by text -------------------------------------------
    (re.compile(r"\b(?:encuentra|busca)\s+(?:la\s+secci[oó]n\s+de\s+|d[oó]nde\s+hablan\s+de\s+)(.+)", re.I),
     Intent.BROWSER_FIND.value, lambda m: {"query": _clean(m.group(1))}),
    (re.compile(r"\bfind\s+(?:the\s+)?(.+?)\s+section\b", re.I),
     Intent.BROWSER_FIND.value, lambda m: {"query": _clean(m.group(1))}),

    # --- browser: scroll / open link / close ---------------------------------------
    (re.compile(r"\b(?:despl[aá]zate|baja|scroll)\s+(?:hacia\s+)?(abajo|down)\b", re.I),
     Intent.BROWSER_SCROLL.value, lambda m: {"direction": "down"}),
    (re.compile(r"\b(?:despl[aá]zate|sube|scroll)\s+(?:hacia\s+)?(arriba|up)\b", re.I),
     Intent.BROWSER_SCROLL.value, lambda m: {"direction": "up"}),
    (re.compile(r"\b(?:abre|open)\s+(?:el\s+)?(?:enlace|link)\s+(.+)", re.I),
     Intent.BROWSER_OPEN_LINK.value, lambda m: {"text": _clean(m.group(1))}),
    (re.compile(r"\b(?:cierra|close)\s+(?:el\s+)?(?:navegador|browser|sesi[oó]n)\b", re.I),
     Intent.BROWSER_CLOSE.value, lambda m: {}),

    # --- browser: prepare form -----------------------------------------------------
    (re.compile(r"\b(?:prepara|llena|rellena)\s+(?:este\s+|el\s+)?formulario\b", re.I),
     Intent.BROWSER_PREPARE_FORM.value, lambda m: {"fields": {}}),
    (re.compile(r"\bprepare\s+(?:this\s+|the\s+)?form\b", re.I),
     Intent.BROWSER_PREPARE_FORM.value, lambda m: {"fields": {}}),

    # --- browser: open a site / url ------------------------------------------------
    (re.compile(r"\b(?:abre|ve\s+a|navega\s+a)\s+(?:el\s+)?(?:sitio\s+web\s+|p[aá]gina\s+(?:de\s+|web\s+)?)?"
                r"((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)", re.I),
     Intent.BROWSER_OPEN.value, lambda m: {"url": _clean(m.group(1))}),
    (re.compile(r"\b(?:open|go\s+to|navigate\s+to)\s+(?:the\s+)?(?:site\s+|page\s+)?"
                r"((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)", re.I),
     Intent.BROWSER_OPEN.value, lambda m: {"url": _clean(m.group(1))}),
    (re.compile(r"\babre?\s+(?:el\s+)?navegador\b", re.I),
     Intent.BROWSER_OPEN.value, lambda m: {"url": ""}),
    (re.compile(r"\bopen\s+(?:the\s+)?browser\b", re.I),
     Intent.BROWSER_OPEN.value, lambda m: {"url": ""}),

    # --- text: drafting ------------------------------------------------------------
    (re.compile(r"\b(?:redacta|redactar|escribe\s+un\s+borrador\s+de)\s+(.+)", re.I),
     Intent.DRAFT_TEXT.value, lambda m: {"instruction": _clean(m.group(1))}),
    (re.compile(r"\bdraft\s+(?:an|a|me)?\s*(.+)", re.I),
     Intent.DRAFT_TEXT.value, lambda m: {"instruction": _clean(m.group(1))}),
    (re.compile(r"\b(?:reescribe|reformula|reescribir)\s+(.+)", re.I),
     Intent.REWRITE_TEXT.value, lambda m: {"instruction": _clean(m.group(1))}),
    (re.compile(r"\brewrite\s+(.+)", re.I),
     Intent.REWRITE_TEXT.value, lambda m: {"instruction": _clean(m.group(1))}),

    # --- text: insertion into the focused window -----------------------------------
    (re.compile(r"\b(?:escribe\s+esto|inserta(?:\s+este\s+texto)?|escribe)\s+en\s+"
                r"(?:la\s+ventana|notepad|wordpad|word|el\s+bloc)\s*:?\s*(.*)", re.I),
     Intent.TYPE_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "insert"}),
    (re.compile(r"\b(?:type|insert|write)\s+this\s+(?:in(?:to)?\s+the\s+window|text)\s*:?\s*(.*)", re.I),
     Intent.TYPE_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "insert"}),
    (re.compile(r"\b(?:a[ñn]ade|agrega)\s+(.+)", re.I),
     Intent.APPEND_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "append"}),
    (re.compile(r"\bappend\s+(.+)", re.I),
     Intent.APPEND_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "append"}),
    (re.compile(r"\b(?:reemplaza\s+lo\s+seleccionado(?:\s+(?:con|por))?)\s+(.+)", re.I),
     Intent.REPLACE_SELECTED_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "replace"}),
    (re.compile(r"\breplace\s+(?:the\s+)?selection\s+with\s+(.+)", re.I),
     Intent.REPLACE_SELECTED_TEXT.value, lambda m: {"text": _clean(m.group(1)), "mode": "replace"}),
]

_CANCEL = re.compile(r"^(?:cancelar|cancela(?:\s+(?:eso|la\s+acci[oó]n))?|cancel(?:\s+that|\s+the\s+action)?)$", re.I)

# Integrations (WhatsApp, email) register their own patterns here so app/core
# never imports app/integrations. Checked BEFORE the built-in patterns so an
# integration phrase ("busca a Ana en WhatsApp") wins over a generic one.
_EXTRA_PATTERNS: list[tuple[re.Pattern[str], str, object]] = []


def register_patterns(patterns) -> None:
    _EXTRA_PATTERNS.extend(patterns)


def parse(text: str) -> ServiceCommand | None:
    routing = (text or "").strip()
    if not routing:
        return None

    # 1. A confirmation phrase for ANY registered domain (never a bare "sí").
    if pending.phrase_domain(routing) is not None:
        return ServiceCommand(intent="confirm", is_confirmation=True,
                              confirmation_phrase=routing)

    # 2. Cancel the active pending action.
    if _CANCEL.match(routing.rstrip(".")):
        return ServiceCommand(intent="cancel")

    # 3. Integration patterns, then the built-in service intents.
    for pattern, intent, build in (*_EXTRA_PATTERNS, *_PATTERNS):
        match = pattern.search(routing)
        if match:
            return ServiceCommand(intent=intent, arguments=build(match))
    return None
