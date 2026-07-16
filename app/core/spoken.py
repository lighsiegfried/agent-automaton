"""Concise bilingual spoken responses for conversational outcomes (Phase 4B.1).

Only ever receives SAFE context (counts, app/page titles, recipient names) — never
the draft text, field values, tokens, or hidden content. The dispatcher is
responsible for passing safe context; these templates just phrase it.
"""

from __future__ import annotations

# situation -> (english, spanish). {app}/{count}/{title}/{phrase}/{recipient}
# are the only interpolations, and all are non-secret.
_TEMPLATES: dict[str, tuple[str, str]] = {
    "draft_prepared": (
        "I drafted {count} characters for {app}. Say '{phrase}' to insert it.",
        "Preparé un borrador de {count} caracteres para {app}. Di '{phrase}' para insertarlo.",
    ),
    "form_prepared": (
        "The form is ready with {count} field(s). Say '{phrase}' to fill it — I won't submit.",
        "El formulario está listo con {count} campo(s). Di '{phrase}' para llenarlo — no lo enviaré.",
    ),
    "inserted": (
        "Done — I inserted the text.",
        "Listo — inserté el texto.",
    ),
    "inserted_simulated": (
        "Simulated — real text input is off, so nothing was typed.",
        "Simulado — la escritura real está desactivada, no se escribió nada.",
    ),
    "form_filled": (
        "Done — I filled the form without submitting it.",
        "Listo — llené el formulario sin enviarlo.",
    ),
    "confirmation_required": (
        "I need an exact confirmation phrase — a plain 'yes' won't do it.",
        "Necesito una frase de confirmación exacta — un simple 'sí' no basta.",
    ),
    "confirmation_mismatch": (
        "That phrase doesn't match the pending action. Nothing was done.",
        "Esa frase no corresponde a la acción pendiente. No hice nada.",
    ),
    "target_changed": (
        "The target changed since I prepared it, so I cancelled the action.",
        "El destino cambió desde que lo preparé, así que cancelé la acción.",
    ),
    "action_expired": (
        "That action expired. Please ask again.",
        "Esa acción expiró. Pídemelo de nuevo.",
    ),
    "field_blocked": (
        "I won't fill that — it looks like a sensitive field.",
        "No llenaré eso — parece un campo sensible.",
    ),
    "secret_blocked": (
        "I won't insert that — it looks like it contains a secret.",
        "No insertaré eso — parece contener información secreta.",
    ),
    "ambiguous": (
        "There are several matches — please be more specific.",
        "Hay varias coincidencias — sé más específico, por favor.",
    ),
    "browser_unavailable": (
        "The browser isn't available right now.",
        "El navegador no está disponible ahora mismo.",
    ),
    "feature_disabled": (
        "That capability is turned off.",
        "Esa función está desactivada.",
    ),
    "cancelled": (
        "Cancelled.",
        "Cancelado.",
    ),
    "opened": (
        "Opened the page{title}.",
        "Abrí la página{title}.",
    ),
    "searched": (
        "I searched and opened the results{title}.",
        "Busqué y abrí los resultados{title}.",
    ),
    "summarized": (
        "Here's the page{title}.",
        "Aquí está la página{title}.",
    ),
    "found": (
        "I found {count} match(es).",
        "Encontré {count} coincidencia(s).",
    ),
    "not_found": (
        "I couldn't find that on the page.",
        "No encontré eso en la página.",
    ),
    "scrolled": (
        "Scrolled {direction}.",
        "Me desplacé hacia {direction}.",
    ),
    "closed": (
        "Closed the browser session.",
        "Cerré la sesión del navegador.",
    ),
    "nothing_pending": (
        "There's nothing pending to confirm or cancel.",
        "No hay nada pendiente que confirmar o cancelar.",
    ),
    "failed": (
        "That didn't work.",
        "Eso no funcionó.",
    ),
    "unsupported": (
        "I can't do that yet.",
        "Todavía no puedo hacer eso.",
    ),
}


def speak(situation: str, language: str = "es", **ctx) -> str:
    english, spanish = _TEMPLATES.get(situation, _TEMPLATES["failed"])
    template = english if str(language).lower().startswith("en") else spanish
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template
