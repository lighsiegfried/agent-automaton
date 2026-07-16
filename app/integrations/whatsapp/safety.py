"""Confirmation phrases, recipient extraction, and content safety for WhatsApp.

Two confirmation stages, each with EXACT phrases:
- place the draft into the composer: "colocar borrador" / "insertar mensaje" /
  "place draft";
- send it: "confirmar envío a <recipient>" / "enviar mensaje a <recipient>" /
  "confirm send to <recipient>" — the recipient MUST be named. A plain "sí",
  "confirmar", or the place phrase never sends.
"""

from __future__ import annotations

from app.text.secrets import contains_likely_secret  # reuse the secret detector

PLACE_PHRASES = frozenset({"colocar borrador", "insertar mensaje", "place draft"})
_SEND_PREFIXES = ("confirmar envío a ", "confirmar envio a ", "enviar mensaje a ",
                  "confirm send to ")


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_place_phrase(text: str) -> bool:
    return normalize(text) in PLACE_PHRASES


def is_send_phrase(text: str) -> bool:
    """True for a send phrase that NAMES a recipient (used by the broker matcher)."""
    n = normalize(text)
    return any(n.startswith(p) and len(n) > len(p) for p in _SEND_PREFIXES)


def extract_recipient(phrase: str) -> str | None:
    n = normalize(phrase)
    for prefix in _SEND_PREFIXES:
        if n.startswith(prefix):
            recipient = n[len(prefix):].strip()
            return recipient or None
    return None


def recipient_matches(phrase: str, expected_recipient: str) -> bool:
    """The recipient named in the send phrase must match the pending recipient."""
    named = extract_recipient(phrase)
    if not named or not expected_recipient:
        return False
    expected = normalize(expected_recipient)
    # Exact, or the expected name starts with the spoken recipient (first name).
    return named == expected or expected.startswith(named) or named.startswith(expected)


def check_message(text: str, max_chars: int) -> tuple[bool, str]:
    """(ok, reason) — reject secrets and over-length messages. Empty is invalid."""
    if not text or not text.strip():
        return False, "the message is empty"
    if len(text) > max_chars:
        return False, f"the message exceeds {max_chars} characters"
    is_secret, reasons = contains_likely_secret(text)
    if is_secret:
        return False, f"the message looks like it contains a secret ({', '.join(reasons)})"
    return True, ""
