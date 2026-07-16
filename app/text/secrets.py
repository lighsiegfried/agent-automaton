"""Confirmation-phrase matching + likely-secret / password-field detection.

Real text insertion must never type secrets and requires an EXACT confirmation
phrase — a plain "sí" is deliberately not enough. These helpers are pure and
stdlib-only, so the whole gate is testable without any UI.
"""

from __future__ import annotations

import re

# The ONLY phrases that authorize a real insertion (item 4). Case/space
# insensitive, but a normal "sí"/"yes" never matches.
CONFIRMATION_PHRASES = frozenset({
    "confirmar escritura",
    "escribirlo",
    "insert text",
    "confirm typing",
})


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_confirmation_phrase(text: str) -> bool:
    return normalize(text) in CONFIRMATION_PHRASES


# --- secret detection --------------------------------------------------------------

_TOKEN_PREFIXES = ("sk-", "sk_live_", "sk_test_", "ghp_", "gho_", "xoxb-", "xoxp-",
                   "AKIA", "ASIA", "AIza")

_PATTERNS = [
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    ("password_label", re.compile(
        r"\b(pass(word)?|contrase[nñ]a|clave|pin|otp|c[oó]digo\s+de\s+verificaci[oó]n)"
        r"\s*[:=]\s*\S+", re.IGNORECASE)),
    ("high_entropy_token", re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")),
]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _looks_like_card(text: str) -> bool:
    for match in re.findall(r"\b(?:\d[ -]?){13,19}\b", text):
        digits = re.sub(r"\D", "", match)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def contains_likely_secret(text: str) -> tuple[bool, list[str]]:
    """(is_secret, reasons). Conservative: flags cards, keys, tokens, credentials."""
    reasons: list[str] = []
    if not text:
        return False, reasons
    lowered = text
    if any(prefix in lowered for prefix in _TOKEN_PREFIXES):
        reasons.append("api_key_prefix")
    if _looks_like_card(text):
        reasons.append("payment_card_number")
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            reasons.append(name)
    # De-dupe while keeping order.
    seen, unique = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return (bool(unique), unique)


# Control types / hints that indicate a password or otherwise secret input.
_PASSWORD_HINTS = ("password", "passwd", "contraseña", "contrasena", "otp", "pin",
                   "secret", "clave")


def looks_like_password_field(control_type: str, is_password_flag: bool,
                              field_name: str = "") -> bool:
    if is_password_flag:
        return True
    haystack = f"{control_type} {field_name}".lower()
    return any(hint in haystack for hint in _PASSWORD_HINTS)
