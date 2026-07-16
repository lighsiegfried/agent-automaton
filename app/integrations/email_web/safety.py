"""Email safety: confirmation phrases, recipient rules, and untrusted-content checks.

Two confirmation stages (place + send). A send must NAME every recipient. Reply-all
and BCC are blocked in this phase; attachments must be zero. Email content is
untrusted — ``detect_prompt_injection`` flags in-email attempts to change safety,
reveal secrets, or trigger actions (which are never executed regardless).
"""

from __future__ import annotations

import re

from app.text.secrets import contains_likely_secret

PLACE_PHRASES = frozenset({"colocar borrador", "insertar correo", "place email draft"})
_SEND_PREFIXES = ("confirmar envío a ", "confirmar envio a ", "enviar correo a ",
                  "confirm email to ")

_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(?:disregard|forget)\s+(?:all\s+)?(?:previous|prior|your)\b", re.I),
    re.compile(r"\bignora\s+(?:las\s+)?instrucciones\b", re.I),
    re.compile(r"\breveal\s+(?:your\s+)?(?:system\s+prompt|instructions|secret|password)\b", re.I),
    re.compile(r"\b(?:disable|turn\s+off|bypass)\s+(?:the\s+)?(?:safety|security|confirmation)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b|\bnew\s+instructions?\b", re.I),
    re.compile(r"\b(?:wire|transfer|send)\s+(?:money|\$|the\s+funds|payment)\b", re.I),
    re.compile(r"\b(?:open|click)\s+(?:the\s+)?(?:link|attachment|url)\b", re.I),
    re.compile(r"\bdownload\s+(?:the\s+)?(?:file|attachment)\b", re.I),
    re.compile(r"\bforward\s+this\s+to\b", re.I),
]


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_place_phrase(text: str) -> bool:
    return normalize(text) in PLACE_PHRASES


def is_send_phrase(text: str) -> bool:
    n = normalize(text)
    return any(n.startswith(p) and len(n) > len(p) for p in _SEND_PREFIXES)


def extract_recipients(phrase: str) -> list[str]:
    n = normalize(phrase)
    for prefix in _SEND_PREFIXES:
        if n.startswith(prefix):
            rest = n[len(prefix):]
            parts = re.split(r"\s*(?:,| y | and )\s*", rest)
            return [p.strip() for p in parts if p.strip()]
    return []


def _recipient_token(recipient: str) -> str:
    """A comparable token: the local part of an address, or the lowered name."""
    r = recipient.strip().lower()
    return r.split("@")[0] if "@" in r else r


def recipients_match(phrase: str, expected: list[str]) -> bool:
    """Every expected recipient must be named in the confirmation phrase."""
    if not expected:
        return False
    named = " ".join(extract_recipients(phrase))
    if not named:
        return False
    return all(_recipient_token(r) in named for r in expected)


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """Flag likely in-email instructions. Content is never executed regardless."""
    hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text or "")]
    return (bool(hits), hits[:3])


def domain_of(address: str) -> str:
    return address.split("@")[-1].strip().lower() if "@" in address else ""


def external_recipients(recipients: list[str], own_domains: list[str]) -> list[str]:
    own = {d.strip().lower() for d in own_domains if d}
    return [r for r in recipients if domain_of(r) and domain_of(r) not in own]


def check_body(text: str, max_chars: int) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "the email body is empty"
    if len(text) > max_chars:
        return False, f"the body exceeds {max_chars} characters"
    is_secret, reasons = contains_likely_secret(text)
    if is_secret:
        return False, f"the body looks like it contains a secret ({', '.join(reasons)})"
    return True, ""


def check_send_compose(compose: dict, settings) -> tuple[bool, str, str]:
    """(ok, error_code, reason) for a compose state immediately before send."""
    from app.core import errors

    to = compose.get("to") or []
    cc = compose.get("cc") or []
    bcc = compose.get("bcc") or []
    if compose.get("attachment_count", 0) and not settings.email_allow_attachments:
        return False, errors.ATTACHMENT_BLOCKED, "attachments are blocked in this phase"
    if bcc and not settings.email_allow_bcc:
        return False, errors.SEND_BLOCKED, "BCC is blocked in this phase"
    if len(to) + len(cc) > settings.email_max_recipients:
        return False, errors.SEND_BLOCKED, "too many recipients"
    # Reply-all (more than one recipient on a reply) is blocked this phase.
    if compose.get("mode") == "reply" and (len(to) + len(cc)) > 1:
        return False, errors.SEND_BLOCKED, "reply-all is blocked in this phase"
    if not to:
        return False, errors.SEND_BLOCKED, "no recipient"
    return True, "", ""
