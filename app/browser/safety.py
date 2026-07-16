"""URL policy + form-field safety + confirmation phrases for browser automation.

Pure and stdlib-only, so every protection is testable without Playwright:
- normalize + validate every URL before navigation (scheme allowlist, blocked
  schemes, private-network blocking);
- decide which form fields are too sensitive to ever fill (passwords, payment,
  OTP/2FA, API keys, identity documents, auth);
- match the exact form-fill confirmation phrases.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")

# Schemes we must never navigate to.
BLOCKED_SCHEMES = frozenset({
    "file", "javascript", "data", "chrome", "edge", "about", "view-source",
    "blob", "ftp", "ws", "wss", "chrome-extension", "vbscript",
})

# Actions that are blocked entirely in this phase (reference for the planner).
BLOCKED_ACTIONS = frozenset({
    "submit", "purchase", "buy", "checkout", "pay", "publish", "post", "upload",
    "delete", "remove", "accept_terms", "agree", "send", "login", "sign_in",
    "sign_up", "register", "logout",
})

# The ONLY phrases that authorize filling a prepared form (item 7).
CONFIRMATION_PHRASES = frozenset({
    "confirmar formulario", "llenar formulario", "confirm form fill",
})

# Substrings that mark a field as too sensitive to ever auto-fill.
_SENSITIVE_HINTS = (
    "password", "passwd", "contraseña", "contrasena", "clave",
    "card", "tarjeta", "cc-number", "cardnumber", "creditcard", "cvv", "cvc",
    "cc-csc", "security code", "iban", "account number", "routing", "swift",
    "ssn", "social security", "otp", "one-time", "one time", "2fa", "mfa",
    "verification code", "codigo de verificacion", "código de verificación",
    "api key", "api-key", "apikey", "token", "secret", "private key",
    "passport", "pasaporte", "dni", "curp", "rfc", "nif", "licencia", "license",
    "date of birth", "fecha de nacimiento", "birthdate",
)
_SENSITIVE_AUTOCOMPLETE = frozenset({
    "current-password", "new-password", "cc-number", "cc-csc", "cc-exp",
    "cc-exp-month", "cc-exp-year", "one-time-code",
})
_SENSITIVE_TYPES = frozenset({"password"})


_KNOWN_SCHEMES = BLOCKED_SCHEMES | {"http", "https", "mailto", "tel"}


def normalize_url(raw: str) -> str:
    """Trim, add https:// when no scheme is present, and lower scheme/host.

    A scheme-only URL (``javascript:``, ``data:…``, ``about:blank``) is left
    intact so the scheme allowlist can reject it — we only prepend https:// when
    there is genuinely no scheme (e.g. ``example.com/path``).
    """
    text = (raw or "").strip()
    if not text:
        return ""
    match = _SCHEME_RE.match(text)
    has_scheme = bool(match) and (
        text[match.end():].startswith("//") or match.group(1).lower() in _KNOWN_SCHEMES
    )
    if not has_scheme:
        text = "https://" + text
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


def _hostname(netloc: str) -> str:
    host = netloc.split("@")[-1]           # strip any userinfo
    if host.startswith("["):               # IPv6 literal [::1]:port
        return host[1:].split("]")[0]
    return host.split(":")[0]


def is_private_host(host: str) -> bool:
    """True for loopback / private / link-local / reserved hosts and *.local."""
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in ("localhost",) or host.endswith((".local", ".localhost", ".internal", ".home", ".lan")):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_unspecified or ip.is_multicast)
    except ValueError:
        return False


def validate_url(raw: str, allowed_schemes: list[str], block_private: bool = True) -> dict:
    """Return ``{ok, url, scheme, host, reason}`` after normalising ``raw``."""
    url = normalize_url(raw)
    if not url:
        return {"ok": False, "url": "", "reason": "empty URL"}
    parts = urlsplit(url)
    scheme, host = parts.scheme.lower(), _hostname(parts.netloc)
    if scheme in BLOCKED_SCHEMES:
        return {"ok": False, "url": url, "scheme": scheme, "reason": f"blocked scheme: {scheme}"}
    if scheme not in {s.lower() for s in allowed_schemes}:
        return {"ok": False, "url": url, "scheme": scheme,
                "reason": f"scheme {scheme!r} is not in the allowlist"}
    if not host:
        return {"ok": False, "url": url, "scheme": scheme, "reason": "missing host"}
    if block_private and is_private_host(host):
        return {"ok": False, "url": url, "scheme": scheme, "host": host,
                "reason": "private/loopback network is blocked"}
    return {"ok": True, "url": url, "scheme": scheme, "host": host, "reason": ""}


def is_sensitive_field(name: str = "", field_type: str = "", label: str = "",
                       autocomplete: str = "") -> tuple[bool, str]:
    """Whether a form field must never be auto-filled, and why."""
    if (field_type or "").lower() in _SENSITIVE_TYPES:
        return True, "password field"
    if (autocomplete or "").lower() in _SENSITIVE_AUTOCOMPLETE:
        return True, f"sensitive autocomplete: {autocomplete}"
    haystack = " ".join((name, field_type, label, autocomplete)).lower()
    spaced = haystack.replace("_", " ").replace("-", " ")   # api_key -> "api key"
    for hint in _SENSITIVE_HINTS:
        if hint in haystack or hint in spaced:
            return True, f"sensitive field ({hint})"
    return False, ""


def is_confirmation_phrase(text: str) -> bool:
    return " ".join((text or "").strip().lower().split()) in CONFIRMATION_PHRASES
