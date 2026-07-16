"""Consent phrases + sensitive-data blocking + deterministic content parsing (5A).

Two independent guarantees live here:

1. **Explicit consent** — the ONLY phrases that authorize a write/forget. A create
   is proposed by an explicit intent ("recuerda que…") and committed only by an
   exact confirmation phrase; forgetting is two-staged (soft, then a separate
   permanent phrase). A plain "sí" is deliberately never enough.
2. **Sensitive-data blocking** — content that looks like a password, OTP, API key,
   payment card, bank/IBAN, private key, session cookie, precise address, or
   explicit sensitive personal information (medical/political) is refused before it
   can ever be stored. Sensitive attributes are never *inferred* — Fifi only refuses
   to store content that plainly contains them.

Everything here is pure and stdlib-only, so the whole gate is testable without a
database or any UI.
"""

from __future__ import annotations

import re

from app.memory.models import MEMORY_TYPES, MemoryType, normalize
from app.text.secrets import contains_likely_secret

# --- consent / confirmation phrases (item 3, 8) ------------------------------------

# Explicit intents that PROPOSE a memory (never auto-store — they create a pending
# proposal that still needs a confirmation phrase). Used by the NL layer.
PROPOSE_PHRASES = frozenset({
    "recuerda que", "recuérdame que", "recuerdame que", "guarda esto", "guarda que",
    "apunta que", "toma nota de que", "remember that", "remember", "save this",
    "take note that", "note that",
})

# The ONLY phrases that COMMIT a create/update.
WRITE_CONFIRM_PHRASES = frozenset({
    "confirmar memoria", "guardar memoria", "confirm memory", "save memory",
})
# Soft forget (exclude from retrieval).
FORGET_CONFIRM_PHRASES = frozenset({
    "confirmar olvido", "confirm forget",
})
# Permanent delete — a separate, stronger confirmation.
DELETE_CONFIRM_PHRASES = frozenset({
    "eliminar memoria permanentemente", "delete memory permanently",
    "borrar memoria permanentemente",
})


def is_write_confirm(text: str) -> bool:
    return normalize(text) in WRITE_CONFIRM_PHRASES


def is_forget_confirm(text: str) -> bool:
    return normalize(text) in FORGET_CONFIRM_PHRASES


def is_delete_confirm(text: str) -> bool:
    return normalize(text) in DELETE_CONFIRM_PHRASES


# --- sensitive-data blocking (item 4) ----------------------------------------------

# Extra categories on top of app.text.secrets.contains_likely_secret (which already
# covers api keys/tokens, cards, private-key blocks, and password/OTP labels using
# ':' or '='). Spoken memory tends to use a copula ("mi contraseña ES hunter2"),
# which the label-with-colon pattern misses — so catch that form explicitly here.
_CREDENTIAL_COPULA = re.compile(
    r"\b(contrase[nñ]a|password|passwd|clave|pin|otp|"
    r"c[oó]digo\s+de\s+verificaci[oó]n|one[-\s]?time\s+password|verification\s+code)\b"
    r"\s*(?:es|son|is|are|:|=)\s*\S+",
    re.IGNORECASE,
)
_SESSION_COOKIE = re.compile(
    r"\b(set-cookie|session[_-]?id|sessionid|phpsessid|jsessionid|"
    r"csrf[_-]?token|xsrf[_-]?token|auth[_-]?token|access[_-]?token|"
    r"refresh[_-]?token|sid)\b\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_BANK_LABEL = re.compile(
    r"\b(iban|swift|bic|clabe|routing\s*number|account\s*number|"
    r"n[uú]mero\s+de\s+cuenta|cuenta\s+bancaria|tarjeta\s+de\s+cr[eé]dito)\b"
    r"\s*[:#=]?\s*[\dA-Z\-\s]{6,}",
    re.IGNORECASE,
)
# A precise street address needs BOTH a house number AND a street-type keyword in
# close proximity (kept strict to avoid blocking ordinary facts like "on 5th St").
_STREET_TYPES = (r"calle|avenida|avda|carrera|cra|autopista|bulevar|boulevard|blvd|"
                 r"street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|plaza")
_ADDRESS = re.compile(
    rf"(?:\b\d{{1,6}}\b[\w\s.,#-]{{0,30}}?\b(?:{_STREET_TYPES})\b"
    rf"|\b(?:{_STREET_TYPES})\b[\w\s.,#-]{{0,30}}?\b\d{{1,6}}\b)"
    rf"[\w\s.,#-]{{0,40}}?\b\d{{4,6}}\b",   # …followed by a postal code
    re.IGNORECASE,
)
_ADDRESS_LABELED = re.compile(
    r"\b(direcci[oó]n|domicilio|home\s+address|address)\b\s*[:=]\s*\S+.*\b\d{1,6}\b",
    re.IGNORECASE,
)
_MEDICAL = re.compile(
    r"\b(diagnostic[ao]d[oa]\s+con|tengo\s+(c[aá]ncer|diabetes|vih|hiv|sida)|"
    r"mi\s+medicaci[oó]n|estoy\s+tomando\s+\w+\s+para|blood\s+type|"
    r"tipo\s+de\s+sangre|prescrib(ed|ió|io))\b",
    re.IGNORECASE,
)
_POLITICAL = re.compile(
    r"\b(voto\s+por|milito\s+en|afiliaci[oó]n\s+pol[ií]tica|soy\s+(de\s+)?(izquierda|derecha)|"
    r"i\s+vote(d)?\s+for|my\s+political\s+party|card-?carrying\s+member)\b",
    re.IGNORECASE,
)

_CATEGORY_CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("credentials", _CREDENTIAL_COPULA),
    ("session_cookie", _SESSION_COOKIE),
    ("banking", _IBAN),
    ("banking", _BANK_LABEL),
    ("precise_address", _ADDRESS),
    ("precise_address", _ADDRESS_LABELED),
    ("sensitive_personal_medical", _MEDICAL),
    ("sensitive_personal_political", _POLITICAL),
]

# The blocked categories, for docs/status/audit (never the offending content).
BLOCKED_CATEGORIES = (
    "passwords", "otps", "api_keys", "tokens", "payment_cards", "banking",
    "private_keys", "session_cookies", "auth_secrets", "precise_addresses",
    "sensitive_personal_medical", "sensitive_personal_political",
)


def check_storable(title: str, content: str) -> tuple[bool, str, list[str]]:
    """(ok, category, reasons). ``ok`` is False when the content must not be stored.

    Conservative and explicit: it flags content that plainly contains a secret or
    sensitive datum. It never *infers* a sensitive attribute about the user.
    """
    text = f"{title}\n{content}".strip()
    if not text:
        return False, "empty", ["nothing to store"]
    is_secret, reasons = contains_likely_secret(text)
    if is_secret:
        return False, "secret", reasons
    for category, pattern in _CATEGORY_CHECKS:
        if pattern.search(text):
            return False, category, [category]
    return True, "", []


# --- deterministic content parsing (for the voice/text "recuerda que…" flow) -------
# The API accepts explicit type/title/subject/entities/tags; these helpers only
# fill in sensible defaults when a spoken command supplies just free-form content.

_TYPE_HINTS: list[tuple[MemoryType, re.Pattern[str]]] = [
    (MemoryType.INSTRUCTION, re.compile(
        r"\b(siempre|nunca|no\s+(uses|hagas|abras)|cuando\s+yo|recu[ée]rdame\s+siempre|"
        r"always|never|whenever\s+i|remind\s+me\s+to)\b", re.IGNORECASE)),
    (MemoryType.PREFERENCE, re.compile(
        r"\b(prefiero|preferid[oa]|favorit[oa]|me\s+gusta|prefer|preferred|favou?rite|i\s+like)\b",
        re.IGNORECASE)),
    (MemoryType.DECISION, re.compile(
        r"\b(decid[íi]|decidimos|elegimos|eleg[íi]|vamos\s+a\s+usar|we\s+decided|"
        r"we'?ll\s+use|chose|decision)\b", re.IGNORECASE)),
    (MemoryType.WORKFLOW, re.compile(
        r"\b(flujo|workflow|proceso|pasos\s+para|c[oó]mo\s+(despliego|hago|construyo)|"
        r"steps\s+to|how\s+i\s+(deploy|build|run)|deploy)\b", re.IGNORECASE)),
    (MemoryType.PROJECT, re.compile(
        r"\b(proyecto|project|repositorio|repo\b|codebase)\b", re.IGNORECASE)),
    (MemoryType.PERSON, re.compile(
        r"\b(es\s+mi\s+(amig[oa]|herman[oa]|jef[ea]|colega|pareja|madre|padre)|"
        r"su\s+(cumplea[nñ]os|correo|tel[eé]fono)|is\s+my\s+(friend|boss|colleague|brother|sister)|"
        r"'s\s+birthday)\b", re.IGNORECASE)),
]

_COPULA = re.compile(r"^(.*?)\s+(?:es|son|is|are|=)\s+(.+)$", re.IGNORECASE)
_LEAD_POSSESSIVE = re.compile(r"^(?:mi|mis|my|el|la|los|las|the|un|una)\s+", re.IGNORECASE)


def infer_type(content: str) -> str:
    for mem_type, pattern in _TYPE_HINTS:
        if pattern.search(content or ""):
            return mem_type.value
    return MemoryType.FACT.value


def infer_subject(content: str) -> str:
    """A normalized property key for conflict detection.

    "mi editor preferido es VS Code" → "editor preferido" (so a later "…es Neovim"
    conflicts). Returns "" when there is no copula to key on."""
    match = _COPULA.match((content or "").strip().rstrip("."))
    if not match:
        return ""
    left = _LEAD_POSSESSIVE.sub("", match.group(1).strip())
    return normalize(left)


def infer_title(content: str, subject: str = "") -> str:
    if subject:
        return subject[:1].upper() + subject[1:]
    text = " ".join((content or "").split())
    if len(text) <= 60:
        return text
    cut = text[:60].rsplit(" ", 1)[0]
    return (cut or text[:60]) + "…"


_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9.+#-]{2,}")


def infer_entities(content: str) -> list[str]:
    """Best-effort normalized entities/tags: the value after a copula plus any
    capitalized tokens (product/proper names). De-duplicated, order-preserving."""
    entities: list[str] = []
    match = _COPULA.match((content or "").strip().rstrip("."))
    if match:
        value = match.group(2).strip().rstrip(".")
        if value:
            entities.append(normalize(value))
    for token in _WORD.findall(content or ""):
        if token[0].isupper() and normalize(token) not in entities:
            entities.append(normalize(token))
    # De-dupe while preserving order; drop empties.
    seen, out = set(), []
    for e in entities:
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out[:8]


def valid_type(mem_type: str) -> bool:
    return mem_type in MEMORY_TYPES
