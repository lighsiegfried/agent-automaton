"""Interpret WhatsApp session reads: login, contact matching, chat/composer state.

Pure helpers over already-extracted, SAFE data (visible names, titles, composer
text) — never cookies, tokens, or QR content. Contact matching NEVER auto-selects
a fuzzy result: only an exact visible name (or a verified phone) resolves.
"""

from __future__ import annotations

import hashlib


def chat_signature(title: str) -> str:
    """Stable identity for the active chat; a changed chat changes this."""
    return hashlib.sha256(("chat:" + (title or "")).encode("utf-8")).hexdigest()[:16]


def composer_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _looks_like_phone(query: str) -> bool:
    digits = "".join(ch for ch in query if ch.isdigit())
    return query.strip().startswith("+") and len(digits) >= 8


def match_contacts(candidates: list[str], query: str) -> dict:
    """Resolve a contact from visible candidates.

    Returns {status: "ok"|"ambiguous"|"not_found", contact?, candidates?}. An
    exact visible-name (or verified-phone) match with exactly one result resolves;
    anything else is ambiguous or not found — never a silent first-fuzzy pick.
    """
    q = (query or "").strip().lower()
    if not q:
        return {"status": "not_found", "candidates": []}

    exact = [c for c in candidates if c.strip().lower() == q]
    if _looks_like_phone(query):
        exact = [c for c in candidates if "".join(ch for ch in c if ch.isdigit())
                 == "".join(ch for ch in query if ch.isdigit())] or exact
    if len(exact) == 1:
        return {"status": "ok", "contact": exact[0]}
    if len(exact) > 1:
        return {"status": "ambiguous", "candidates": exact}

    partial = [c for c in candidates if q in c.strip().lower()]
    if not partial:
        return {"status": "not_found", "candidates": []}
    # Never auto-select a fuzzy result — surface the candidates for the user.
    return {"status": "ambiguous", "candidates": partial}
