"""Shape a SAFE page snapshot and resolve element targets unambiguously.

The snapshot contains only what the LLM needs to reason about a page — URL,
title, headings, visible text (capped), visible links/buttons, form controls
(names/labels/types, NEVER values), and accessibility labels. It never contains
raw HTML, scripts, hidden fields, cookies, localStorage, tokens, or a DOM dump.

Element targeting prefers accessible role + label + visible text; an ambiguous
target (more than one match) is rejected rather than guessed.
"""

from __future__ import annotations

from app.browser import safety

MAX_LINKS = 200
MAX_BUTTONS = 100
MAX_HEADINGS = 100
MAX_FORM_CONTROLS = 100


def build_snapshot(raw: dict, max_chars: int) -> dict:
    """Turn raw extracted data into the safe, capped snapshot.

    ``raw`` is produced by our own in-page extractor and already excludes
    scripts/cookies/storage; this shapes and truncates it and flags which form
    controls are safe to fill. Hidden inputs are dropped entirely.
    """
    text = (raw.get("text") or "")[:max_chars]
    headings = [
        {"level": h.get("level", 2), "text": (h.get("text") or "").strip()}
        for h in (raw.get("headings") or [])[:MAX_HEADINGS]
        if (h.get("text") or "").strip()
    ]
    links = [
        {"text": (link.get("text") or "").strip(), "href": link.get("href") or ""}
        for link in (raw.get("links") or [])[:MAX_LINKS]
        if (link.get("text") or "").strip() and link.get("href")
    ]
    buttons = [
        {"text": (b.get("text") or "").strip(), "role": b.get("role") or "button"}
        for b in (raw.get("buttons") or [])[:MAX_BUTTONS]
        if (b.get("text") or "").strip()
    ]
    form_controls = []
    for control in (raw.get("form_controls") or [])[:MAX_FORM_CONTROLS]:
        ctype = (control.get("type") or "").lower()
        if ctype == "hidden":
            continue  # never surface hidden fields
        sensitive, reason = safety.is_sensitive_field(
            name=control.get("name", ""), field_type=ctype,
            label=control.get("label", ""), autocomplete=control.get("autocomplete", ""),
        )
        form_controls.append({
            "name": control.get("name", ""),
            "label": (control.get("label") or "").strip(),
            "type": ctype,
            "role": control.get("role", ""),
            "autocomplete": control.get("autocomplete", ""),
            # NEVER include the control's current value.
            "fillable": not sensitive,
            "sensitive_reason": reason,
        })
    return {
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "headings": headings,
        "visible_text": text,
        "text_truncated": len(raw.get("text") or "") > max_chars,
        "links": links,
        "buttons": buttons,
        "form_controls": form_controls,
        "landmarks": raw.get("landmarks") or [],
    }


def find_text(snapshot: dict, query: str) -> dict:
    """Locate a visible section/heading/link by text (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q:
        return {"found": False, "reason": "empty query"}
    matches = []
    for heading in snapshot.get("headings", []):
        if q in heading["text"].lower():
            matches.append({"kind": "heading", "text": heading["text"]})
    for link in snapshot.get("links", []):
        if q in link["text"].lower():
            matches.append({"kind": "link", "text": link["text"], "href": link["href"]})
    if q in (snapshot.get("visible_text") or "").lower():
        matches.append({"kind": "text", "text": query})
    return {"found": bool(matches), "matches": matches, "count": len(matches)}


def resolve_link(snapshot: dict, *, text: str) -> dict:
    """Resolve a link by visible text. Ambiguous (>1) is rejected."""
    q = (text or "").strip().lower()
    if not q:
        return {"ok": False, "reason": "empty link text"}
    matches = [link for link in snapshot.get("links", []) if q == link["text"].lower()]
    if not matches:                              # fall back to substring
        matches = [link for link in snapshot.get("links", []) if q in link["text"].lower()]
    if not matches:
        return {"ok": False, "reason": f"no link matching {text!r}"}
    if len(matches) > 1:
        return {"ok": False, "reason": f"ambiguous link {text!r} ({len(matches)} matches)",
                "count": len(matches)}
    return {"ok": True, "link": matches[0], "confidence": 1.0}


def resolve_target(snapshot: dict, *, role: str = "", label: str = "", text: str = "") -> dict:
    """Resolve an actionable element by role/label/text. Ambiguous is rejected.

    Returns ``{ok, element, confidence}`` or ``{ok: False, reason, count}``.
    """
    role_l, label_l, text_l = role.lower(), label.lower(), text.lower()
    if not any((role_l, label_l, text_l)):
        return {"ok": False, "reason": "no target descriptor (role/label/text) given"}

    candidates = list(snapshot.get("buttons", [])) + [
        {"text": c.get("label", ""), "role": c.get("role", ""), "name": c.get("name", ""),
         "control": True}
        for c in snapshot.get("form_controls", [])
    ]
    matches = []
    for element in candidates:
        etext = (element.get("text") or "").lower()
        erole = (element.get("role") or "").lower()
        if role_l and erole != role_l:
            continue
        if label_l and label_l not in etext and label_l not in (element.get("name") or "").lower():
            continue
        if text_l and text_l not in etext:
            continue
        matches.append(element)
    if not matches:
        return {"ok": False, "reason": "no matching element", "count": 0}
    if len(matches) > 1:
        return {"ok": False, "reason": "ambiguous target (multiple matches)",
                "count": len(matches)}
    return {"ok": True, "element": matches[0], "confidence": 1.0}
