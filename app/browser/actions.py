"""Pure helpers for browser actions: search URLs, scrolling, and form plans.

Kept separate from the session/service so the form-plan construction (which
decides what gets filled and what is skipped as sensitive) is testable on its own.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from app.browser import safety

SEARCH_ENDPOINT = "https://duckduckgo.com/?q="


def search_url(query: str) -> str:
    return SEARCH_ENDPOINT + quote_plus((query or "").strip())


def scroll_delta(direction: str, amount: int = 600) -> int:
    return {"up": -amount, "down": amount, "top": -100000, "bottom": 100000}.get(
        (direction or "down").lower(), amount
    )


def page_signature(url: str, title: str) -> str:
    """Identity used to detect the page changed between prepare and confirm."""
    return f"{url}||{title}"


def build_form_plan(
    snapshot: dict,
    requested_fields: dict,
    *,
    action_id: str,
    expires_at_iso: str,
) -> dict:
    """Build a pending form-fill plan from a snapshot + requested {name: value}.

    Fillable, non-sensitive controls are included; sensitive controls (password,
    payment, OTP, secret, identity, auth) are recorded in
    ``skipped_sensitive_fields`` and NEVER filled; unknown fields are skipped.
    """
    controls = {c["name"]: c for c in snapshot.get("form_controls", []) if c.get("name")}
    fields, skipped_sensitive, skipped_unknown = [], [], []
    for name, value in (requested_fields or {}).items():
        control = controls.get(name)
        if control is None:
            skipped_unknown.append(name)
            continue
        if not control.get("fillable", False):
            skipped_sensitive.append({"name": name,
                                      "reason": control.get("sensitive_reason") or "sensitive"})
            continue
        fields.append({"name": name, "label": control.get("label", ""), "value": value})
    return {
        "action_id": action_id,
        "page_url": snapshot.get("url", ""),
        "page_title": snapshot.get("title", ""),
        "fields": fields,
        "skipped_sensitive_fields": skipped_sensitive,
        "skipped_unknown_fields": skipped_unknown,
        "expires_at": expires_at_iso,
        "requires_confirmation": True,
    }


def is_sensitive_requested(snapshot: dict, requested_fields: dict) -> bool:
    """True if any requested field maps to a sensitive control."""
    controls = {c["name"]: c for c in snapshot.get("form_controls", []) if c.get("name")}
    return any(
        (controls.get(name) or {}).get("fillable") is False for name in (requested_fields or {})
    )
