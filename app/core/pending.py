"""Unified pending-action broker (Phase 4B.1).

Only ONE sensitive pending action may exist at a time across every domain (text
insertion, browser form fill, and — later — WhatsApp send). Creating a new one
supersedes the previous, calling its cancel callback so the other domain's
service discards its pending. Confirmation phrases are routed to the correct
domain: "confirmar escritura" can only confirm a text draft, "confirmar
formulario" only a browser form — a plain "sí" confirms nothing.

Domains register their confirmation phrases (or a matcher, for phrases that carry
a recipient) so new integrations extend routing without editing this file.
"""

from __future__ import annotations

from app.core import errors

# domain -> exact confirmation phrases / matcher / primary display phrase.
_DOMAIN_PHRASES: dict[str, frozenset[str]] = {}
_DOMAIN_MATCHERS: dict[str, object] = {}
_DOMAIN_PRIMARY: dict[str, str] = {}


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _emit_pending(event_type: str, status: str, domain: str, action_id: str, target: str) -> None:
    """Additive Activity Center hook (Phase 5D) — a pending action's lifecycle. Only
    the sub-domain and a safe target (an app name or recipient) are published."""
    try:
        from app.core import eventbus

        eventbus.emit(domain="pending", event_type=event_type, status=status,
                      related_id=action_id, title=f"Pending {domain} action",
                      metadata={"target": target})
    except Exception:
        pass


def register_domain(domain: str, phrases=(), primary: str = "", matcher=None) -> None:
    """Register a confirmation domain. ``matcher`` handles phrases that vary
    (e.g. a recipient in "confirmar envío a Ana")."""
    if phrases:
        _DOMAIN_PHRASES[domain] = frozenset(normalize(p) for p in phrases)
    if matcher is not None:
        _DOMAIN_MATCHERS[domain] = matcher
    _DOMAIN_PRIMARY[domain] = primary or (next(iter(phrases), "") if phrases else "")


def domain_accepts(domain: str, phrase: str) -> bool:
    """Whether ``phrase`` is a valid confirmation for a SPECIFIC domain.

    Used to route against the ACTIVE pending domain first, so integrations whose
    send phrases overlap (WhatsApp and email both use "confirmar envío a X") are
    disambiguated by which action is actually pending."""
    n = normalize(phrase)
    if not n:
        return False
    if domain in _DOMAIN_PHRASES and n in _DOMAIN_PHRASES[domain]:
        return True
    matcher = _DOMAIN_MATCHERS.get(domain)
    if matcher is not None:
        try:
            return bool(matcher(n))
        except Exception:
            return False
    return False


def phrase_domain(phrase: str) -> str | None:
    """Which domain a confirmation phrase belongs to, or None (e.g. a bare 'sí').

    A global best-effort match — used only to answer "is this a confirmation
    phrase at all?". Routing to a pending action uses ``domain_accepts`` on the
    active domain first (see ``route_confirmation``)."""
    n = normalize(phrase)
    if not n:
        return None
    for domain, phrases in _DOMAIN_PHRASES.items():
        if n in phrases:
            return domain
    for domain, matcher in _DOMAIN_MATCHERS.items():
        try:
            if matcher(n):
                return domain
        except Exception:
            continue
    return None


def primary_phrase(domain: str) -> str:
    return _DOMAIN_PRIMARY.get(domain, "")


def first_phrase(domain: str) -> str:
    """An EXACT confirmable phrase for a domain (not the human display string).

    Used by the Activity Center / desktop so a UI confirm button sends a real phrase.
    Deterministic (sorted) so the same phrase shows every time. Domains that vary by
    recipient (matcher-based sends) have no fixed phrase — the caller builds
    "confirmar envío a <recipient>" from the pending target instead."""
    phrases = _DOMAIN_PHRASES.get(domain)
    return sorted(phrases)[0] if phrases else ""


# Domains whose confirmation phrase varies by recipient (matcher-based sends).
_SEND_DOMAINS = ("email_send", "whatsapp_send")


def confirm_phrase_for(summary: dict | None) -> str:
    """The EXACT phrase a one-click UI should send to confirm a pending action.

    For recipient-sensitive sends this includes the recipient (e.g. "confirmar envío a
    Ana"), which the UI also shows; otherwise it is the domain's fixed phrase. Returns
    "" when nothing is pending or the domain has no confirmable phrase — NEVER a generic
    'yes'. One implementation shared by the Activity Center and the desktop bridge."""
    if not summary:
        return ""
    domain = summary.get("domain", "")
    if domain in _SEND_DOMAINS:
        target = (summary.get("target") or "").strip()
        return f"confirmar envío a {target}".strip()
    return first_phrase(domain)


class PendingBroker:
    """Holds at most one active pending sensitive action across all domains."""

    def __init__(self) -> None:
        self._active: dict | None = None

    def register(self, *, domain: str, action_id: str, target: str, expires_at: str,
                 cancel) -> dict:
        """Register a new pending action, superseding any prior one (cancelled)."""
        if self._active is not None and self._active["action_id"] != action_id:
            try:
                self._active["cancel"]()
            except Exception:
                pass
        self._active = {
            "domain": domain,
            "action_id": action_id,
            "target": target,
            "expires_at": expires_at,
            "status": errors.AWAITING_CONFIRMATION,
            "cancel": cancel,
        }
        _emit_pending("pending_registered", errors.AWAITING_CONFIRMATION, domain, action_id, target)
        return self.summary()

    def clear(self, action_id: str | None = None) -> None:
        if self._active is None:
            return
        if action_id is None or self._active["action_id"] == action_id:
            self._active = None

    def cancel_active(self) -> bool:
        """Cancel the active pending (calling its callback). Returns True if there was one."""
        if self._active is None:
            return False
        domain, action_id, target = (self._active["domain"], self._active["action_id"],
                                     self._active["target"])
        try:
            self._active["cancel"]()
        except Exception:
            pass
        self._active = None
        _emit_pending("pending_cancelled", "cancelled", domain, action_id, target)
        return True

    def active(self) -> dict | None:
        return self._active

    def active_domain(self) -> str | None:
        return self._active["domain"] if self._active else None

    def summary(self) -> dict | None:
        if self._active is None:
            return None
        domain = self._active["domain"]
        return {
            "domain": domain,
            "action_id": self._active["action_id"],
            "target": self._active["target"],
            "expires_at": self._active["expires_at"],
            "required_confirmation_phrase": primary_phrase(domain),
            "status": self._active["status"],
        }

    def route_confirmation(self, phrase: str) -> dict:
        """Route a confirmation phrase to the active pending's domain.

        {ok, domain?, action_id?, code?, reason} — a phrase for a different
        domain (or a bare 'sí') is a mismatch, never a silent confirmation.
        """
        active = self._active
        if active is None:
            return {"ok": False, "code": errors.CONFIRMATION_REQUIRED,
                    "reason": "there is no pending action to confirm"}
        # Route against the ACTIVE domain first (handles overlapping phrases).
        if domain_accepts(active["domain"], phrase):
            return {"ok": True, "domain": active["domain"], "action_id": active["action_id"]}
        other = phrase_domain(phrase)
        if other is None:
            return {"ok": False, "code": errors.CONFIRMATION_MISMATCH,
                    "reason": "not a recognised confirmation phrase "
                              "(a plain 'sí' never confirms a sensitive action)"}
        return {"ok": False, "code": errors.CONFIRMATION_MISMATCH,
                "domain": active["domain"], "reason":
                f"that phrase confirms {other!r}, but the pending action is "
                f"{active['domain']!r}"}


# Built-in domains (text + browser). 4C registers its WhatsApp domains similarly.
register_domain("text",
                phrases=("confirmar escritura", "escribirlo", "insert text", "confirm typing"),
                primary="confirmar escritura / insert text")
register_domain("browser",
                phrases=("confirmar formulario", "llenar formulario", "confirm form fill"),
                primary="confirmar formulario / confirm form fill")


_broker: PendingBroker | None = None


def get_pending_broker() -> PendingBroker:
    global _broker
    if _broker is None:
        _broker = PendingBroker()
    return _broker
