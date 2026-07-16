"""SecurityService: sessions, elevation, the central authorization gate (Phase 6B).

Ties the profile matrix, session manager, and central ``authorize`` together, installs
the deterministic dispatch gate (every service command is authorized before work), and
exposes lock/unlock/elevation/profile-change flows. Changing the base profile is
sensitive and confirmed through the pending broker ("confirmar cambio de perfil"); a
plain "sí" or a wake never confirms. Nothing here weakens an existing domain
confirmation — the invariants live in the domain services and are never bypassed.

Importing this module registers the broker domain and installs the gate.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.logger import get_logger
from app.security import audit as audit_mod
from app.security import permissions, policy
from app.security.profiles import LOCKED, PROFILE_SUMMARY, PROFILES, is_profile
from app.security.repository import SecurityRepository
from app.security.sessions import SessionManager

log = get_logger(__name__)

PROFILE_DOMAIN = "security_profile"
PROFILE_PHRASES = frozenset({"confirmar cambio de perfil", "confirm profile change"})


def _emit(event_type, *, severity="info", status="", metadata=None):
    try:
        from app.core import eventbus
        eventbus.emit(domain="settings", event_type=event_type, severity=severity, status=status,
                      title="Security", metadata=metadata or {})
    except Exception:
        pass


def _r(status, *, state, spoken="", code=None, action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": "security", "action_id": action_id,
            "state": state, "spoken": spoken, "data": data or {}}


class SecurityService:
    def __init__(self, *, repository=None, db_path=None, now_fn=None, clock=time.monotonic,
                 hello_probe=None, settings_provider=get_settings):
        self._settings = settings_provider
        self.repo = repository or SecurityRepository(
            db_path or settings_provider().security_db_path, now_fn=now_fn)
        self.sessions = SessionManager(self.repo, settings_provider=settings_provider,
                                       now_fn=now_fn, clock=clock, hello_probe=hello_probe)
        self._clock = clock
        self._pending_profile = None       # (action_id, to_profile, expires_monotonic)

    # -- read ----------------------------------------------------------------------

    def status(self, *args) -> dict:
        snap = self.sessions.snapshot().public()
        snap["enabled"] = self._settings().enable_security_profiles
        snap["profile_summary"] = PROFILE_SUMMARY.get(snap["profile"], "")
        return _r("ok", state=errors.COMPLETED, data=snap)

    def profile(self, *args) -> dict:
        prof = self.sessions.effective_profile()
        return _r("ok", state=errors.COMPLETED, data={
            "profile": prof, "base_profile": self.sessions.base_profile(),
            "summary": PROFILE_SUMMARY.get(prof, ""), "profiles": list(PROFILES)})

    def permissions_view(self) -> dict:
        """The effective permission matrix for the current profile (deterministic)."""
        settings = self._settings()
        prof = self.sessions.effective_profile()
        out = {}
        for intent, (domain, effect, scope) in permissions.INTENT_POLICY.items():
            d = policy.authorize(self.sessions.owner, self.sessions.effective_profile(domain),
                                 domain, intent, effect, resource=scope, settings=settings)
            out[intent] = {"domain": domain, "effect": effect, "allowed": d.allowed,
                           "elevation_required": d.elevation_required,
                           "confirmation_required": d.confirmation_required}
        return _r("ok", state=errors.COMPLETED, data={"profile": prof, "permissions": out})

    def audit(self, *, limit=100) -> list[dict]:
        return audit_mod.safe_events(self.repo.audit_recent(self.sessions.owner, limit=limit))

    # -- unlock / lock -------------------------------------------------------------

    def set_password(self, password: str) -> dict:
        if not password:
            return _r("rejected", state=errors.BLOCKED, code=errors.UNLOCK_FAILED)
        self.sessions.set_password(password)
        return _r("ok", state=errors.COMPLETED, data={"verifier_set": True})

    def unlock(self, method: str, secret: str) -> dict:
        result = self.sessions.unlock(method=method, secret=secret)
        if not result["ok"]:
            _emit("unlock_failed", severity="warning", status="rejected",
                  metadata={"error_code": result["error_code"]})
            return _r("rejected", state=errors.BLOCKED, code=result["error_code"], data=result)
        return _r("ok", state=errors.COMPLETED, data=result)

    def lock(self) -> dict:
        return _r("ok", state=errors.COMPLETED, data=self.sessions.lock())

    # -- profile change (sensitive → broker) ---------------------------------------

    def prepare_profile_change(self, to_profile: str, broker) -> dict:
        settings = self._settings()
        if not is_profile(to_profile):
            return _r("rejected", state=errors.BLOCKED, code=errors.INVALID_PROFILE)
        if self.sessions.base_profile() == LOCKED:
            return _r("rejected", state=errors.BLOCKED, code=errors.LOCKED)
        action_id = "prof_" + uuid.uuid4().hex
        expires = settings.memory_action_expires_seconds
        self._pending_profile = {"action_id": action_id, "to_profile": to_profile,
                                 "expires_monotonic": self._clock() + expires}
        broker.register(domain=PROFILE_DOMAIN, action_id=action_id, target=to_profile,
                        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=expires)
                                    ).isoformat(timespec="seconds"), cancel=self._clear_profile)
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION, action_id=action_id,
                  data={"to_profile": to_profile})

    def _clear_profile(self):
        self._pending_profile = None

    def confirm_profile_change(self, action_id, phrase, language, broker, wake=False):
        if wake:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH)
        pp = self._pending_profile
        if pp is None or pp["action_id"] != action_id:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED)
        if self._clock() >= pp["expires_monotonic"]:
            self._clear_profile(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED)
        result = self.sessions.set_base_profile(pp["to_profile"])
        self._clear_profile(); broker.clear(action_id)
        if not result["ok"]:
            return _r("rejected", state=errors.BLOCKED, code=result["error_code"])
        _emit("profile_changed", metadata={"to": pp["to_profile"]})
        return _r("executed", state=errors.COMPLETED, data=result)

    # -- elevation -----------------------------------------------------------------

    def prepare_elevation(self, capability: str) -> dict:
        result = self.sessions.request_elevation(capability)
        if not result["ok"]:
            return _r("rejected", state=errors.BLOCKED, code=result["error_code"])
        _emit("elevation_granted", metadata={"capability": capability})
        return _r("ok", state=errors.COMPLETED, data=result)

    # (elevation is granted directly after an unlocked, capability-scoped request; the
    # "prepare/confirm" split at the API layer keeps parity with the other flows.)
    def confirm_elevation(self, capability: str) -> dict:
        return self.prepare_elevation(capability)

    def revoke_elevation(self) -> dict:
        _emit("elevation_revoked")
        return _r("ok", state=errors.COMPLETED, data=self.sessions.revoke_elevation())

    # -- the central dispatch gate -------------------------------------------------

    def gate(self, command, settings) -> dict | None:
        """Return None to allow; a denied normalized result to block. Confirmations are
        not re-gated (they complete an already-authorized action)."""
        if not settings.enable_security_profiles:
            return None
        if getattr(command, "is_confirmation", False) or command.intent in ("confirm", "cancel"):
            return None
        self.sessions.check_idle()
        pol = permissions.policy_for(command.intent)
        if pol is None:
            return None                    # un-mapped intent → not a gated effect
        domain, effect, scope = pol
        decision = policy.authorize(self.sessions.owner, self.sessions.effective_profile(domain),
                                    domain, command.intent, effect, resource=scope, settings=settings)
        self.repo.audit("authorize", owner=self.sessions.owner,
                        detail=f"{command.intent} {decision.audit.get('decision')} {decision.error_code or ''}")
        if decision.allowed:
            return None
        _emit("permission_denied", severity="warning", status="rejected",
              metadata={"error_code": decision.error_code, "domain": domain})
        spoken = ("Unlock Fifi to do that." if decision.error_code == errors.LOCKED else
                  "Your current profile isn't permitted to do that.")
        return _r("rejected", state=errors.BLOCKED, code=decision.error_code, spoken=spoken,
                  data={"reason": decision.reason, "elevation_required": decision.elevation_required,
                        "domain": domain, "effect": effect})


_service: SecurityService | None = None


def get_security_service() -> SecurityService:
    global _service
    if _service is None:
        _service = SecurityService()
    return _service


def _authorization_gate(command, settings):
    # Never even instantiate the service (or its DB) while security is disabled.
    if not settings.enable_security_profiles:
        return None
    return get_security_service().gate(command, settings)


def _register() -> None:
    pending.register_domain(PROFILE_DOMAIN, phrases=PROFILE_PHRASES,
                            primary="confirmar cambio de perfil / confirm profile change")
    conversation.register_confirm_handler(
        PROFILE_DOMAIN,
        lambda aid, phrase, lang, b, wake=False: get_security_service().confirm_profile_change(
            aid, phrase, lang, b, wake=wake))
    conversation.register_authorization_gate(_authorization_gate)


_register()
