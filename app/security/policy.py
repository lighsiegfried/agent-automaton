"""The one central authorization function (Phase 6B, item 9).

``authorize`` is the single deterministic gate every service dispatch path calls
before doing work. It is DEFAULT-DENY under an enabled profile, and it NEVER weakens a
domain confirmation — the ``confirmation_required`` it returns mirrors the existing
domain gate, and no profile (including developer) can set "no confirmation". When
security is disabled it is a transparent allow, so existing behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core import errors
from app.security import permissions as perm
from app.security.permissions import (
    EFFECT_RANK, ELEVATABLE_TO, EXTERNAL_EFFECT, OWNER_RESOURCE_DOMAINS, PROFILE_CEILING,
    READ, STATUS_ACTIONS, real_effect_enabled,
)
from app.security.profiles import GUEST, LOCKED, STANDARD


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    error_code: str | None = None
    confirmation_required: bool = False
    effective_scope: dict = field(default_factory=dict)
    elevation_required: bool = False
    audit: dict = field(default_factory=dict)


def _deny(code, reason, *, elevation=False, domain="", effect="", profile="") -> Decision:
    return Decision(False, reason=reason, error_code=code, elevation_required=elevation,
                    audit={"decision": "deny", "domain": domain, "effect": effect,
                           "profile": profile, "code": code})


def _allow(*, domain, effect, profile, scope="owner", confirmation=False, force_simulate=False) -> Decision:
    return Decision(True, confirmation_required=confirmation,
                    effective_scope={"scope": scope, "force_simulate": force_simulate},
                    audit={"decision": "allow", "domain": domain, "effect": effect, "profile": profile})


def authorize(identity, profile, domain, action, effect_level, *, resource=None,
              context=None, settings=None) -> Decision:
    """The central gate. ``effect_level`` is one of permissions.EFFECT_RANK. Returns a
    deterministic :class:`Decision`."""
    from app.config import get_settings
    settings = settings or get_settings()
    ctx = context or {}

    if not settings.enable_security_profiles:
        return _allow(domain=domain, effect=effect_level, profile=profile or "disabled")

    rank = EFFECT_RANK.get(effect_level, EFFECT_RANK[READ])
    needs_conf = rank >= EFFECT_RANK[EXTERNAL_EFFECT]   # informational; the domain still enforces it

    if profile == LOCKED:
        if domain == "runtime" and rank == EFFECT_RANK[READ] and action in STATUS_ACTIONS:
            return _allow(domain=domain, effect=effect_level, profile=profile, scope="public")
        return _deny(errors.LOCKED, "Fifi is locked — unlock to continue", domain=domain,
                     effect=effect_level, profile=profile)

    if profile == GUEST:
        if domain in OWNER_RESOURCE_DOMAINS:
            return _deny(errors.GUEST_ISOLATION, "guest cannot access owner resources",
                         domain=domain, effect=effect_level, profile=profile)
        if rank > EFFECT_RANK[READ]:
            return _deny(errors.PROFILE_INSUFFICIENT, "guest is read-only", domain=domain,
                         effect=effect_level, profile=profile)
        return _allow(domain=domain, effect=effect_level, profile=profile, scope="public")

    ceiling = PROFILE_CEILING.get(profile, -1)
    if rank <= ceiling:
        return _allow(domain=domain, effect=effect_level, profile=profile,
                      scope=(resource or "owner"), confirmation=needs_conf)

    # Above the profile's ceiling.
    if profile == STANDARD and effect_level == EXTERNAL_EFFECT and not real_effect_enabled(domain, settings):
        # Standard may run SIMULATED external effects (the real flag is off).
        return _allow(domain=domain, effect=effect_level, profile=profile, scope=(resource or "owner"),
                      confirmation=True, force_simulate=True)

    elevatable = (ELEVATABLE_TO.get(profile) is not None
                  and rank <= PROFILE_CEILING[ELEVATABLE_TO[profile]])
    return _deny(errors.ELEVATION_REQUIRED if elevatable else errors.PROFILE_INSUFFICIENT,
                 f"{profile} may not perform a {effect_level} action in {domain}",
                 elevation=elevatable, domain=domain, effect=effect_level, profile=profile)
