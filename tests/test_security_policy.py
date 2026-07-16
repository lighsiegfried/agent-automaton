"""Phase 6B — the central authorize() decision matrix (pure, no DB).

Proves the deterministic policy: default-deny under an enabled profile, the
locked/guest/standard/trusted/developer bundles, owner-resource isolation for guest,
standard's simulate-vs-elevate rule for real external effects, and the invariant that
NO profile (including developer) ever downgrades a domain confirmation.
"""

import pytest

from app.config import get_settings
from app.core import errors
from app.security import permissions as perm
from app.security.permissions import (
    ADMINISTRATIVE, DESTRUCTIVE, EXTERNAL_EFFECT, EXTERNAL_PREPARE, LOCAL_WRITE, READ,
)
from app.security.policy import authorize


@pytest.fixture
def s(monkeypatch):
    """Settings with security ON and every real-effect flag OFF (simulate by default)."""
    st = get_settings()
    monkeypatch.setattr(st, "enable_security_profiles", True)
    for flag in ("enable_real_text_input", "enable_real_whatsapp_send",
                 "enable_real_email_send", "enable_browser_automation"):
        monkeypatch.setattr(st, flag, False, raising=False)
    return st


def auth(s, profile, domain, effect, action="x", resource="owner"):
    return authorize("local", profile, domain, action, effect, resource=resource, settings=s)


# --- disabled => transparent allow (backward compatible) ---------------------------


def test_disabled_is_transparent_allow(monkeypatch):
    st = get_settings()
    monkeypatch.setattr(st, "enable_security_profiles", False)
    for profile in ("locked", "guest", "standard", "trusted", "developer"):
        d = authorize("local", profile, "email", "email_prepare_send", EXTERNAL_EFFECT,
                      settings=st)
        assert d.allowed is True


# --- locked: nothing but the runtime status allowlist ------------------------------


def test_locked_denies_everything_but_status(s):
    d = auth(s, "locked", "memory", READ)
    assert d.allowed is False and d.error_code == errors.LOCKED
    d2 = auth(s, "locked", "email", EXTERNAL_EFFECT)
    assert d2.allowed is False and d2.error_code == errors.LOCKED
    # the explicit runtime status allowlist is readable while locked
    ok = authorize("local", "locked", "runtime", "status", READ, settings=s)
    assert ok.allowed is True and ok.effective_scope["scope"] == "public"


# --- guest: public read only, never owner resources --------------------------------


@pytest.mark.parametrize("domain", sorted(perm.OWNER_RESOURCE_DOMAINS))
def test_guest_cannot_touch_owner_domains(s, domain):
    d = auth(s, "guest", domain, READ)
    assert d.allowed is False and d.error_code == errors.GUEST_ISOLATION


def test_guest_public_read_allowed_write_denied(s):
    assert auth(s, "guest", "browser", READ, resource="public").allowed is True
    d = auth(s, "guest", "browser", EXTERNAL_EFFECT, resource="public")
    assert d.allowed is False and d.error_code == errors.PROFILE_INSUFFICIENT


# --- standard: owner read/write/prepare; external effect simulate-or-elevate -------


def test_standard_allows_up_to_prepare(s):
    for effect in (READ, LOCAL_WRITE, EXTERNAL_PREPARE):
        assert auth(s, "standard", "memory", effect).allowed is True


def test_standard_external_effect_simulated_when_real_flag_off(s):
    d = auth(s, "standard", "email", EXTERNAL_EFFECT)
    assert d.allowed is True
    assert d.effective_scope["force_simulate"] is True
    assert d.confirmation_required is True          # never downgraded


def test_standard_external_effect_requires_elevation_when_real(s, monkeypatch):
    monkeypatch.setattr(s, "enable_real_email_send", True)
    d = auth(s, "standard", "email", EXTERNAL_EFFECT)
    assert d.allowed is False
    assert d.error_code == errors.ELEVATION_REQUIRED and d.elevation_required is True


def test_standard_destructive_is_elevatable(s):
    d = auth(s, "standard", "knowledge", DESTRUCTIVE)
    assert d.allowed is False and d.error_code == errors.ELEVATION_REQUIRED


def test_standard_administrative_not_elevatable(s):
    d = auth(s, "standard", "settings", ADMINISTRATIVE)
    assert d.allowed is False and d.error_code == errors.PROFILE_INSUFFICIENT
    assert d.elevation_required is False


# --- trusted: up to destructive, still confirmed; not administrative ---------------


def test_trusted_allows_destructive_with_confirmation(s, monkeypatch):
    monkeypatch.setattr(s, "enable_real_email_send", True)
    d = auth(s, "trusted", "email", EXTERNAL_EFFECT)
    assert d.allowed is True and d.confirmation_required is True
    assert auth(s, "trusted", "knowledge", DESTRUCTIVE).allowed is True


def test_trusted_cannot_do_administrative(s):
    d = auth(s, "trusted", "settings", ADMINISTRATIVE)
    assert d.allowed is False and d.error_code == errors.PROFILE_INSUFFICIENT


# --- developer: full ceiling, but NO safety bypass ---------------------------------


def test_developer_reaches_administrative(s):
    assert auth(s, "developer", "settings", ADMINISTRATIVE).allowed is True


def test_no_profile_downgrades_confirmation(s, monkeypatch):
    """The invariant: an external-effect+ action ALWAYS reports confirmation_required,
    for every profile that is allowed to run it — developer included."""
    monkeypatch.setattr(s, "enable_real_email_send", True)
    for profile in ("trusted", "developer"):
        for effect in (EXTERNAL_EFFECT, DESTRUCTIVE):
            d = auth(s, profile, "email" if effect == EXTERNAL_EFFECT else "knowledge", effect)
            if d.allowed:
                assert d.confirmation_required is True, (profile, effect)


# --- default-deny: an unknown/over-ceiling combo is never silently allowed ---------


def test_default_deny_over_ceiling(s):
    # guest attempting a destructive action is denied, not defaulted-allow
    d = auth(s, "guest", "browser", DESTRUCTIVE, resource="public")
    assert d.allowed is False
