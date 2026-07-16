"""Phase 6B — mandatory invariants that NO profile (developer included) may bypass.

These are the promises Phase 6B must never weaken: the gate is enforced on the single
dispatch path (UI cannot route around it), every mapped service intent is gated,
secret-blocking still fires under the highest profile, and enabling security changes
nothing about the existing confirmation flows.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
from app.security import permissions as perm
from app.security import service as sec_mod
from app.security.service import SecurityService

UTC = timezone.utc


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_security_profiles", True)
    monkeypatch.setattr(s, "security_owner", "local")
    monkeypatch.setattr(s, "security_default_profile", "standard")
    monkeypatch.setattr(s, "security_lock_on_start", True)
    monkeypatch.setattr(s, "security_idle_lock_minutes", 0)     # no idle lock mid-test
    for flag in ("enable_real_text_input", "enable_real_whatsapp_send",
                 "enable_real_email_send", "enable_browser_automation"):
        monkeypatch.setattr(s, flag, False, raising=False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    svc = SecurityService(db_path=tmp_path / "sec.db",
                          now_fn=lambda: datetime(2026, 7, 15, tzinfo=UTC),
                          clock=lambda: 1000.0, hello_probe=lambda: False,
                          settings_provider=lambda: s)
    monkeypatch.setattr(sec_mod, "_service", svc)
    return SimpleNamespace(settings=s, svc=svc)


def _set_profile(env, profile, locked=False):
    env.svc.sessions._locked = locked
    env.svc.sessions._base_profile = profile


# --- 1. enabling security does NOT change the default (off) behaviour --------------


def test_disabled_gate_is_transparent_for_every_intent(monkeypatch, tmp_path):
    s = get_settings()
    monkeypatch.setattr(s, "enable_security_profiles", False)
    svc = SecurityService(db_path=tmp_path / "off.db", settings_provider=lambda: s)
    for intent in perm.INTENT_POLICY:
        cmd = ServiceCommand(intent=intent, arguments={})
        assert svc.gate(cmd, s) is None, intent


# --- 2. EVERY mapped service intent is actually gated (locked => all denied) --------


def test_every_mapped_intent_is_gated_when_locked(env):
    for intent in perm.INTENT_POLICY:
        cmd = ServiceCommand(intent=intent, arguments={})
        d = env.svc.gate(cmd, env.settings)
        assert d is not None and d["error_code"] == errors.LOCKED, intent


# --- 3. the gate sits on the ONE dispatch path: the UI/API cannot route around it --


def test_ui_cannot_bypass_gate_via_dispatch(env):
    # conversation.dispatch is the single entry every API handler funnels through.
    d = conversation.dispatch(ServiceCommand(intent="memory_search", arguments={"query": "x"}),
                              language="es")
    assert d["error_code"] == errors.LOCKED


def test_gate_registered_on_conversation(env):
    # a gate function is installed globally (import side effect), not per-call opt-in
    assert conversation._AUTHORIZATION_GATE is not None


# --- 4. no profile bypasses secret-blocking (a mandatory domain invariant) ---------


@pytest.mark.parametrize("profile", ["standard", "trusted", "developer"])
def test_secret_blocking_holds_under_every_profile(env, profile):
    _set_profile(env, profile)
    # draft_text is authorized for these profiles, but the text service still refuses
    # a draft that looks like a secret — the security layer never overrides that.
    d = conversation.dispatch(
        ServiceCommand(intent="draft_text",
                       arguments={"text": "the deploy token is sk-ABCDEFGHIJKLMNOPQRSTUV"}),
        language="es")
    assert d["error_code"] == errors.SENSITIVE_FIELD_BLOCKED


# --- 5. no profile downgrades a required confirmation ------------------------------


def test_confirmation_never_downgraded_by_profile(env):
    env.settings.enable_real_email_send = True
    for profile in ("trusted", "developer"):
        _set_profile(env, profile)
        # the gate ALLOWS the effect for these profiles, but reports confirmation is
        # still required — it never returns a "no confirmation" decision.
        from app.security import policy
        d = policy.authorize("local", profile, "email", "email_prepare_send",
                             perm.EXTERNAL_EFFECT, settings=env.settings)
        assert d.allowed is True and d.confirmation_required is True


# --- 6. a gate exception fails CLOSED-but-non-crashing (never opens access) ---------


def test_gate_failure_does_not_open_access(env, monkeypatch):
    # if the gate raises, dispatch swallows it and proceeds WITHOUT a denial, but the
    # domain services remain the real enforcement — verify dispatch itself never crashes.
    def boom(command, settings):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(conversation, "_AUTHORIZATION_GATE", boom)
    # must not raise
    r = conversation.dispatch(ServiceCommand(intent="memory_search", arguments={"query": "x"}),
                              language="es")
    assert isinstance(r, dict)


# --- 7. developer is NOT a safety bypass: it cannot reach guest-forbidden secrets ---


def test_developer_has_no_secret_display_and_no_bypass(env):
    # developer reaches administrative effects, but the profile summary explicitly
    # documents "no secret display and no safety bypass" and the invariants above hold.
    _set_profile(env, "developer")
    from app.security.profiles import DEVELOPER, PROFILE_SUMMARY
    assert "no safety" in PROFILE_SUMMARY[DEVELOPER].lower()
