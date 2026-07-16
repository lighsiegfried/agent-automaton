"""Phase 6B — SecurityService, SessionManager, repository (temp DB).

Unlock/lock, failed-unlock lockout that survives a restart, the salted verifier that
never stores plaintext, capability-scoped temporary elevation with automatic expiry,
reboot clearing elevation, the broker-confirmed profile change (wake can never confirm
it), the central dispatch gate, and audit redaction.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.core import conversation, errors, pending
from app.core.nl import ServiceCommand
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
    monkeypatch.setattr(s, "security_idle_lock_minutes", 15)
    monkeypatch.setattr(s, "security_max_failed_unlocks", 3)
    monkeypatch.setattr(s, "security_lockout_minutes", 5)
    monkeypatch.setattr(s, "security_temp_elevation_minutes", 10)
    monkeypatch.setattr(s, "security_allow_local_password", True)
    monkeypatch.setattr(s, "security_allow_windows_hello", True)
    for flag in ("enable_real_text_input", "enable_real_whatsapp_send",
                 "enable_real_email_send", "enable_browser_automation"):
        monkeypatch.setattr(s, flag, False, raising=False)
    monkeypatch.setattr(pending, "_broker", pending.PendingBroker())
    clock = {"t": 1000.0}
    db = tmp_path / "sec.db"

    def make(path=None, hello=False):
        return SecurityService(
            db_path=path or db,
            now_fn=lambda: datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            clock=lambda: clock["t"], hello_probe=lambda: hello,
            settings_provider=lambda: s)

    svc = make()
    monkeypatch.setattr(sec_mod, "_service", svc)
    return SimpleNamespace(settings=s, svc=svc, make=make, clock=clock, db=db,
                           broker=pending.get_pending_broker())


def gate(env, intent, **a):
    return env.svc.gate(ServiceCommand(intent=intent, arguments=a), env.settings)


def unlocked(env, password="hunter2!"):
    env.svc.set_password(password)
    env.svc.unlock("password", password)
    return env.svc


# --- lock / unlock -----------------------------------------------------------------


def test_starts_locked(env):
    st = env.svc.status()["data"]
    assert st["locked"] is True and st["profile"] == "locked" and st["enabled"] is True


def test_unlock_happy_path(env):
    env.svc.set_password("hunter2!")
    r = env.svc.unlock("password", "hunter2!")
    assert r["status"] == "ok" and r["data"]["profile"] == "standard"


def test_unlock_wrong_password(env):
    env.svc.set_password("hunter2!")
    r = env.svc.unlock("password", "nope")
    assert r["status"] == "rejected" and r["error_code"] == errors.UNLOCK_FAILED


def test_unlock_without_verifier(env):
    r = env.svc.unlock("password", "whatever")
    assert r["error_code"] == errors.NO_VERIFIER


def test_windows_hello_unlock(env):
    svc = env.make(hello=True)
    r = svc.unlock("windows_hello", "")
    assert r["status"] == "ok" and r["data"]["profile"] == "standard"


# --- lockout survives a restart ----------------------------------------------------


def test_lockout_after_max_failures(env):
    env.svc.set_password("pw")
    for _ in range(3):
        r = env.svc.unlock("password", "bad")
    assert r["data"]["locked_out"] is True
    assert env.svc.unlock("password", "pw")["error_code"] == errors.ACCOUNT_LOCKED_OUT


def test_lockout_persists_across_restart(env):
    env.svc.set_password("pw")
    for _ in range(3):
        env.svc.unlock("password", "bad")
    restarted = env.make()                     # new process, same DB
    assert restarted.unlock("password", "pw")["error_code"] == errors.ACCOUNT_LOCKED_OUT


# --- the salted verifier never stores plaintext ------------------------------------


def test_password_never_stored_plaintext(env):
    env.svc.set_password("S3cr3t-Passphrase!")
    raw = env.db.read_bytes()
    assert b"S3cr3t-Passphrase!" not in raw
    assert b"pbkdf2" in raw                     # a salted verifier IS present


def test_verifier_public_hides_salt_and_hash(env):
    env.svc.set_password("pw")
    v = env.svc.repo.get_verifier("local")
    pub = v.public()
    assert "salt" not in pub and "hash" not in pub and pub["set"] is True


# --- temporary, capability-scoped elevation ----------------------------------------


def test_elevation_requires_unlocked(env):
    assert env.svc.prepare_elevation("email")["error_code"] == errors.UNLOCK_REQUIRED


def test_elevation_only_from_standard(env):
    unlocked(env)
    env.svc.sessions._base_profile = "guest"
    assert env.svc.prepare_elevation("email")["error_code"] == errors.PROFILE_INSUFFICIENT


def test_elevation_scoped_and_expires(env):
    unlocked(env)
    env.settings.enable_real_email_send = True
    env.settings.enable_real_whatsapp_send = True
    assert env.svc.prepare_elevation("email")["status"] == "ok"
    assert env.svc.sessions.effective_profile("email") == "trusted"
    assert env.svc.sessions.effective_profile("whatsapp") == "standard"   # scoped
    env.clock["t"] += 10 * 60 + 1
    assert env.svc.sessions.effective_profile("email") == "standard"      # expired


def test_elevation_revoke(env):
    unlocked(env)
    env.svc.prepare_elevation("email")
    env.svc.revoke_elevation()
    assert env.svc.sessions.effective_profile("email") == "standard"


def test_reboot_clears_elevation(env):
    unlocked(env)
    env.svc.prepare_elevation("email")
    assert env.svc.sessions._elevation is not None
    restarted = env.make()                     # fresh SessionManager on same DB
    assert restarted.sessions._elevation is None


def test_no_global_unrestricted_elevation(env):
    """Elevation is always tied to ONE capability; there is no '*'/global grant API."""
    unlocked(env)
    env.settings.enable_real_email_send = True
    env.svc.prepare_elevation("email")
    # a different real-effect domain is still not elevated
    env.settings.enable_real_whatsapp_send = True
    d = gate(env, "whatsapp_prepare_send")
    assert d is not None and d["error_code"] == errors.ELEVATION_REQUIRED


# --- profile change through the broker; wake can never confirm ---------------------


def test_profile_change_needs_confirmation(env):
    unlocked(env)
    r = env.svc.prepare_profile_change("trusted", env.broker)
    assert r["status"] == "needs_confirmation"


def test_wake_cannot_confirm_profile_change(env):
    unlocked(env)
    env.svc.prepare_profile_change("trusted", env.broker)
    wake = conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True,
                       confirmation_phrase="confirmar cambio de perfil"),
        language="es", wake=True)
    assert wake["error_code"] == errors.CONFIRMATION_MISMATCH
    assert env.svc.sessions.base_profile() == "standard"      # unchanged


def test_real_confirm_changes_profile(env):
    unlocked(env)
    env.svc.prepare_profile_change("trusted", env.broker)
    conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True,
                       confirmation_phrase="confirmar cambio de perfil"),
        language="es")
    assert env.svc.sessions.base_profile() == "trusted"


def test_profile_change_rejects_invalid(env):
    unlocked(env)
    assert env.svc.prepare_profile_change("root", env.broker)["error_code"] == errors.INVALID_PROFILE


def test_profile_change_blocked_when_locked(env):
    assert env.svc.prepare_profile_change("trusted", env.broker)["error_code"] == errors.LOCKED


# --- the central dispatch gate -----------------------------------------------------


def test_gate_allows_when_disabled(env, monkeypatch):
    monkeypatch.setattr(env.settings, "enable_security_profiles", False)
    assert gate(env, "memory_search", query="x") is None


def test_gate_ignores_confirmations(env):
    # a confirmation completes an already-authorized action; never re-gated
    assert env.svc.gate(ServiceCommand(intent="confirm", is_confirmation=True), env.settings) is None


def test_gate_ignores_unmapped_intents(env):
    assert gate(env, "some_unknown_intent") is None


def test_gate_denies_when_locked(env):
    d = gate(env, "memory_search", query="x")
    assert d is not None and d["error_code"] == errors.LOCKED


def test_gate_guest_isolation(env):
    unlocked(env)
    env.svc.sessions._base_profile = "guest"
    assert gate(env, "email_search")["error_code"] == errors.GUEST_ISOLATION


def test_idle_lock(env):
    unlocked(env)
    assert env.svc.sessions.base_profile() == "standard"
    env.clock["t"] += 15 * 60 + 1               # exceed idle window
    env.svc.sessions.check_idle()
    assert env.svc.sessions.base_profile() == "locked"


# --- audit redaction ---------------------------------------------------------------


def test_audit_redacts_secrets(env):
    import json
    unlocked(env, password="hunter2!")
    env.svc.prepare_profile_change("trusted", env.broker)
    events = env.svc.audit(limit=50)
    blob = json.dumps(events)
    assert len(events) > 0
    assert "hunter2!" not in blob
    assert '"salt"' not in blob and '"hash"' not in blob
