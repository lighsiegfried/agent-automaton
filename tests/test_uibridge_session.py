"""Phase 6C — short-lived UI session tokens (in-memory, HMAC, no DB/log).

Mint/verify, expiry, tamper resistance, revoke, sweep, and the guarantee that a server
restart (a new manager with a new secret) invalidates every outstanding token."""

import pytest

from app.core import errors
from app.uibridge.session import UiSessionManager


def mgr(ttl=100, t0=1000, secret=b"k" * 32):
    clock = {"t": t0}
    counter = {"n": 0}

    def ids():
        counter["n"] += 1
        return f"sid{counter['n']}"

    m = UiSessionManager(secret=secret, ttl_seconds=ttl, clock=lambda: clock["t"], id_factory=ids)
    return m, clock


def test_mint_then_verify():
    m, _ = mgr()
    token = m.mint()["token"]
    r = m.verify(token)
    assert r["ok"] is True and r["sid"] == "sid1"


def test_expiry():
    m, clock = mgr(ttl=100)
    token = m.mint()["token"]
    clock["t"] += 101
    r = m.verify(token)
    assert r["ok"] is False and r["error_code"] == errors.UI_TOKEN_EXPIRED


def test_tampered_signature_rejected():
    m, _ = mgr()
    token = m.mint()["token"]
    assert m.verify(token + "x")["error_code"] == errors.UI_TOKEN_INVALID
    sid, expiry, _sig = token.split(".")
    forged = f"{sid}.{expiry}.deadbeef"
    assert m.verify(forged)["ok"] is False


def test_malformed_tokens_rejected():
    m, _ = mgr()
    for bad in ("", "a", "a.b", "a.b.c.d", "sid.notanumber.sig", None):
        assert m.verify(bad)["ok"] is False


def test_revoke():
    m, _ = mgr()
    token = m.mint()["token"]
    assert m.revoke(token) is True
    assert m.verify(token)["error_code"] == errors.UI_TOKEN_INVALID
    assert m.revoke(token) is False          # already gone


def test_restart_invalidates_all_tokens():
    m, _ = mgr(secret=b"a" * 32)
    token = m.mint()["token"]
    # a fresh process => fresh random secret AND empty live-set
    fresh = UiSessionManager(secret=b"b" * 32)
    assert fresh.verify(token)["ok"] is False


def test_sweep_drops_expired_only():
    m, clock = mgr(ttl=50)
    m.mint(); m.mint()
    clock["t"] += 60
    m.mint()                                  # a still-live one
    assert m.active_count() == 3
    removed = m.sweep()
    assert removed == 2 and m.active_count() == 1


def test_no_plaintext_secret_in_token():
    # the token is sid.expiry.sig — the HMAC secret never appears in it
    m, _ = mgr(secret=b"S3CRET-KEY-MATERIAL-32bytes-long!"[:32])
    token = m.mint()["token"]
    assert "S3CRET" not in token
