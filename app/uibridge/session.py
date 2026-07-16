"""Short-lived UI session tokens for the native desktop app (Phase 6C).

The desktop client mints a token, then presents it on the stream/overview endpoints.
Tokens are HMAC-signed with a per-PROCESS secret and kept ONLY in memory — they never
touch a database or a log, and a server restart invalidates every outstanding token
(there is nothing to persist). This is a lightweight loopback session, not a public
auth system: its job is to bind the streaming/overview endpoints to a client that
completed the local handshake, and to expire automatically.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid

from app.core import errors


def _b64(raw: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class UiSessionManager:
    """Mints and verifies short-lived, in-memory UI tokens.

    A token is ``"{sid}.{expiry}.{sig}"`` where ``sig = HMAC(secret, "{sid}.{expiry}")``.
    ``verify`` checks the signature in constant time, the expiry, and that the sid is
    still live (not revoked and not from a previous process)."""

    def __init__(self, *, secret: bytes | None = None, ttl_seconds: int = 1800,
                 clock=time.time, id_factory=lambda: uuid.uuid4().hex):
        self._secret = secret or os.urandom(32)
        self._ttl = int(ttl_seconds)
        self._clock = clock
        self._id = id_factory
        self._live: dict[str, int] = {}          # sid -> expiry (epoch seconds)

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).digest())

    def mint(self) -> dict:
        sid = self._id()
        expiry = int(self._clock()) + self._ttl
        payload = f"{sid}.{expiry}"
        token = f"{payload}.{self._sign(payload)}"
        self._live[sid] = expiry
        return {"token": token, "expires_at": expiry, "ttl_seconds": self._ttl}

    def _parse(self, token: str):
        parts = (token or "").split(".")
        if len(parts) != 3:
            return None
        sid, expiry, sig = parts
        if not sid or not expiry.isdigit():
            return None
        return sid, int(expiry), sig

    def verify(self, token: str) -> dict:
        """{ok: bool, error_code?, sid?}. Never raises."""
        parsed = self._parse(token or "")
        if parsed is None:
            return {"ok": False, "error_code": errors.UI_TOKEN_INVALID}
        sid, expiry, sig = parsed
        expected = self._sign(f"{sid}.{expiry}")
        if not hmac.compare_digest(sig, expected):
            return {"ok": False, "error_code": errors.UI_TOKEN_INVALID}
        if sid not in self._live:                 # revoked or from a previous process
            return {"ok": False, "error_code": errors.UI_TOKEN_INVALID}
        if int(self._clock()) >= expiry:
            self._live.pop(sid, None)
            return {"ok": False, "error_code": errors.UI_TOKEN_EXPIRED}
        return {"ok": True, "sid": sid}

    def revoke(self, token: str) -> bool:
        parsed = self._parse(token or "")
        if parsed is None:
            return False
        return self._live.pop(parsed[0], None) is not None

    def sweep(self) -> int:
        """Drop expired sids (bounded memory). Returns how many were removed."""
        now = int(self._clock())
        dead = [sid for sid, exp in self._live.items() if now >= exp]
        for sid in dead:
            self._live.pop(sid, None)
        return len(dead)

    def active_count(self) -> int:
        return len(self._live)


_manager: UiSessionManager | None = None


def get_session_manager() -> UiSessionManager:
    global _manager
    if _manager is None:
        from app.config import get_settings
        _manager = UiSessionManager(
            ttl_seconds=get_settings().desktop_session_ttl_minutes * 60)
    return _manager


def reset_session_manager() -> None:
    """Tests: drop the singleton so a fresh secret/clock can be injected."""
    global _manager
    _manager = None
