"""Session security: identity, unlock, lockout, temporary elevation (Phase 6B, 6-7).

The authenticated OS user is the identity boundary. Unlock is via optional Windows
Hello (when available) or a SALTED PBKDF2 password verifier — the plaintext password is
never stored, logged, or placed in arguments. Failed unlocks trigger a lockout that
survives a restart; elevations are capability-scoped, visibly expiring, and cleared on
reboot (the manager revokes them at startup). No global unrestricted elevation exists.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core.logger import get_logger
from app.security.models import Elevation, SessionSnapshot
from app.security.profiles import DEVELOPER, LOCKED, STANDARD, TRUSTED, is_profile

log = get_logger(__name__)

_ITERATIONS = 120_000


def hash_password(password: str, salt: bytes | None = None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return salt.hex(), digest.hex()


class SessionManager:
    def __init__(self, repo, *, settings_provider=get_settings, now_fn=None, clock=time.monotonic,
                 hello_probe=None):
        self.repo = repo
        self._settings = settings_provider
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._clock = clock
        self._hello_probe = hello_probe or (lambda: False)
        settings = settings_provider()
        self.owner = settings.security_owner
        self._base_profile = LOCKED if settings.security_lock_on_start else settings.security_default_profile
        self._locked = settings.security_lock_on_start
        self._last_activity = self._clock()
        # Reboot clears elevations (they never survive a restart).
        self.repo.revoke_all_elevations(self.owner)
        self._elevation: Elevation | None = None

    # -- helpers -------------------------------------------------------------------

    def touch(self):
        self._last_activity = self._clock()

    def _lockout_active(self) -> tuple[bool, str | None]:
        cfg = self.repo.get_config(self.owner)
        lo = cfg.get("lockout_until")
        if lo and self._now().isoformat() < lo:
            return True, lo
        return False, lo

    def unlock_methods(self) -> list:
        settings = self._settings()
        methods = []
        if settings.security_allow_windows_hello and self._hello_probe():
            methods.append("windows_hello")
        if settings.security_allow_local_password and self.repo.get_verifier(self.owner):
            methods.append("password")
        return methods

    # -- password verifier ---------------------------------------------------------

    def set_password(self, password: str) -> None:
        from app.security.models import Verifier
        salt, digest = hash_password(password)
        self.repo.set_verifier(self.owner, Verifier(algo="pbkdf2_sha256", salt=salt, hash=digest,
                                                    iterations=_ITERATIONS, created_at=self.repo.now_iso()))
        self.repo.audit("password_set", owner=self.owner)

    def _verify_password(self, password: str) -> bool:
        v = self.repo.get_verifier(self.owner)
        if v is None:
            return False
        _, digest = hash_password(password, bytes.fromhex(v.salt))
        return hmac.compare_digest(digest, v.hash)

    # -- unlock / lock -------------------------------------------------------------

    def unlock(self, *, method: str = "password", secret: str = "") -> dict:
        from app.core import errors
        settings = self._settings()
        locked_out, until = self._lockout_active()
        if locked_out:
            self.repo.audit("unlock_blocked", owner=self.owner, detail="locked_out")
            return {"ok": False, "error_code": errors.ACCOUNT_LOCKED_OUT, "lockout_until": until}

        ok = False
        if method == "windows_hello" and settings.security_allow_windows_hello and self._hello_probe():
            ok = True
        elif method == "password" and settings.security_allow_local_password:
            if self.repo.get_verifier(self.owner) is None:
                return {"ok": False, "error_code": errors.NO_VERIFIER}
            ok = self._verify_password(secret)

        cfg = self.repo.get_config(self.owner)
        if not ok:
            failed = cfg["failed_unlocks"] + 1
            lockout = None
            if failed >= settings.security_max_failed_unlocks:
                lockout = (self._now() + timedelta(minutes=settings.security_lockout_minutes)
                           ).isoformat(timespec="seconds")
            self.repo.set_lockout(self.owner, failed=failed, lockout_until=lockout)
            self.repo.audit("unlock_failed", owner=self.owner, detail=f"failed={failed}")
            return {"ok": False, "error_code": errors.UNLOCK_FAILED, "failed": failed,
                    "locked_out": lockout is not None}

        self.repo.set_lockout(self.owner, failed=0, lockout_until=None)
        self._locked = False
        self._base_profile = settings.security_default_profile
        self.touch()
        self.repo.audit("unlocked", owner=self.owner, detail=f"method={method} profile={self._base_profile}")
        return {"ok": True, "profile": self._base_profile}

    def lock(self) -> dict:
        self._locked = True
        self._base_profile = LOCKED
        self._revoke_elevation("locked")
        self.repo.audit("locked", owner=self.owner)
        return {"ok": True, "locked": True}

    def check_idle(self) -> bool:
        settings = self._settings()
        if not self._locked and settings.security_idle_lock_minutes > 0:
            if (self._clock() - self._last_activity) > settings.security_idle_lock_minutes * 60:
                self.lock()
                self.repo.audit("idle_locked", owner=self.owner)
                return True
        return False

    # -- profile / elevation -------------------------------------------------------

    def base_profile(self) -> str:
        return LOCKED if self._locked else self._base_profile

    def _elevation_active(self) -> bool:
        return (self._elevation is not None and not self._elevation.revoked
                and self._clock() < self._elevation.expires_monotonic)

    def effective_profile(self, domain: str = "") -> str:
        base = self.base_profile()
        if self._elevation_active() and (not domain or self._elevation.capability in (domain, "*")):
            return self._elevation.to_profile
        return base

    def set_base_profile(self, profile: str) -> dict:
        from app.core import errors
        if not is_profile(profile):
            return {"ok": False, "error_code": errors.INVALID_PROFILE}
        if self._locked:
            return {"ok": False, "error_code": errors.LOCKED}
        self._base_profile = profile
        self.repo.audit("profile_changed", owner=self.owner, detail=profile)
        return {"ok": True, "profile": profile}

    def request_elevation(self, capability: str, *, to_profile: str = TRUSTED) -> dict:
        """Temporary trusted access to ONE capability (never a global unrestricted
        elevation). Requires an unlocked session; expires automatically."""
        from app.core import errors
        settings = self._settings()
        if self._locked:
            return {"ok": False, "error_code": errors.UNLOCK_REQUIRED}
        if self.base_profile() != STANDARD:
            return {"ok": False, "error_code": errors.PROFILE_INSUFFICIENT}
        minutes = settings.security_temp_elevation_minutes
        elev = Elevation(
            elevation_id=self.repo.new_id(), owner=self.owner, from_profile=self.base_profile(),
            to_profile=to_profile, capability=capability, granted_at=self.repo.now_iso(),
            expires_at_iso=(self._now() + timedelta(minutes=minutes)).isoformat(timespec="seconds"),
            expires_monotonic=self._clock() + minutes * 60)
        self.repo.create_elevation(elev)
        self._elevation = elev
        self.repo.audit("elevation_granted", owner=self.owner,
                        detail=f"cap={capability} to={to_profile} mins={minutes}")
        return {"ok": True, "elevation": elev.public()}

    def revoke_elevation(self) -> dict:
        self._revoke_elevation("manual")
        return {"ok": True}

    def _revoke_elevation(self, reason):
        if self._elevation is not None:
            self._elevation.revoked = True
            self.repo.revoke_all_elevations(self.owner)
            self.repo.audit("elevation_revoked", owner=self.owner, detail=reason)
            self._elevation = None

    # -- snapshot ------------------------------------------------------------------

    def snapshot(self) -> SessionSnapshot:
        locked_out, until = self._lockout_active()
        cfg = self.repo.get_config(self.owner)
        elev = self._elevation.public() if self._elevation_active() else None
        return SessionSnapshot(
            owner=self.owner, profile=self.effective_profile(), base_profile=self.base_profile(),
            locked=self._locked, elevation=elev, failed_unlocks=cfg["failed_unlocks"],
            locked_out=locked_out, lockout_until=until, unlock_methods=self.unlock_methods())
