"""Security state models (Phase 6B). Pure dataclasses — no plaintext credentials."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verifier:
    """A SALTED password verifier — the plaintext is never stored anywhere."""
    algo: str
    salt: str          # hex
    hash: str          # hex
    iterations: int
    created_at: str = ""

    def public(self) -> dict:      # never expose salt/hash
        return {"algo": self.algo, "created_at": self.created_at, "set": True}


@dataclass
class Elevation:
    elevation_id: str
    owner: str
    from_profile: str
    to_profile: str
    capability: str            # the domain the elevation applies to
    granted_at: str
    expires_at_iso: str
    expires_monotonic: float
    revoked: bool = False

    def public(self) -> dict:
        return {"elevation_id": self.elevation_id, "from_profile": self.from_profile,
                "to_profile": self.to_profile, "capability": self.capability,
                "granted_at": self.granted_at, "expires_at": self.expires_at_iso,
                "revoked": self.revoked}


@dataclass
class SessionSnapshot:
    owner: str
    profile: str
    base_profile: str
    locked: bool
    elevation: dict | None
    failed_unlocks: int
    locked_out: bool
    lockout_until: str | None
    unlock_methods: list

    def public(self) -> dict:
        return {"owner": self.owner, "profile": self.profile, "base_profile": self.base_profile,
                "locked": self.locked, "elevation": self.elevation,
                "failed_unlocks": self.failed_unlocks, "locked_out": self.locked_out,
                "lockout_until": self.lockout_until, "unlock_methods": self.unlock_methods}
