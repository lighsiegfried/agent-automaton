"""Profile storage + atomic activation.

Profiles live in config/voices/profiles/<name>.json (shared, committed).
The active selection lives in config/voices/active.json and is updated
ATOMICALLY (write temp file in the same directory, then os.replace) so the
main project can never read a half-written file.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ACTIVE_PROFILE_PATH, PROFILES_DIR
from app.profiles.schema import VoiceProfile


class ProfileError(Exception):
    """Raised for unknown/invalid profiles; carries a user-safe message."""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Temp file in the SAME directory + os.replace = atomic on Windows/POSIX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json.tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class ProfileManager:
    def __init__(
        self,
        profiles_dir: Path | None = None,
        active_path: Path | None = None,
    ) -> None:
        self.profiles_dir = Path(profiles_dir or PROFILES_DIR)
        self.active_path = Path(active_path or ACTIVE_PROFILE_PATH)

    # -- reading -------------------------------------------------------------------

    def list_profiles(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        if not self.profiles_dir.is_dir():
            return profiles
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(VoiceProfile(**data))
            except Exception:
                # A broken profile file must never take the worker down; it is
                # simply not offered. (Name-only report; never the path.)
                continue
        return profiles

    def get_profile(self, name: str) -> VoiceProfile:
        path = self.profiles_dir / f"{name}.json"
        if not path.is_file():
            known = ", ".join(p.name for p in self.list_profiles()) or "(none)"
            raise ProfileError(f"Unknown profile {name!r}. Available: {known}")
        try:
            return VoiceProfile(**json.loads(path.read_text(encoding="utf-8")))
        except ProfileError:
            raise
        except Exception as exc:
            raise ProfileError(f"Profile {name!r} is invalid: {exc}") from exc

    # -- activation ----------------------------------------------------------------

    def active(self) -> dict[str, Any] | None:
        """The current active.json content, or None when unreadable/missing."""
        try:
            return json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def active_profile_name(self) -> str | None:
        data = self.active()
        return data.get("profile") if isinstance(data, dict) else None

    # -- startup / readiness check ---------------------------------------------------

    def store_check(self) -> dict[str, Any]:
        """Is the SHARED profile store usable? (Phase 3D.0.3a startup check.)

        Degraded states are REPORTED, never repaired: a valid active profile is
        never silently replaced with none, and no second store is ever created
        under voice_lab/ or the CWD. A missing active.json is fine (the
        configured default applies); an unreadable one, or one pointing at a
        profile that does not exist, is degraded with a clear error.
        """
        profiles = self.list_profiles()
        result: dict[str, Any] = {
            "valid": True,
            "error": "",
            "profiles_dir_exists": self.profiles_dir.is_dir(),
            "profile_count": len(profiles),
            "active_profile": None,
            "active_exists": None,
        }
        if not result["profiles_dir_exists"]:
            result["valid"] = False
            result["error"] = (
                "el directorio compartido de perfiles (config/voices/profiles) no existe"
            )
            return result
        if not profiles:
            result["valid"] = False
            result["error"] = "no se encontró ningún perfil de voz en el almacén compartido"
            return result

        if not self.active_path.exists():
            return result  # no selection yet — the configured default applies
        active = self.active()
        if not isinstance(active, dict) or not active.get("profile"):
            result["valid"] = False
            result["error"] = "active.json existe pero no se puede leer o no tiene perfil"
            return result
        name = active["profile"]
        result["active_profile"] = name
        result["active_exists"] = any(p.name == name for p in profiles)
        if not result["active_exists"]:
            result["valid"] = False
            result["error"] = (
                f"active.json apunta al perfil {name!r}, que no existe en el almacén"
            )
        return result

    def select(self, name: str) -> dict[str, Any]:
        """Atomically activate a profile. Returns the new active.json content."""
        profile = self.get_profile(name)  # validates before anything is written
        payload = {
            "schema_version": 1,
            "profile": profile.name,
            "engine": profile.engine,
            "fallback_engine": profile.fallback_engine,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._atomic_write(payload)
        return payload

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.active_path, payload)
