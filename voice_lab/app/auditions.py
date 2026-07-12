"""Saved-profile auditioning with a preview cache.

"Probar voz" synthesizes a short test sentence with ANY saved profile —
without activating it. Results are cached under
voice_lab/storage/previews/auditions/ keyed by (profile, text): repeating the
same audition is a cache hit and costs nothing. Cache entries are explicitly
saved previews (kept, gitignored); per-job temp files live elsewhere.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from app.config import PREVIEWS_DIR
from app.profiles.manager import ProfileManager, atomic_write_json

AUDITIONS_DIR = PREVIEWS_DIR / "auditions"

DEFAULT_AUDITION_TEXT = (
    "Hola, soy Fifi. Esta es una prueba de mi voz. ¿Cómo te gustaría que sonara?"
)

_KEY_LEN = 20


def profile_fingerprint(manager: ProfileManager, profile_name: str) -> str:
    """Cache-busting fingerprint: profile CONTENT (engine/model/speed/etc.).

    Editing a profile invalidates its cached previews — the hash covers the
    profile identifier, version (file content), and synthesis settings."""
    path = manager.profiles_dir / f"{profile_name}.json"
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "noprofile"


def audition_key(profile_name: str, text: str, fingerprint: str = "") -> str:
    digest = hashlib.sha1(
        f"{profile_name}|{fingerprint}|{text}".encode("utf-8")
    ).hexdigest()
    return f"{profile_name}-{digest[:8]}"[:64]


def valid_key(key: str) -> bool:
    return bool(key) and all(c.isalnum() or c in "_-" for c in key) and len(key) <= 80


def audition_paths(key: str) -> tuple[Path, Path]:
    if not valid_key(key):
        raise ValueError("invalid audition key")
    return AUDITIONS_DIR / f"{key}.wav", AUDITIONS_DIR / f"{key}.json"


def _payload(meta: dict[str, Any], key: str, wav_path: Path, cache_hit: bool) -> dict[str, Any]:
    """The full preview result (Phase 3D.0.4, req. 26) — every field a player
    needs, with a VERSIONED relative audio URL (regeneration busts caches)."""
    try:
        version = int(wav_path.stat().st_mtime)
    except OSError:
        version = 0
    return {
        **meta,
        "cache_hit": cache_hit,
        "key": key,
        "preview_id": key,
        "audio_url": f"/audio/previews/{key}?v={version}",
        "file": f"previews/auditions/{key}.wav",
        "requested_profile": meta.get("profile"),
        "actual_engine": meta.get("engine"),
        "synthesis_seconds": meta.get("seconds"),
        "audio_duration_seconds": meta.get("duration_seconds"),
    }


def cached_audition(
    profile_name: str, text: str, fingerprint: str = ""
) -> dict[str, Any] | None:
    key = audition_key(profile_name, text, fingerprint)
    wav_path, meta_path = audition_paths(key)
    if not (wav_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _payload(meta, key, wav_path, cache_hit=True)


def drop_profile_auditions(profile_name: str) -> None:
    """Remove cached auditions when a profile is deleted."""
    if not AUDITIONS_DIR.is_dir():
        return
    for path in AUDITIONS_DIR.glob(f"{profile_name}-*"):
        try:
            path.unlink()
        except OSError:
            pass


def run_audition(
    manager: ProfileManager,
    synthesize: Callable[..., dict[str, Any]],
    profile_name: str,
    text: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Synthesize (or reuse) an audition. NEVER activates the profile.

    `synthesize(text, profile, out_path)` is the worker's fallback-chain
    synthesis; the result reports the engine ACTUALLY used. force=True skips
    and replaces the cache ("Regenerar prueba").
    """
    from app import progress

    from datetime import datetime, timezone

    text = (text or "").strip() or DEFAULT_AUDITION_TEXT
    fingerprint = profile_fingerprint(manager, profile_name)
    cached = None if force else cached_audition(profile_name, text, fingerprint)
    if cached is not None:
        progress.report("generating", percentage=100, cache_hit=True,
                        message="Preview en caché — sin síntesis nueva.")
        return cached

    profile = manager.get_profile(profile_name)  # ProfileError bubbles cleanly
    key = audition_key(profile_name, text, fingerprint)
    wav_path, meta_path = audition_paths(key)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    progress.report(
        "generating", engine=profile.engine, cache_hit=False,
        message=f"Sintetizando prueba con {profile_name}…",
    )
    started = time.monotonic()
    result = synthesize(text, profile, wav_path)
    if result.get("status") != "ok":
        raise RuntimeError(
            f"audition failed ({result.get('engine')}): {result.get('message')}"
        )
    # A preview is only real once the WAV validates (nonzero, proper header,
    # positive duration) — the validated figures feed the player metadata.
    from app.audio.validation import validate_wav

    check = validate_wav(wav_path)
    if not check["valid"]:
        raise RuntimeError(f"audition audio failed validation: {check['reason']}")
    meta = {
        "profile": profile_name,
        "requested_engine": profile.engine,
        "engine": result.get("engine"),
        "fallback_used": bool(result.get("fallback_used")),
        "fallback_reason": result.get("fallback_reason", ""),
        "text": text,
        "seconds": round(time.monotonic() - started, 2),
        "duration_seconds": check["duration_seconds"],
        "sample_rate": check.get("sample_rate") or result.get("sample_rate"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_json(meta_path, meta)
    return _payload(meta, key, wav_path, cache_hit=False)
