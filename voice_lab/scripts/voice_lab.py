"""Fifi Voice Lab CLI — try, compare, and activate voices.

    python voice_lab/scripts/voice_lab.py setup                # isolated env install
    python voice_lab/scripts/voice_lab.py status               # worker + active profile
    python voice_lab/scripts/voice_lab.py list                 # available profiles
    python voice_lab/scripts/voice_lab.py speakers             # base-voice catalog
    python voice_lab/scripts/voice_lab.py preview <profile>    # hear one profile
    python voice_lab/scripts/voice_lab.py design <profile>     # hear a designed identity
    python voice_lab/scripts/voice_lab.py compare              # hear every profile
    python voice_lab/scripts/voice_lab.py compare-speakers     # one per base voice
    python voice_lab/scripts/voice_lab.py compare-identities   # the designed identities
    python voice_lab/scripts/voice_lab.py select <profile>     # activate (atomic)
    python voice_lab/scripts/voice_lab.py play <profile>       # replay latest preview
    python voice_lab/scripts/voice_lab.py unload               # free engine VRAM

`select` works even when the worker is down — it validates the profile and
atomically rewrites config/voices/active.json, which the main project reads.
Preview/compare/unload need the worker:  python scripts/local_runtime.py voice-start

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import sys
from pathlib import Path

import httpx

VOICE_LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VOICE_LAB_ROOT))

from app.config import get_settings  # noqa: E402  (voice_lab's app package)
from app.profiles.manager import ProfileError, ProfileManager  # noqa: E402

REQUEST_TIMEOUT = 300.0  # first neural synthesis may include a model load


def worker_url() -> str:
    settings = get_settings()
    return f"http://{settings.host}:{settings.port}"


def _get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{worker_url()}{path}", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        response = httpx.post(
            f"{worker_url()}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        print(f"Worker request failed: {exc}")
        return None


def _need_worker() -> bool:
    if _get("/health") is None:
        print(
            f"Voice Lab worker is not running at {worker_url()}.\n"
            "Start it with:  python scripts/local_runtime.py voice-start"
        )
        return False
    return True


# --- commands -----------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    print("=== Fifi Voice Lab: status ===")
    health = _get("/health")
    print(f"Worker          : {'running' if health else 'not running'} ({worker_url()})")
    manager = ProfileManager()
    print(f"Active profile  : {manager.active_profile_name() or '(none selected)'}")
    if health:
        for engine, ok in (health.get("engines_available") or {}).items():
            print(f"Engine          : {engine} available={ok}")
        status = _get("/status") or {}
        print(f"Engines loaded  : {status.get('engines_loaded') or {}}")
    else:
        print("Engines         : (start the worker for availability)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    manager = ProfileManager()
    active = manager.active_profile_name()
    profiles = manager.list_profiles()
    if not profiles:
        print("No profiles found under config/voices/profiles/")
        return 1
    print("=== Fifi Voice Lab: profiles ===")
    for profile in profiles:
        marker = " *ACTIVE*" if profile.name == active else ""
        print(
            f"{profile.name:<20} engine={profile.engine:<12} "
            f"speaker={profile.speaker or '-':<12} speed={profile.speed:g} "
            f"lang={profile.language}{marker}"
        )
        if profile.style_instruction:
            print(f"{'':<20} style: {profile.style_instruction}")
    return 0


def _preview_one(name: str) -> bool:
    print(f"Preview         : {name} ...")
    result = _post("/preview", {"profile": name})
    if result is None:
        return False
    if result.get("status") != "ok":
        print(f"  FAILED ({result.get('engine')}): {result.get('message')}")
        return False
    note = " (fallback engine used)" if result.get("fallback_used") else ""
    print(
        f"  ok: engine={result.get('engine')} "
        f"duration={result.get('duration_seconds')}s "
        f"file={result.get('file')}{note}"
    )
    return True


def cmd_preview(args: argparse.Namespace) -> int:
    if not _need_worker():
        return 1
    return 0 if _preview_one(args.profile) else 1


def cmd_compare(args: argparse.Namespace) -> int:
    if not _need_worker():
        return 1
    profiles = ProfileManager().list_profiles()
    if not profiles:
        print("No profiles to compare.")
        return 1
    print("=== Fifi Voice Lab: compare (one after another) ===")
    failures = 0
    for profile in profiles:
        if not _preview_one(profile.name):
            failures += 1
    print(f"Compared {len(profiles)} profiles, {failures} failures.")
    print("Activate your favorite:  python voice_lab/scripts/voice_lab.py select <profile>")
    return 0 if failures == 0 else 1


def cmd_select(args: argparse.Namespace) -> int:
    manager = ProfileManager()
    try:
        active = manager.select(args.profile)
    except ProfileError as exc:
        print(str(exc))
        return 1
    print(f"Active profile  : {active['profile']} (engine {active['engine']})")
    print("config/voices/active.json updated atomically — Fifi reads it on next speak.")
    return 0


def cmd_speakers(args: argparse.Namespace) -> int:
    """Base-voice catalog: speakers/timbres, NOT speed/style profile variants."""
    from app.engines.catalog import build_catalog

    profiles = ProfileManager().list_profiles()
    # Availability comes from the WORKER (its venv has the neural engines);
    # locally-computed availability is only the offline fallback.
    health = _get("/health")
    catalog = build_catalog(
        profiles, (health or {}).get("engines_available")
    )
    if not health:
        print("(worker not running — availability reflects this shell, not the lab)")
    print("=== Fifi Voice Lab: speakers (base voices) ===")
    print("A profile listed under a voice is a speed/style VARIANT of it,")
    print("not a separate voice. 'designed' voices are defined by instruction.\n")
    current_engine = None
    for entry in catalog:
        if entry["engine"] != current_engine:
            current_engine = entry["engine"]
            print(f"[{current_engine}]  available={entry['available']}")
        used_by = f"  profiles: {', '.join(entry['profiles'])}" if entry["profiles"] else ""
        print(
            f"  {entry['id']:<18} lang={entry['language']:<6} "
            f"{entry['gender']:<9} {entry['kind']:<9} {entry['timbre']}"
        )
        if used_by:
            print(f"  {'':<18}{used_by.strip()}")
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    """Preview a DESIGNED identity, showing the instruction that defines it."""
    manager = ProfileManager()
    try:
        profile = manager.get_profile(args.profile)
    except ProfileError as exc:
        print(str(exc))
        return 1
    if profile.engine != "qwen3_tts" or not profile.style_instruction:
        print(
            f"{args.profile!r} is not a designed identity (engine={profile.engine}). "
            "Designed identities are qwen3_tts profiles with a style_instruction."
        )
        return 1
    print(f"Identity        : {profile.name}")
    print(f"Design          : {profile.style_instruction}")
    if not _need_worker():
        return 1
    return 0 if _preview_one(profile.name) else 1


def _compare_profiles(profiles, title: str) -> int:
    if not profiles:
        print("Nothing to compare.")
        return 1
    if not _need_worker():
        return 1
    print(f"=== Fifi Voice Lab: {title} ===")
    failures = 0
    for profile in profiles:
        detail = f" [{profile.engine}/{profile.speaker or 'designed'}]"
        print(f"Voice           : {profile.name}{detail}")
        if not _preview_one(profile.name):
            failures += 1
    print(f"Compared {len(profiles)} voices, {failures} failures.")
    return 0 if failures == 0 else 1


def cmd_compare_speakers(args: argparse.Namespace) -> int:
    """One preview per BASE speaker (style variants collapsed)."""
    from app.engines.catalog import base_speaker_profiles

    profiles = base_speaker_profiles(
        ProfileManager().list_profiles(), language=args.language or None
    )
    return _compare_profiles(
        sorted(profiles, key=lambda p: (p.engine, p.speaker)),
        "compare-speakers (one per base voice)",
    )


def cmd_compare_identities(args: argparse.Namespace) -> int:
    """Preview every designed voice identity with the same sentence."""
    from app.engines.catalog import designed_identity_profiles

    return _compare_profiles(
        designed_identity_profiles(ProfileManager().list_profiles()),
        "compare-identities (designed voices)",
    )


def cmd_setup(args: argparse.Namespace) -> int:
    """Run the isolated environment setup (voice_lab/scripts/setup.py)."""
    import setup as setup_module  # same directory; installs into voice_lab/.venv only

    return setup_module.main(["--with-qwen"] if args.with_qwen else [])


def cmd_play(args: argparse.Namespace) -> int:
    """Replay the newest saved preview for a profile; generate one if missing."""
    from app.audio.playback import play_wav

    previews_dir = VOICE_LAB_ROOT / "storage" / "previews"
    candidates = sorted(
        previews_dir.glob(f"preview_{args.profile}_*.wav"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        print(f"No saved preview for {args.profile!r} — generating one via the worker.")
        if not _need_worker():
            return 1
        return 0 if _preview_one(args.profile) else 1
    latest = candidates[-1]
    print(f"Playing         : storage/previews/{latest.name}")
    result = play_wav(latest)
    if not result.get("played"):
        print(f"Playback failed : {result.get('note')}")
        return 1
    return 0


def cmd_unload(args: argparse.Namespace) -> int:
    if not _need_worker():
        return 1
    result = _post("/unload", {})
    if result is None:
        return 1
    unloaded = result.get("unloaded") or []
    print(f"Unloaded engines: {', '.join(unloaded) if unloaded else '(none were loaded)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fifi Voice Lab CLI.")
    sub = parser.add_subparsers(dest="command")
    p_setup = sub.add_parser("setup", help="set up the isolated voice_lab/.venv")
    p_setup.add_argument(
        "--with-qwen", action="store_true", help="also install optional Qwen3-TTS deps"
    )
    p_setup.set_defaults(func=cmd_setup)
    sub.add_parser("status", help="worker + active profile").set_defaults(func=cmd_status)
    sub.add_parser("list", help="list voice profiles").set_defaults(func=cmd_list)
    sub.add_parser("speakers", help="base-voice catalog (speakers, not variants)").set_defaults(
        func=cmd_speakers
    )
    p_design = sub.add_parser("design", help="preview a designed identity + its instruction")
    p_design.add_argument("profile")
    p_design.set_defaults(func=cmd_design)
    p_cmp_speakers = sub.add_parser("compare-speakers", help="one preview per base voice")
    p_cmp_speakers.add_argument("--language", default="", help="filter, e.g. es")
    p_cmp_speakers.set_defaults(func=cmd_compare_speakers)
    sub.add_parser("compare-identities", help="preview all designed identities").set_defaults(
        func=cmd_compare_identities
    )
    p_preview = sub.add_parser("preview", help="synthesize + play one profile")
    p_preview.add_argument("profile")
    p_preview.set_defaults(func=cmd_preview)
    sub.add_parser("compare", help="preview every profile").set_defaults(func=cmd_compare)
    p_select = sub.add_parser("select", help="activate a profile (atomic)")
    p_select.add_argument("profile")
    p_select.set_defaults(func=cmd_select)
    p_play = sub.add_parser("play", help="replay the latest saved preview")
    p_play.add_argument("profile")
    p_play.set_defaults(func=cmd_play)
    sub.add_parser("unload", help="free engine VRAM in the worker").set_defaults(func=cmd_unload)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
