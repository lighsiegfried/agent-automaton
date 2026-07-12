"""CWD-independent Voice Lab worker launcher (Phase 3D.0.3a).

    python voice_lab/run_worker.py                 # start the worker
    python voice_lab/run_worker.py --print-paths   # diagnostics: resolved paths

Why this exists: `python -m uvicorn app.main:app` resolves `app` from the
process CWD — launched from the repository root it would import the MAIN
project's app package instead of the Voice Lab's. This launcher pins
voice_lab/ first on sys.path from its own file location, so the worker, the
profile store, and the models directory are identical no matter which
directory it was started from.

--print-paths prints a JSON snapshot (paths relative to the repository, plus
the discovered profiles) and exits without binding the port or importing
FastAPI — used by tests and voice-doctor to verify CWD-independence.
"""

import json
import sys
from pathlib import Path

VOICE_LAB_ROOT = Path(__file__).resolve().parent

# voice_lab/ must win the `app` package lookup over any CWD-relative package.
if str(VOICE_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_LAB_ROOT))


def print_paths() -> int:
    from app.build import API_BUILD, repo_fingerprint, source_fingerprint
    from app.config import PROFILES_DIR, PROJECT_ROOT, ACTIVE_PROFILE_PATH
    from app.profiles.manager import ProfileManager

    manager = ProfileManager()
    store = manager.store_check()
    print(json.dumps({
        # Relative paths only — this output must stay safe to share.
        "profiles_dir": PROFILES_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "active_path": ACTIVE_PROFILE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "repo_fingerprint": repo_fingerprint(),
        "source_fingerprint": source_fingerprint(),
        "api_build": API_BUILD,
        "active_profile": manager.active_profile_name(),
        "profiles": sorted(p.name for p in manager.list_profiles()),
        "store_valid": store["valid"],
    }))
    return 0


def serve() -> int:
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    if "--print-paths" in sys.argv[1:]:
        raise SystemExit(print_paths())
    raise SystemExit(serve())
