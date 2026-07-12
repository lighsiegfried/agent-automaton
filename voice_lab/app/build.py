"""Build identity — how the UI, the worker, and the runtime prove they match.

The Phase 3D.0.3 incident: a stale worker process (old code, old routes) kept
serving the NEW designer.html straight from disk, so every new endpoint
404'd and the UI silently coerced that into "idle / no active voice / no
profiles / no se pudo iniciar". Three artifacts now carry the same build id:

- API_BUILD here (imported by the worker, exposed via /worker/info);
- UI_BUILD in static/designer.html (compared by the browser);
- scripts/local_runtime.py parses this file to know which build it EXPECTS.

Bump API_BUILD (and UI_BUILD in designer.html) together whenever the
frontend/backend contract changes. The source fingerprint is a separate,
automatic signal: it hashes the worker's own source tree at startup so a
running process can be told apart from the code currently on disk.
"""

import hashlib
import sys
from pathlib import Path

API_BUILD = "3d.0.4"

_APP_DIR = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    """Deterministic hash of the worker's source files (never their paths)."""
    digest = hashlib.sha256()
    files = sorted(_APP_DIR.rglob("*.py")) + [_APP_DIR / "static" / "designer.html"]
    for path in files:
        if "__pycache__" in path.parts or not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def repo_fingerprint() -> str:
    """Identifies WHICH checkout this worker serves without exposing the path."""
    repo_root = _APP_DIR.parent.parent  # agent-automaton/
    return hashlib.sha256(str(repo_root).lower().encode("utf-8")).hexdigest()[:12]


def python_fingerprint() -> str:
    """Identifies the interpreter without exposing its path."""
    return hashlib.sha256(sys.executable.lower().encode("utf-8")).hexdigest()[:12]


def venv_isolated() -> bool:
    """True only when this process runs on voice_lab/.venv (the isolation rule)."""
    voice_lab_root = _APP_DIR.parent
    try:
        Path(sys.prefix).resolve().relative_to(voice_lab_root)
    except ValueError:
        return False
    return True
