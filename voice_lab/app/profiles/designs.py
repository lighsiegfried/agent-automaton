"""Voice design metadata — the committed half of a designed/cloned identity.

A frozen identity has two committed files and one local artifact:
- config/voices/profiles/<name>.json   — the profile Fifi selects (committed)
- config/voices/designs/<name>.json    — HOW the voice was made (committed):
  design fields, composed instruction, the reference transcript, and the
  authorization record for cloning
- voice_lab/storage/voice-identities/<name>/reference.wav — the reference
  audio itself (generated, gitignored, NEVER committed)

All paths stored here are RELATIVE to voice_lab/ — designs are portable and
never leak absolute paths through APIs.

Cloning safety: synthesizing with a cloned voice requires the design's
authorization record. Voices designed from text (VoiceDesign) are synthetic —
they carry authorization type "synthetic" automatically. Cloning an EXTERNAL
reference voice requires explicit authorization metadata; without it the
engine refuses.
"""

import json
import re
from pathlib import Path
from typing import Any

from app.config import VOICE_LAB_ROOT, VOICES_CONFIG_DIR
from app.profiles.manager import atomic_write_json

DESIGNS_DIR = VOICES_CONFIG_DIR / "designs"

_NAME_RE = re.compile(r"^[a-z0-9_\-]+$")


def valid_identity_name(name: str) -> bool:
    """Strict slug — also the path-traversal guard for every designer path."""
    return bool(name) and bool(_NAME_RE.match(name)) and len(name) <= 64


def design_path(name: str, designs_dir: Path | None = None) -> Path:
    if not valid_identity_name(name):
        raise ValueError(f"invalid identity name {name!r}")
    return Path(designs_dir or DESIGNS_DIR) / f"{name}.json"


def load_design(name: str, designs_dir: Path | None = None) -> dict[str, Any] | None:
    try:
        path = design_path(name, designs_dir)
    except ValueError:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_design(name: str, payload: dict[str, Any], designs_dir: Path | None = None) -> Path:
    path = design_path(name, designs_dir)
    atomic_write_json(path, payload)
    return path


def list_designs(designs_dir: Path | None = None) -> list[dict[str, Any]]:
    directory = Path(designs_dir or DESIGNS_DIR)
    designs: list[dict[str, Any]] = []
    if not directory.is_dir():
        return designs
    for path in sorted(directory.glob("*.json")):
        data = load_design(path.stem, directory)
        if data is not None:
            designs.append(data)
    return designs


def reference_audio_path(design: dict[str, Any]) -> Path | None:
    """Absolute path to the design's reference WAV (never exposed to callers).

    The design stores it RELATIVE to voice_lab/; anything absolute or escaping
    voice_lab/ is rejected.
    """
    relative = (design.get("reference") or {}).get("audio") or ""
    if not relative or ":" in relative or relative.startswith(("/", "\\")) or ".." in relative:
        return None
    return VOICE_LAB_ROOT / Path(relative)


def clone_authorized(design: dict[str, Any]) -> tuple[bool, str]:
    """(authorized, reason). Synthetic designs are self-authorized; external
    reference clones need an explicit authorization record."""
    reference = design.get("reference") or {}
    authorization = reference.get("authorization") or {}
    if authorization.get("type") == "synthetic":
        return True, ""
    if authorization.get("authorized") is True and authorization.get("statement"):
        return True, ""
    return False, (
        "voice cloning requires authorization metadata — the design's "
        "reference.authorization must be type 'synthetic' (VoiceDesign output) "
        "or carry authorized=true with an authorization statement"
    )
