"""Windows desktop tools.

Simulated by default. When ENABLE_REAL_WINDOWS_TOOLS=true (and running on a
Windows host — never inside Docker), a small set of safe actions runs for real:

- open_folder: validated, existing, non-system folders only
- open_app:    strict allowlist (ALLOWED_APPS)

type_text stays simulated regardless of the flag (real input injection is a
Phase 4 concern). Validation always runs, even in simulated mode, so both modes
reject the same inputs.
"""

import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings, real_windows_tools_enabled
from app.schemas.commands import SafetyLevel
from app.tools.registry import registry

# Friendly folder names (English + Spanish) resolved under the home directory.
# The router also uses these to tell "abre descargas" (folder) apart from
# "abre notepad" (app).
FOLDER_ALIASES: dict[str, str] = {
    "home": "",
    "inicio": "",
    "downloads": "Downloads",
    "descargas": "Downloads",
    "documents": "Documents",
    "documentos": "Documents",
    "desktop": "Desktop",
    "escritorio": "Desktop",
    "pictures": "Pictures",
    "imagenes": "Pictures",
    "imágenes": "Pictures",
    "music": "Music",
    "musica": "Music",
    "música": "Music",
    "videos": "Videos",
    "vídeos": "Videos",
}

# Spanish / friendly names -> canonical app names used in ALLOWED_APPS.
APP_ALIASES: dict[str, str] = {
    "calculadora": "calculator",
    "explorador": "explorer",
    "bloc de notas": "notepad",
    "navegador": "edge",
}

# Canonical app name -> what Windows `start` should resolve (System32 exes or
# App Paths registry entries). Only reachable after the allowlist check, and
# values are code-defined — user text never becomes a command.
_APP_COMMANDS: dict[str, str] = {
    "notepad": "notepad",
    "calculator": "calc",
    "chrome": "chrome",
    "edge": "msedge",
    "explorer": "explorer",
}

# Environment variables whose directories (and everything inside) are refused.
_BLOCKED_ROOT_ENV_VARS = ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")


def validate_folder(raw: str) -> tuple[Path | None, str | None]:
    """Resolve a folder request to (path, None) or (None, rejection reason)."""
    raw = (raw or "").strip().strip('"')
    if not raw:
        return Path.home(), None

    alias = FOLDER_ALIASES.get(raw.lower())
    if alias is not None:
        path = Path.home() / alias
    else:
        path = Path(raw).expanduser()

    try:
        path = path.resolve()
    except OSError:
        return None, f"Invalid path: {raw!r}"

    if not path.exists():
        return None, f"Folder does not exist: {path}"
    if not path.is_dir():
        return None, f"Not a folder: {path}"

    for env_var in _BLOCKED_ROOT_ENV_VARS:
        root = os.environ.get(env_var)
        if root and (path == Path(root) or path.is_relative_to(Path(root))):
            return None, f"System location refused: {path}"

    attributes = getattr(path.stat(), "st_file_attributes", 0)
    if attributes & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM):
        return None, f"Hidden/system folder refused: {path}"

    return path, None


def resolve_app(raw: str) -> tuple[str | None, str | None]:
    """Resolve an app request to (canonical name, None) or (None, rejection reason)."""
    name = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if not name:
        return None, "No app name given."
    canonical = APP_ALIASES.get(name, name)
    allowed = get_settings().allowed_apps_list
    if canonical not in allowed:
        return None, (
            f"App {raw!r} is not in the allowlist. Allowed apps: {', '.join(allowed)}."
        )
    return canonical, None


@registry.register(
    name="open_folder",
    description="Open a folder in Windows Explorer (validated, non-system paths only)",
    safety_level=SafetyLevel.SAFE,
)
def open_folder(path: str = "") -> dict[str, Any]:
    folder, error = validate_folder(path)
    if error:
        return {"action": "open_folder", "error": error}
    if real_windows_tools_enabled():
        os.startfile(str(folder))  # noqa: S606 — validated, existing directory
        return {"simulated": False, "action": "open_folder", "opened": str(folder)}
    return {
        "simulated": True,
        "action": "open_folder",
        "would_do": f"Open Windows Explorer at {folder}",
    }


@registry.register(
    name="open_app",
    description="Launch an allowlisted application by name",
    safety_level=SafetyLevel.SENSITIVE,
)
def open_app(app: str = "") -> dict[str, Any]:
    canonical, error = resolve_app(app)
    if error:
        return {"action": "open_app", "error": error}
    if real_windows_tools_enabled():
        command = _APP_COMMANDS.get(canonical, canonical)
        # `start` resolves App Paths entries (chrome, msedge); no shell string,
        # and `command` comes from the code-defined map, never raw user text.
        subprocess.Popen(["cmd", "/c", "start", "", command])
        return {"simulated": False, "action": "open_app", "launched": canonical}
    return {
        "simulated": True,
        "action": "open_app",
        "would_do": f"Launch application '{canonical}'",
    }


@registry.register(
    name="type_text",
    description="Type text into the currently focused window",
    safety_level=SafetyLevel.SENSITIVE,
)
def type_text(text: str = "") -> dict[str, Any]:
    # Intentionally simulated even when ENABLE_REAL_WINDOWS_TOOLS=true:
    # real input injection needs a focus-check and preview flow first (Phase 4).
    return {
        "simulated": True,
        "action": "type_text",
        "would_do": f"Type into the focused window: {text!r}",
    }


@registry.register(
    name="shutdown_pc",
    description="Shut down the computer (demo of a destructive tool)",
    safety_level=SafetyLevel.DESTRUCTIVE,
)
def shutdown_pc() -> dict[str, Any]:
    # Registered only to demonstrate that the safety layer blocks destructive
    # tools; the router never reaches this handler.
    return {
        "simulated": True,
        "action": "shutdown_pc",
        "would_do": "Shut down the PC",
    }
