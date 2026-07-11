"""File tools — SIMULATED.

Read-only operations may become real early; anything that writes or deletes
stays simulated until the safety model around it is solid.
"""

from typing import Any

from app.schemas.commands import SafetyLevel
from app.tools.registry import registry


@registry.register(
    name="list_folder",
    description="List the contents of a folder",
    safety_level=SafetyLevel.SAFE,
)
def list_folder(path: str = "") -> dict[str, Any]:
    target = path or "the user's home folder"
    return {
        "simulated": True,
        "action": "list_folder",
        "would_do": f"List files in {target}",
        # Real impl: Path(path).iterdir() restricted to allowlisted roots.
    }


@registry.register(
    name="delete_file",
    description="Delete a file (demo of a destructive tool)",
    safety_level=SafetyLevel.DESTRUCTIVE,
)
def delete_file(path: str = "") -> dict[str, Any]:
    # Registered only to demonstrate blocking; the safety layer never lets
    # destructive tools run.
    return {
        "simulated": True,
        "action": "delete_file",
        "would_do": f"Delete {path!r}",
    }
