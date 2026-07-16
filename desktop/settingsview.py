"""Read-only settings view for the desktop app (Phase 6C).

Builds a SAFE, display-ready list of settings rows from the local API's ``/identity``,
``/health`` and (optional) ``/security/status`` — the same non-secret metadata the tray
settings page shows. It NEVER surfaces tokens, keys, credentials, salts, hashes, or file
paths, and the desktop cannot CHANGE settings here (sensitive changes stay behind their
own confirmed flows). Every value is sanitized for display."""

from __future__ import annotations

from dataclasses import dataclass

from desktop.safety import sanitize_line

# Only these keys are ever shown. Anything not on the allowlist is dropped, so a future
# field that happens to hold a secret can't leak through this pane by accident.
_IDENTITY_KEYS = {
    "project_name": "Project",
    "agent_name": "Assistant",
    "version": "Version",
    "voice_enabled": "Voice",
    "llm_planner_enabled": "LLM planner",
    "real_windows_tools_enabled": "Real Windows tools",
    "wake_word": "Wake word",
}
_HEALTH_KEYS = {
    "tools_registered": "Tools registered",
    "real_windows_tools": "Real actions",
}


@dataclass(frozen=True)
class SettingRow:
    label: str
    value: str


def settings_view(identity: dict, health: dict, security: dict | None = None) -> list:
    """A list of :class:`SettingRow` (label, safe value) for the settings pane."""
    rows: list = []
    for key, label in _IDENTITY_KEYS.items():
        if key in (identity or {}):
            rows.append(SettingRow(label, sanitize_line(identity[key], max_len=60)))
    for key, label in _HEALTH_KEYS.items():
        if key in (health or {}):
            rows.append(SettingRow(label, sanitize_line(health[key], max_len=60)))
    if security is not None:
        enabled = security.get("enabled")
        rows.append(SettingRow("Security profiles", "on" if enabled else "off"))
        if enabled:
            state = "locked" if security.get("locked") else sanitize_line(
                security.get("profile", ""), max_len=40)
            rows.append(SettingRow("Session", state))
    return rows
