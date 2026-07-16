"""Phase 6C — the read-only settings view: allowlisted, sanitized, no secrets."""

from desktop.settingsview import SettingRow, settings_view


IDENTITY = {"project_name": "agent-automaton", "agent_name": "Fifi", "version": "0.3.2",
            "voice_enabled": False, "real_windows_tools_enabled": False,
            # a hypothetical secret field must NEVER be surfaced (not on the allowlist):
            "api_token": "sk-SECRET", "db_path": "C:/secret/path"}
HEALTH = {"tools_registered": 12, "real_windows_tools": False, "internal_secret": "x"}


def test_settings_view_allowlists_fields():
    rows = settings_view(IDENTITY, HEALTH)
    labels = {r.label for r in rows}
    assert "Assistant" in labels and "Version" in labels
    # secrets / paths are dropped
    joined = " ".join(f"{r.label}={r.value}" for r in rows)
    assert "sk-SECRET" not in joined and "C:/secret/path" not in joined
    assert "api_token" not in joined and "internal_secret" not in joined


def test_settings_view_sanitizes_values():
    rows = settings_view({"agent_name": "<b>Fifi</b>"}, {})
    val = [r.value for r in rows if r.label == "Assistant"][0]
    assert "<b>" not in val


def test_settings_view_includes_security_when_present():
    rows = settings_view({}, {}, security={"enabled": True, "locked": True, "profile": "locked"})
    labels = {r.label: r.value for r in rows}
    assert labels["Security profiles"] == "on" and labels["Session"] == "locked"


def test_settings_view_security_off():
    rows = settings_view({}, {}, security={"enabled": False})
    labels = {r.label: r.value for r in rows}
    assert labels["Security profiles"] == "off" and "Session" not in labels


def test_rows_are_typed():
    rows = settings_view(IDENTITY, HEALTH)
    assert all(isinstance(r, SettingRow) for r in rows)
