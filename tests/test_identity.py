"""Fifi identity layer tests: persona is presentation-only, never safety."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, get_settings
from app.core.router import handle_command
from app.llm.response_generator import generate_assistant_message
from app.main import app
from app.schemas.commands import (
    CommandRequest,
    CommandResponse,
    ExecutionStatus,
    Intent,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# --- identity config -----------------------------------------------------------


def test_default_agent_name_is_fifi():
    assert get_settings().agent_name == "Fifi"


def test_wake_word_listening_is_never_started_by_the_api():
    """Phase 3D.1: a real wake service exists, but only the explicit host-side
    listener (scripts/fifi_wake.py) uses it — the API never instantiates it,
    never opens a microphone, and wake listening stays opt-in and off."""
    assert get_settings().wake_word == "fifi"
    assert get_settings().enable_wake_word is False  # default off
    import app.main as main_module
    import app.voice.api as voice_api_module

    for module in (main_module, voice_api_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "WakeWordService" not in source
        assert "wake_word import" not in source


# --- /identity endpoint ----------------------------------------------------------


def test_identity_endpoint(client):
    data = client.get("/identity").json()
    assert data["project_name"] == "agent-automaton"
    assert data["agent_name"] == "Fifi"
    assert data["wake_word"] == "fifi"
    assert data["wake_word_active"] is False
    assert data["voice_enabled"] is False
    assert data["llm_planner_enabled"] is False
    assert data["real_windows_tools_enabled"] is False
    assert data["version"]


def test_identity_exposes_no_paths_or_secrets(client):
    data = client.get("/identity").json()
    for key in data:
        assert "path" not in key.lower()
        assert "dir" not in key.lower()
        assert "url" not in key.lower()
    for value in data.values():
        assert ":\\" not in str(value)  # no Windows filesystem paths


# --- responses use the name naturally ---------------------------------------------


def test_unknown_command_introduces_fifi(client):
    english = client.post("/command", json={"text": "what is the weather today"}).json()
    assert "I'm Fifi" in english["assistant_message"]

    spanish = client.post("/command", json={"text": "hazme un sandwich"}).json()
    assert "Soy Fifi" in spanish["assistant_message"]


def test_name_is_not_overused(client):
    # ordinary outcomes don't mention the name
    for body in (
        {"text": "search for cats"},
        {"text": "open app notepad"},
        {"text": "open app notepad", "confirm": True},
    ):
        data = client.post("/command", json=body).json()
        assert "Fifi" not in data["assistant_message"]


def test_response_generator_uses_configured_name(settings):
    settings.agent_name = "Automatón"
    response = CommandResponse(
        input_text="hazme un sandwich",
        intent=Intent.UNKNOWN,
        status=ExecutionStatus.NOT_HANDLED,
        message="",
    )
    assert "Soy Automatón" in generate_assistant_message(response, language="es")


# --- persona never changes safety ---------------------------------------------------


def test_agent_name_does_not_alter_safety(settings):
    settings.agent_name = "sudo-admin-override"
    response = handle_command(CommandRequest(text="open app notepad"))
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_saying_fifi_does_not_bypass_confirmation(settings):
    response = handle_command(CommandRequest(text="Fifi open app notepad"))
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_destructive_still_blocked_with_persona(settings, launches):
    from app.core.safety import evaluate
    from app.schemas.commands import SafetyLevel

    decision = evaluate(SafetyLevel.DESTRUCTIVE, confirmed=True)
    assert not decision.allowed
    assert launches.popen_calls == []


# --- project is NOT renamed ----------------------------------------------------------


def test_project_folder_and_naming_unchanged():
    assert PROJECT_ROOT.name == "agent-automaton"
    assert get_settings().app_name == "agent-automaton"
    assert (PROJECT_ROOT / "compose.yml").exists()
    assert not (PROJECT_ROOT / "docker-compose.yml").exists()
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "# agent-automaton" in readme
