"""Docker/compose conventions and containment: support services only,
never desktop control, never reachable from assistant tools."""

import sys

from app.config import PROJECT_ROOT, real_windows_tools_enabled
from app.core.router import handle_command
from app.schemas.commands import CommandRequest, ExecutionStatus
from app.tools.registry import registry


def compose_text() -> str:
    return (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")


def test_compose_naming_convention():
    assert (PROJECT_ROOT / "compose.yml").exists()
    assert not (PROJECT_ROOT / "docker-compose.yml").exists()


def test_ollama_service_is_profile_gated():
    ollama_block = compose_text().split("ollama:", 1)[1]
    assert "profiles:" in ollama_block
    assert "llm" in ollama_block.split("ports:")[0]  # profile declared on the service
    assert "127.0.0.1:11434:11434" in ollama_block  # localhost only
    assert "ollama-models" in ollama_block  # named volume for models


def _active_compose_lines() -> list[str]:
    return [
        line.strip()
        for line in compose_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_ollama_gpu_reservation_is_active_not_commented():
    lines = _active_compose_lines()
    assert "- driver: nvidia" in lines
    assert "count: all" in lines
    assert "capabilities: [gpu]" in lines
    assert "NVIDIA_VISIBLE_DEVICES: all" in lines
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in lines


def test_api_service_stays_simulated_in_docker():
    api_block = compose_text().split("api:", 1)[1].split("ollama:", 1)[0]
    assert 'ENABLE_REAL_WINDOWS_TOOLS: "false"' in api_block


def test_docker_mode_never_reports_real_tools(settings, monkeypatch):
    # even with the flag forced on, a non-Windows host (container) stays simulated
    settings.enable_real_windows_tools = True
    monkeypatch.setattr(sys, "platform", "linux")
    assert real_windows_tools_enabled() is False


def test_tool_surface_is_locked():
    names = {tool.name for tool in registry.all()}
    assert names == {
        "open_folder",
        "open_app",
        "type_text",
        "shutdown_pc",
        "search_web",
        "list_folder",
        "delete_file",
        "speak",
    }
    forbidden = {"shell", "docker", "compose", "run_command", "execute", "powershell", "cmd"}
    assert not names & forbidden


def test_docker_and_shell_not_launchable_via_commands(settings, launches):
    for text in (
        "run docker",
        "open app docker",
        "abre docker",
        "ejecuta docker compose up",
        "run powershell",
        "open app cmd",
    ):
        response = handle_command(CommandRequest(text=text, confirm=True))
        assert response.status in (
            ExecutionStatus.REJECTED,
            ExecutionStatus.NOT_HANDLED,
        ), text
    assert launches.popen_calls == []
    assert launches.startfile_calls == []
