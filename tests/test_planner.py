"""LLM planner tests. The Ollama call is always mocked — no model required."""

import json
from pathlib import Path

import pytest

from app.core.router import handle_command
from app.llm import command_planner
from app.llm.command_planner import plan_command
from app.schemas.commands import (
    CommandRequest,
    ExecutionStatus,
    Intent,
    PlannerSource,
)


def plan_json(**overrides) -> str:
    base = {
        "intent": "open_folder",
        "tool_name": "open_folder",
        "arguments": {"path": "downloads"},
        "confidence": 0.9,
        "reasoning_summary": "User wants their downloads folder.",
        "requires_confirmation": False,
        "language": "en",
    }
    base.update(overrides)
    return json.dumps(base)


@pytest.fixture
def mock_llm(settings, monkeypatch):
    """Enable the planner and control the raw LLM output. Returns a setter."""
    settings.enable_llm_planner = True
    holder = {"raw": None, "calls": 0}

    def fake_call(text: str):
        holder["calls"] += 1
        return holder["raw"]

    monkeypatch.setattr(command_planner, "_call_llm", fake_call)

    def set_raw(raw):
        holder["raw"] = raw

    set_raw.holder = holder
    return set_raw


needs_downloads = pytest.mark.skipif(
    not (Path.home() / "Downloads").is_dir(), reason="no Downloads folder"
)


# --- valid plans -------------------------------------------------------------


@needs_downloads
@pytest.mark.parametrize(
    "text", ["open my downloads folder", "abre mi carpeta de descargas"]
)
def test_valid_folder_plan(mock_llm, text):
    mock_llm(plan_json(language="es" if "abre" in text else "en"))
    response = handle_command(CommandRequest(text=text))
    assert response.planner is PlannerSource.LLM_PLANNER
    assert response.tool == "open_folder"
    assert response.status is ExecutionStatus.SIMULATED
    assert "Downloads" in response.result["would_do"]
    assert response.plan.reasoning_summary


@pytest.mark.parametrize(
    "text", ["search the web for local voice models", "busca en internet modelos de voz locales"]
)
def test_valid_search_plan(mock_llm, text):
    mock_llm(
        plan_json(
            intent="search_web",
            tool_name="search_web",
            arguments={"query": "local voice models"},
        )
    )
    response = handle_command(CommandRequest(text=text))
    assert response.planner is PlannerSource.LLM_PLANNER
    assert response.status is ExecutionStatus.SIMULATED
    assert "local+voice+models" in response.result["url"]


@pytest.mark.parametrize("text", ["open calculator", "abre la calculadora"])
def test_valid_app_plan_still_needs_confirmation(mock_llm, text):
    mock_llm(
        plan_json(
            intent="open_app",
            tool_name="open_app",
            arguments={"app": "calculator"},
            requires_confirmation=True,
        )
    )
    first = handle_command(CommandRequest(text=text))
    assert first.planner is PlannerSource.LLM_PLANNER
    assert first.status is ExecutionStatus.NEEDS_CONFIRMATION

    second = handle_command(CommandRequest(text=text, confirm=True))
    assert second.status is ExecutionStatus.SIMULATED
    assert "calculator" in second.result["would_do"]


# --- safety of the plan path -------------------------------------------------


def test_llm_cannot_loosen_sensitive_confirmation(mock_llm):
    # LLM claims no confirmation needed for a sensitive tool — safety layer wins.
    mock_llm(
        plan_json(
            intent="open_app",
            tool_name="open_app",
            arguments={"app": "notepad"},
            requires_confirmation=False,
        )
    )
    response = handle_command(CommandRequest(text="open notepad"))
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_llm_can_tighten_safe_action(mock_llm):
    # LLM asks for confirmation on a safe tool — honored (tighten-only).
    mock_llm(
        plan_json(
            intent="search_web",
            tool_name="search_web",
            arguments={"query": "cats"},
            requires_confirmation=True,
        )
    )
    response = handle_command(CommandRequest(text="search for cats"))
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_destructive_suggestion_never_executes(mock_llm, launches):
    mock_llm(
        plan_json(intent="open_app", tool_name="shutdown_pc", arguments={})
    )
    response = handle_command(CommandRequest(text="hazme un sandwich", confirm=True))
    assert response.planner is PlannerSource.FALLBACK_ROUTER
    assert response.tool != "shutdown_pc"
    assert response.status is ExecutionStatus.NOT_HANDLED  # rules can't route it either
    assert launches.popen_calls == []
    assert launches.startfile_calls == []


# --- rejected plans fall back to the rule router -------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "sorry, I cannot help with that",  # not JSON
        '{"intent": "open_app"',  # truncated JSON
        "[1, 2, 3]",  # JSON but not an object
        plan_json(tool_name="format_disk"),  # unknown tool
        plan_json(confidence=0.2),  # low confidence
        plan_json(arguments={"path": "downloads", "force": "true"}),  # extra arg
        plan_json(intent="search_web"),  # intent/tool mismatch
        plan_json(extra_field="x"),  # unknown top-level key
        None,  # Ollama unreachable
    ],
)
def test_bad_plan_falls_back_to_rules(mock_llm, raw):
    mock_llm(raw)
    response = handle_command(CommandRequest(text="open app notepad"))
    assert response.planner is PlannerSource.FALLBACK_ROUTER
    # rule router still resolves the command normally
    assert response.tool == "open_app"
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_unknown_intent_plan_falls_back_gracefully(mock_llm):
    mock_llm(plan_json(intent="unknown", tool_name="", arguments={}, confidence=0.3))
    response = handle_command(CommandRequest(text="qué hora es"))
    assert response.planner is PlannerSource.FALLBACK_ROUTER
    assert response.status is ExecutionStatus.NOT_HANDLED


# --- planner unit checks -------------------------------------------------------


def test_plan_command_reports_failure_reason(mock_llm):
    mock_llm(plan_json(tool_name="delete_file", intent="open_folder"))
    outcome = plan_command("borra mis archivos")
    assert outcome.plan is None
    assert "destructive" in outcome.failure


def test_plan_arguments_coerced_to_strings(mock_llm):
    mock_llm(plan_json(arguments={"path": 123}))
    outcome = plan_command("open folder 123")
    assert outcome.plan is not None
    assert outcome.plan.arguments == {"path": "123"}


# --- flag off: planner never called --------------------------------------------


def test_flag_off_uses_rule_router(settings, monkeypatch):
    settings.enable_llm_planner = False

    def boom(text):
        raise AssertionError("LLM must not be called when the planner is disabled")

    monkeypatch.setattr(command_planner, "_call_llm", boom)
    response = handle_command(CommandRequest(text="open app notepad"))
    assert response.planner is PlannerSource.RULE_ROUTER
    assert response.status is ExecutionStatus.NEEDS_CONFIRMATION


def test_planner_intent_detection_unchanged_by_default(settings):
    # default settings: identical behavior to Phase 1.5
    assert settings.enable_llm_planner is False
    response = handle_command(CommandRequest(text="abre descargas"))
    assert response.planner is PlannerSource.RULE_ROUTER
    assert response.intent is Intent.OPEN_FOLDER
