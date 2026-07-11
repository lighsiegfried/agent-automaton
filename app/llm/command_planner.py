"""LLM command planner: natural language -> strictly validated command plan.

The LLM NEVER executes anything and its output is NEVER trusted directly.
plan_command() only returns a plan after every check passes:

- response is a single JSON object matching CommandPlan (no extra keys)
- tool_name exists in the registry and is NOT destructive
- intent maps to that exact tool (INTENT_TOOL_MAP)
- arguments are limited to the tool handler's declared parameters
- confidence is at or above MIN_CONFIDENCE

Anything else returns a failure reason and the caller falls back to the
rule-based router. Accepted plans still go through the normal safety layer.
"""

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.core.logger import get_logger
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from app.schemas.commands import Intent, SafetyLevel
from app.tools.registry import INTENT_TOOL_MAP, registry

log = get_logger(__name__)

MIN_CONFIDENCE = 0.6


class CommandPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = ""
    requires_confirmation: bool = False
    language: str = "en"


@dataclass(frozen=True)
class PlannerOutcome:
    plan: CommandPlan | None = None
    failure: str | None = None


def _tool_catalog() -> str:
    """One line per non-destructive tool; destructive tools are never offered."""
    lines = []
    for tool in registry.all():
        if tool.safety_level is SafetyLevel.DESTRUCTIVE:
            continue
        params = ", ".join(inspect.signature(tool.handler).parameters) or "(none)"
        lines.append(
            f"- {tool.name} | {tool.safety_level.value} | {params} | {tool.description}"
        )
    return "\n".join(lines)


def _call_llm(text: str) -> str | None:
    """Raw planner completion, or None if Ollama is unreachable/failed."""
    settings = get_settings()
    client = OllamaClient()
    try:
        result = client.generate(
            prompt=build_planner_prompt(text, _tool_catalog()),
            system=PLANNER_SYSTEM_PROMPT,
            model=settings.llm_planner_model,
            timeout=settings.llm_planner_timeout_seconds,
            json_format=True,
        )
    except httpx.HTTPError as exc:
        log.warning("llm planner request failed: %s", exc)
        return None
    if result.get("simulated") or not result.get("response"):
        return None
    return str(result["response"])


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):  # tolerate a fenced block despite instructions
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _validate_plan(plan: CommandPlan) -> str | None:
    """Return a rejection reason, or None if the plan is acceptable."""
    if plan.intent is Intent.UNKNOWN:
        return "planner returned intent 'unknown'"
    if plan.confidence < MIN_CONFIDENCE:
        return f"confidence {plan.confidence:.2f} below threshold {MIN_CONFIDENCE}"
    tool = registry.get(plan.tool_name)
    if tool is None:
        return f"unknown tool {plan.tool_name!r}"
    if tool.safety_level is SafetyLevel.DESTRUCTIVE:
        return f"destructive tool {plan.tool_name!r} refused"
    if INTENT_TOOL_MAP.get(plan.intent) != plan.tool_name:
        return f"intent {plan.intent.value!r} does not map to tool {plan.tool_name!r}"
    allowed = set(inspect.signature(tool.handler).parameters)
    unexpected = set(plan.arguments) - allowed
    if unexpected:
        return f"unexpected arguments for {plan.tool_name!r}: {sorted(unexpected)}"
    return None


def plan_command(text: str) -> PlannerOutcome:
    raw = _call_llm(text)
    if raw is None:
        return PlannerOutcome(failure="LLM unavailable")

    data = _extract_json(raw)
    if data is None:
        return PlannerOutcome(failure="LLM output was not a JSON object")

    try:
        plan = CommandPlan.model_validate(data)
    except ValidationError as exc:
        return PlannerOutcome(failure=f"plan schema invalid: {exc.error_count()} error(s)")

    plan.arguments = {key: str(value) for key, value in plan.arguments.items()}

    failure = _validate_plan(plan)
    if failure:
        return PlannerOutcome(failure=failure)
    return PlannerOutcome(plan=plan)
