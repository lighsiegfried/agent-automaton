"""Pydantic models shared across the API, router, safety layer and tools."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intent(str, Enum):
    OPEN_FOLDER = "open_folder"
    OPEN_APP = "open_app"
    TYPE_TEXT = "type_text"
    SEARCH_WEB = "search_web"
    SPEAK = "speak"
    UNKNOWN = "unknown"


class SafetyLevel(str, Enum):
    SAFE = "safe"  # can be executed/simulated freely
    SENSITIVE = "sensitive"  # requires explicit confirmation
    DESTRUCTIVE = "destructive"  # blocked by default


class ExecutionStatus(str, Enum):
    SIMULATED = "simulated"  # tool ran in simulation mode
    EXECUTED = "executed"  # tool performed the real action (real mode only)
    REJECTED = "rejected"  # tool refused its input (allowlist, bad path, ...)
    NEEDS_CONFIRMATION = "needs_confirmation"  # re-send with confirm=true
    BLOCKED = "blocked"  # refused by the safety layer
    NOT_HANDLED = "not_handled"  # no tool for this intent


class CommandRequest(BaseModel):
    text: str = Field(min_length=1, description="Natural-language command")
    confirm: bool = Field(
        default=False,
        description="Set true to approve a sensitive action that asked for confirmation",
    )
    language: str | None = Field(
        default=None,
        description="Command language hint ('en'/'es'), e.g. from voice detection. "
        "Auto-guessed when omitted.",
    )


class PlannerSource(str, Enum):
    RULE_ROUTER = "rule_router"  # ENABLE_LLM_PLANNER=false (default)
    LLM_PLANNER = "llm_planner"  # validated LLM plan was used
    FALLBACK_ROUTER = "fallback_router"  # LLM failed/was rejected; rules used


class PlanInfo(BaseModel):
    """Short, safe summary of an accepted LLM plan (no chain-of-thought)."""

    confidence: float
    reasoning_summary: str
    language: str


class CommandResponse(BaseModel):
    input_text: str
    intent: Intent
    tool: str | None = None
    safety_level: SafetyLevel | None = None
    status: ExecutionStatus
    message: str
    result: dict[str, Any] | None = None
    planner: PlannerSource = PlannerSource.RULE_ROUTER
    plan: PlanInfo | None = None
    # Short natural-language reply for the user (bilingual; template-based
    # unless ENABLE_RESPONSE_GENERATOR=true). Suitable for TTS.
    assistant_message: str = ""
