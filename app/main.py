"""agent-automaton — local assistant API server.

Run with:  uvicorn app.main:app --reload   (or scripts/run_dev.ps1)
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import get_settings, real_windows_tools_enabled
from app.core import memory
from app.core.logger import get_logger
from app.core.router import handle_command
from app.schemas.commands import CommandRequest, CommandResponse
from app.tools.registry import load_tools, registry
from app.voice.api import router as voice_router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_tools()
    memory.init_db()
    log.info("%s ready with %d tools", get_settings().app_name, len(registry.all()))
    yield


app = FastAPI(
    title="agent-automaton",
    description="Local-first Windows voice assistant (simulated by default; "
    "safe real actions behind ENABLE_REAL_WINDOWS_TOOLS)",
    version="0.3.2",
    lifespan=lifespan,
)

app.include_router(voice_router)


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    return {
        "status": "ok",
        "app": get_settings().app_name,
        "version": "0.3.2",
        "tools_registered": len(registry.all()),
        "real_windows_tools": real_windows_tools_enabled(),
        "llm_planner": get_settings().enable_llm_planner,
        "voice": get_settings().enable_voice,
    }


@app.get("/identity")
def identity() -> dict[str, str | bool]:
    """Who the assistant is and which optional layers are active.

    Presentation metadata only — the persona never changes safety behavior,
    and wake_word is future Phase 3C metadata (no listening exists yet).
    """
    settings = get_settings()
    return {
        "project_name": settings.app_name,
        "agent_name": settings.agent_name,
        "wake_word": settings.wake_word,
        "wake_word_active": False,
        "version": "0.3.2",
        "voice_enabled": settings.enable_voice,
        "llm_planner_enabled": settings.enable_llm_planner,
        "real_windows_tools_enabled": real_windows_tools_enabled(),
    }


@app.post("/command", response_model=CommandResponse)
def command(request: CommandRequest) -> CommandResponse:
    """Accept a text command; return intent, safety status and simulated result."""
    return handle_command(request)


@app.get("/tools")
def tools() -> list[dict[str, str]]:
    """List registered tools and their safety levels."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "safety_level": tool.safety_level.value,
        }
        for tool in registry.all()
    ]


@app.get("/history")
def history(limit: int = 20) -> list[dict[str, object]]:
    """Recent command log from local memory."""
    return memory.recent_commands(limit)
