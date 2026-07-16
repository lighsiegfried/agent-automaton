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
from app.browser.api import router as browser_router
from app.integrations.email_web.api import router as email_router
from app.integrations.whatsapp.api import router as whatsapp_router
from app.activity.api import router as activity_router
from app.knowledge.api import router as knowledge_router
from app.memory.api import router as memory_router
from app.schedules.api import router as schedules_router
from app.security.api import router as security_router
from app.tasks.api import router as tasks_router
from app.text.api import router as text_router
from app.uibridge.api import router as uibridge_router
from app.tools.registry import load_tools, registry
from app.voice.api import router as voice_router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_tools()
    memory.init_db()
    if get_settings().enable_memory:
        # Warm the personal-memory store: create the SQLite file, enable WAL, and
        # apply schema migrations (recovering a corrupt file) before first use.
        from app.memory.service import get_memory_service

        get_memory_service()
    if get_settings().enable_multi_step_tasks:
        # Warm the task store too (migrations/recovery). Paused tasks are available
        # for inspection after a restart, but nothing auto-resumes execution.
        from app.tasks.service import get_task_service

        get_task_service()
    scheduler_thread = None
    if get_settings().enable_schedules:
        # Warm the schedule store and start the ONE owned scheduler loop (bounded
        # polling; an atomic per-run lease prevents double-runs across restarts).
        import threading

        from app.schedules.service import get_schedule_service

        svc = get_schedule_service()
        scheduler_thread = threading.Thread(target=svc.scheduler.run_forever,
                                            name="fifi-scheduler", daemon=True)
        scheduler_thread.start()
    log.info("%s ready with %d tools", get_settings().app_name, len(registry.all()))
    try:
        yield
    finally:
        if get_settings().enable_schedules:
            from app.schedules.service import get_schedule_service

            get_schedule_service().scheduler.stop()


app = FastAPI(
    title="agent-automaton",
    description="Local-first Windows voice assistant (simulated by default; "
    "safe real actions behind ENABLE_REAL_WINDOWS_TOOLS)",
    version="0.3.2",
    lifespan=lifespan,
)

app.include_router(voice_router)
app.include_router(text_router)
app.include_router(browser_router)
app.include_router(whatsapp_router)
app.include_router(email_router)
app.include_router(memory_router)
app.include_router(tasks_router)
app.include_router(schedules_router)
app.include_router(activity_router)
app.include_router(knowledge_router)
app.include_router(security_router)
app.include_router(uibridge_router)


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


@app.get("/pending")
def pending() -> dict[str, object]:
    """The single cross-domain pending sensitive action (Phase 4B.1), or null.

    A SAFE summary only — domain, action_id, target, expiry, the required
    confirmation phrase, and status. Never the draft text or field values.
    """
    from app.core.pending import get_pending_broker

    return {"pending": get_pending_broker().summary()}


@app.post("/pending/cancel")
def pending_cancel() -> dict[str, object]:
    """Cancel the active pending action across any domain."""
    from app.core.pending import get_pending_broker

    had = get_pending_broker().cancel_active()
    return {"status": "cancelled" if had else "none", "had_pending": had}
