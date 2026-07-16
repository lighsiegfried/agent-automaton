"""Schedules API (Phase 5C).

    POST /schedules/prepare            POST /schedules/confirm
    GET  /schedules                    GET  /schedules/{id}
    POST /schedules/{id}/pause         POST /schedules/{id}/resume
    POST /schedules/{id}/update/prepare POST /schedules/{id}/cancel
    GET  /schedules/{id}/runs          GET  /schedules/audit

Creating/modifying is a two-step prepare→confirm routed through the shared broker;
scheduled runs still pause for each effect's own domain confirmation.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.schedules.service import get_schedule_service

router = APIRouter(prefix="/schedules", tags=["schedules"])


class PrepareRequest(BaseModel):
    text: str = Field(default="", description="Natural-language 'when' + title/action")
    title: str = Field(default="")
    kind: str = Field(default="", description="reminder | task (inferred if empty)")
    reminder_text: str = Field(default="")
    steps: list[dict] | None = Field(default=None)
    run_at: str | None = Field(default=None, description="explicit UTC ISO instant")
    recurrence: dict | None = Field(default=None)
    timezone: str = Field(default="")
    max_runs: int | None = Field(default=None)


class UpdateRequest(PrepareRequest):
    pass


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


def _svc():
    return get_schedule_service()


def _ctx():
    return "es", get_settings(), get_pending_broker()


@router.post("/prepare")
def prepare(request: PrepareRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    args = {k: v for k, v in request.model_dump().items() if v not in (None, "", [])}
    return _svc().prepare(args, lang, settings, broker)


@router.post("/confirm")
def confirm(request: PhraseRequest) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True,
                       confirmation_phrase=request.phrase or "confirmar programación"),
        language="es")


@router.get("")
def list_schedules() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().list({}, lang, settings, broker)


@router.get("/audit")
def audit(limit: int = 100) -> dict[str, Any]:
    return {"events": _svc().audit(limit=limit)}


@router.get("/{schedule_id}")
def get_schedule(schedule_id: str) -> dict[str, Any]:
    data = _svc().get(schedule_id, get_settings())
    return {"schedule": data} if data else {"schedule": None, "error_code": "SCHEDULE_NOT_FOUND"}


@router.post("/{schedule_id}/pause")
def pause(schedule_id: str) -> dict[str, Any]:
    return _svc().pause(schedule_id, get_settings(), "es")


@router.post("/{schedule_id}/resume")
def resume(schedule_id: str) -> dict[str, Any]:
    return _svc().resume(schedule_id, get_settings(), "es")


@router.post("/{schedule_id}/update/prepare")
def update_prepare(schedule_id: str, request: UpdateRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    args = {k: v for k, v in request.model_dump().items() if v not in (None, "", [])}
    return _svc().update_prepare(schedule_id, args, lang, settings, broker)


@router.post("/{schedule_id}/cancel")
def cancel(schedule_id: str) -> dict[str, Any]:
    return _svc().cancel(schedule_id, get_settings(), "es")


@router.post("/{schedule_id}/run")
def run_now(schedule_id: str) -> dict[str, Any]:
    return _svc().run_now(schedule_id, get_settings(), "es")


@router.get("/{schedule_id}/runs")
def runs(schedule_id: str) -> dict[str, Any]:
    return {"schedule_id": schedule_id, "runs": _svc().runs(schedule_id, get_settings())}
