"""Multi-step task API (Phase 5B).

    POST /tasks/plan              GET  /tasks/{task_id}
    GET  /tasks                   POST /tasks/{task_id}/approve
    POST /tasks/{task_id}/run     POST /tasks/{task_id}/resume
    POST /tasks/{task_id}/cancel  GET  /tasks/{task_id}/audit

Planning validates deterministically; approval only lets read-only/local steps
begin; every effect still pauses in the shared broker and needs its own domain
phrase. Nothing here can bypass a confirmation or execute an effect on approval.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tasks.service import get_task_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


class PlanRequest(BaseModel):
    request: str = Field(default="", description="Natural-language task request")
    title: str = Field(default="")
    # Optional explicit, structured steps (deterministic path). Each is
    # {intent, arguments?, dependencies?}; still fully validated server-side.
    steps: list[dict] | None = Field(default=None)


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


def _svc():
    return get_task_service()


@router.post("/plan")
def plan(request: PlanRequest) -> dict[str, Any]:
    settings = get_settings()
    args: dict[str, Any] = {"request": request.request, "title": request.title}
    if request.steps is not None:
        args["steps"] = request.steps
    return _svc().plan(args, "es", settings)


@router.get("")
def list_tasks() -> dict[str, Any]:
    return {"tasks": _svc().list(get_settings())}


@router.get("/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = _svc().get(task_id, get_settings())
    return {"task": task} if task else {"task": None, "error_code": "TASK_NOT_FOUND"}


@router.post("/{task_id}/approve")
def approve(task_id: str, request: PhraseRequest) -> dict[str, Any]:
    return _svc().approve(task_id, request.phrase or "aprobar plan", get_settings(), "es")


@router.post("/{task_id}/run")
def run(task_id: str) -> dict[str, Any]:
    return _svc().run(task_id, get_settings(), "es")


@router.post("/{task_id}/resume")
def resume(task_id: str, request: PhraseRequest) -> dict[str, Any]:
    return _svc().resume(task_id, get_settings(), "es", phrase=request.phrase)


@router.post("/{task_id}/cancel")
def cancel(task_id: str) -> dict[str, Any]:
    return _svc().cancel(task_id, get_settings(), "es")


@router.get("/{task_id}/audit")
def audit(task_id: str, limit: int = 100) -> dict[str, Any]:
    return {"task_id": task_id, "events": _svc().audit(task_id, limit=limit)}
