"""Personal-memory API (Phase 5A).

    GET  /memory/status          POST /memory/propose
    POST /memory/confirm         POST /memory/search
    GET  /memory/list            POST /memory/update
    POST /memory/forget/prepare  POST /memory/forget/confirm
    POST /memory/cancel          GET  /memory/export

Plus read-only observability: GET /memory/audit and GET /memory/revisions.

Writes/updates/forgets are proposals committed only by an EXACT confirmation phrase
routed through the unified pending broker — a text/browser/email confirmation can
never commit a memory, and vice-versa. Export omits secrets (never stored) and
soft-deleted records by default.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.memory.service import get_memory_service

router = APIRouter(prefix="/memory", tags=["memory"])


class ProposeRequest(BaseModel):
    content: str = Field(min_length=1)
    type: str = Field(default="")
    title: str = Field(default="")
    subject: str = Field(default="")
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="api")
    confidence: float | None = Field(default=None)
    expires_at: str | None = Field(default=None)


class UpdateRequest(BaseModel):
    query: str = Field(default="")
    id: str = Field(default="")
    content: str = Field(default="")
    type: str = Field(default="")
    subject: str = Field(default="")


class SearchRequest(BaseModel):
    query: str = Field(default="")


class ForgetPrepareRequest(BaseModel):
    query: str = Field(default="")
    mode: str = Field(default="soft", description="soft | permanent")
    memory_ids: list[str] = Field(default_factory=list)


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


def _svc():
    return get_memory_service()


def _ctx():
    return "es", get_settings(), get_pending_broker()


def _confirm(phrase: str) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
        language="es")


@router.get("/status")
def status() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().status({}, lang, settings, broker)


@router.post("/propose")
def propose(request: ProposeRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    args = request.model_dump()
    if args.get("confidence") is None:
        args.pop("confidence", None)
    return _svc().propose(args, lang, settings, broker)


@router.post("/confirm")
def confirm(request: PhraseRequest) -> dict[str, Any]:
    return _confirm(request.phrase)


@router.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().search({"query": request.query}, lang, settings, broker)


@router.get("/list")
def list_memories(type: str = "") -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().list({"type": type} if type else {}, lang, settings, broker)


@router.post("/update")
def update(request: UpdateRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().propose_update(request.model_dump(), lang, settings, broker)


@router.post("/forget/prepare")
def forget_prepare(request: ForgetPrepareRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().prepare_forget(request.model_dump(), lang, settings, broker)


@router.post("/forget/confirm")
def forget_confirm(request: PhraseRequest) -> dict[str, Any]:
    return _confirm(request.phrase)


@router.post("/cancel")
def cancel() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().cancel({}, lang, settings, broker)


@router.get("/export")
def export(include_forgotten: bool = False) -> dict[str, Any]:
    return _svc().export(get_settings(), include_forgotten=include_forgotten)


@router.get("/audit")
def audit(limit: int = 50) -> dict[str, Any]:
    return {"events": _svc().audit_log(limit)}


@router.get("/revisions")
def revisions(memory_id: str) -> dict[str, Any]:
    return {"memory_id": memory_id, "revisions": _svc().revisions(memory_id)}
