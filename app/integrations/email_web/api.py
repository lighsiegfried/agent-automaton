"""Web-email API (Phase 4D).

    GET  /email/status
    POST /email/search             POST /email/thread/open
    POST /email/thread/summarize   POST /email/draft
    POST /email/draft/place        POST /email/send/prepare
    POST /email/send/confirm       POST /email/cancel   POST /email/close

Placing and sending are two confirmations routed through the unified broker (a
WhatsApp/text/browser confirmation can never send email). Never exposes cookies,
tokens, message bodies-in-full, or filesystem paths.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.integrations.email_web.service import get_email_service

router = APIRouter(prefix="/email", tags=["email"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    provider: str = Field(default="")


class ThreadRequest(BaseModel):
    thread_id: str = Field(default="")
    query: str = Field(default="")
    provider: str = Field(default="")


class DraftRequest(BaseModel):
    mode: str = Field(default="new", description="new | reply")
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str = Field(default="")
    body: str = Field(default="")
    provider: str = Field(default="")


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


def _svc():
    return get_email_service()


def _ctx():
    return "es", get_settings(), get_pending_broker()


@router.get("/status")
def status() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().status({}, lang, settings, broker)


@router.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().search({"query": request.query, "provider": request.provider}, lang, settings, broker)


@router.post("/thread/open")
def thread_open(request: ThreadRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().open_thread(request.model_dump(), lang, settings, broker)


@router.post("/thread/summarize")
def thread_summarize(request: ThreadRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().summarize_thread(request.model_dump(), lang, settings, broker)


@router.post("/draft")
def draft(request: DraftRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    args = {"to": request.to, "cc": request.cc, "subject": request.subject,
            "body": request.body, "provider": request.provider}
    method = _svc().draft_reply if request.mode == "reply" else _svc().draft_new
    return method(args, lang, settings, broker)


@router.post("/draft/place")
def draft_place(request: PhraseRequest) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=request.phrase),
        language="es")


@router.post("/send/prepare")
def send_prepare() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().prepare_send({}, lang, settings, broker)


@router.post("/send/confirm")
def send_confirm(request: PhraseRequest) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=request.phrase),
        language="es")


@router.post("/cancel")
def cancel() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().cancel({}, lang, settings, broker)


@router.post("/close")
def close() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().close({}, lang, settings, broker)
