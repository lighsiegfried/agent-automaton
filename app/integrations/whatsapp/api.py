"""WhatsApp Web API (Phase 4C).

    GET  /whatsapp/status
    POST /whatsapp/contact/find      POST /whatsapp/chat/open
    POST /whatsapp/draft             POST /whatsapp/draft/place
    POST /whatsapp/send/prepare      POST /whatsapp/send/confirm
    POST /whatsapp/cancel            POST /whatsapp/close

Placing a draft and sending are two separate confirmations routed through the
unified broker (so a browser/text confirmation can never send WhatsApp). Never
exposes cookies, tokens, or QR data.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.integrations.whatsapp.service import get_whatsapp_service

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


class NameRequest(BaseModel):
    name: str = Field(min_length=1)


class DraftRequest(BaseModel):
    text: str = Field(default="", description="the message to draft (verbatim)")
    instruction: str = Field(default="")


class PhraseRequest(BaseModel):
    phrase: str = Field(default="", description="an exact confirmation phrase")


def _svc():
    return get_whatsapp_service()


def _ctx():
    return "es", get_settings(), get_pending_broker()


@router.get("/status")
def status() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().status({}, lang, settings, broker)


@router.post("/contact/find")
def contact_find(request: NameRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().find_contact({"name": request.name}, lang, settings, broker)


@router.post("/chat/open")
def chat_open(request: NameRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().open_chat({"name": request.name}, lang, settings, broker)


@router.post("/draft")
def draft(request: DraftRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().draft_message({"text": request.text, "instruction": request.instruction},
                                lang, settings, broker)


@router.post("/draft/place")
def draft_place(request: PhraseRequest) -> dict[str, Any]:
    # Routed through the broker so cross-domain confirmations are rejected.
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
