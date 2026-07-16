"""Text-action + window-target API (Phase 4A).

    POST /text/draft            create a pending draft (safe; nothing is typed)
    GET  /text/pending          the current pending draft preview (or null)
    POST /text/confirm          confirm with an exact phrase -> insert (or simulate)
    POST /text/cancel           discard the pending draft
    GET  /windows/active-target  a SAFE summary of the focused window/control

Responses never expose full window contents or filesystem paths (the executable
is a basename only). Insertion is simulated unless ENABLE_REAL_TEXT_INPUT=true.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.text.service import get_text_service
from app.text.targets import active_target

router = APIRouter(tags=["text"])


class DraftRequest(BaseModel):
    action: str = Field(
        default="type_text",
        description="type_text | append_text | replace_selected_text | draft_text | rewrite_text",
    )
    text: str | None = Field(default=None, description="explicit text to insert (verbatim)")
    instruction: str | None = Field(default=None, description="what to draft/rewrite")
    base_text: str = Field(default="", description="text to rewrite (rewrite_text)")
    language: str = Field(default="es")


class ConfirmRequest(BaseModel):
    action_id: str = Field(min_length=1)
    phrase: str = Field(default="", description="an exact confirmation phrase")


@router.post("/text/draft")
def text_draft(request: DraftRequest) -> dict[str, Any]:
    return get_text_service().draft(
        action=request.action, text=request.text, instruction=request.instruction,
        base_text=request.base_text, language=request.language,
    )


@router.get("/text/pending")
def text_pending() -> dict[str, Any]:
    return {"pending": get_text_service().pending_preview()}


@router.post("/text/confirm")
def text_confirm(request: ConfirmRequest) -> dict[str, Any]:
    return get_text_service().confirm(action_id=request.action_id, phrase=request.phrase)


@router.post("/text/cancel")
def text_cancel() -> dict[str, Any]:
    return get_text_service().cancel()


@router.get("/windows/active-target")
def windows_active_target() -> dict[str, Any]:
    # TargetSummary is already a SAFE summary (no window body, executable basename
    # only) — safe to serialise directly.
    return active_target().model_dump()
