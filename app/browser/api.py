"""Browser automation API (Phase 4B).

    POST /browser/session/start      GET  /browser/session/status
    POST /browser/navigate           POST /browser/search
    GET  /browser/page               POST /browser/find
    POST /browser/action             (scroll | open_link | back | forward)
    POST /browser/form/prepare       POST /browser/form/confirm
    POST /browser/form/cancel        POST /browser/session/close

Reading/searching/scrolling/navigating are safe; filling a form requires an
exact confirmation phrase and never submits. Responses never expose filesystem
paths, cookies, tokens, or raw HTML.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.browser.service import get_browser_service

router = APIRouter(prefix="/browser", tags=["browser"])


class NavigateRequest(BaseModel):
    url: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class ActionRequest(BaseModel):
    type: str = Field(description="scroll | open_link | back | forward")
    direction: str = Field(default="down")
    text: str = Field(default="")


class PrepareFormRequest(BaseModel):
    fields: dict[str, str] = Field(default_factory=dict)


class ConfirmFormRequest(BaseModel):
    action_id: str = Field(min_length=1)
    phrase: str = Field(default="")


@router.post("/session/start")
def session_start() -> dict[str, Any]:
    return get_browser_service().start_session()


@router.get("/session/status")
def session_status() -> dict[str, Any]:
    return get_browser_service().status()


@router.post("/session/close")
def session_close() -> dict[str, Any]:
    return get_browser_service().close_session()


@router.post("/navigate")
def navigate(request: NavigateRequest) -> dict[str, Any]:
    return get_browser_service().navigate(request.url)


@router.post("/search")
def search(request: QueryRequest) -> dict[str, Any]:
    return get_browser_service().search(request.query)


@router.get("/page")
def page() -> dict[str, Any]:
    return get_browser_service().page()


@router.post("/find")
def find(request: QueryRequest) -> dict[str, Any]:
    return get_browser_service().find(request.query)


@router.post("/action")
def action(request: ActionRequest) -> dict[str, Any]:
    service = get_browser_service()
    kind = request.type.lower()
    if kind == "scroll":
        return service.scroll(request.direction)
    if kind == "open_link":
        return service.open_link(request.text)
    if kind == "back":
        return service.go_back()
    if kind == "forward":
        return service.go_forward()
    return {"status": "rejected", "reason": f"unknown action {request.type!r}"}


@router.post("/form/prepare")
def form_prepare(request: PrepareFormRequest) -> dict[str, Any]:
    return get_browser_service().prepare_form(request.fields)


@router.post("/form/confirm")
def form_confirm(request: ConfirmFormRequest) -> dict[str, Any]:
    return get_browser_service().confirm_form(request.action_id, request.phrase)


@router.post("/form/cancel")
def form_cancel() -> dict[str, Any]:
    return get_browser_service().cancel_form()
