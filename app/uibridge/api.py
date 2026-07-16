"""Desktop bridge API (Phase 6C).

    POST /ui/session          mint a short-lived UI session token (loopback only)
    POST /ui/session/revoke   revoke the presented token
    GET  /ui/overview         token-gated SAFE snapshot (pending + security summary)
    GET  /ui/stream           token-gated Server-Sent Events of safe activity

These endpoints exist ONLY to serve the native desktop client. They are gated by
``enable_desktop_bridge`` (off by default), bound to loopback, and require the
short-lived token minted by ``/ui/session``. They expose NO business logic and write
NO database — the desktop is a thin presentation layer over the existing API.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.core import errors
from app.uibridge.session import get_session_manager
from app.uibridge.stream import EventStream, format_sse, heartbeat

router = APIRouter(prefix="/ui", tags=["desktop"])

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _deny(code: str, status: int = 403) -> JSONResponse:
    return JSONResponse(status_code=status, content={"status": "rejected", "error_code": code})


def _is_local(request: Request) -> bool:
    client = request.client
    host = (client.host if client else "") or ""
    return host in _LOOPBACK


def _bridge_ready(request: Request) -> JSONResponse | None:
    """Common guard: feature flag ON and caller is loopback."""
    if not get_settings().enable_desktop_bridge:
        return _deny(errors.DESKTOP_DISABLED)
    if not _is_local(request):
        return _deny(errors.NON_LOCAL_BLOCKED)
    return None


def _token_from(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token", "") or request.headers.get("x-ui-token", "")


def _authorized(request: Request):
    """(guard_response | None, token). guard_response set => refuse with that."""
    guard = _bridge_ready(request)
    if guard is not None:
        return guard, ""
    token = _token_from(request)
    if not token:
        return _deny(errors.UI_TOKEN_REQUIRED, status=401), ""
    result = get_session_manager().verify(token)
    if not result["ok"]:
        status = 401 if result["error_code"] == errors.UI_TOKEN_EXPIRED else 403
        return _deny(result["error_code"], status=status), token
    return None, token


@router.post("/session")
def mint_session(request: Request) -> Any:
    guard = _bridge_ready(request)
    if guard is not None:
        return guard
    minted = get_session_manager().mint()
    return {"status": "ok", "data": minted}


@router.post("/session/revoke")
def revoke_session(request: Request) -> Any:
    guard = _bridge_ready(request)
    if guard is not None:
        return guard
    revoked = get_session_manager().revoke(_token_from(request))
    return {"status": "ok", "data": {"revoked": revoked}}


@router.get("/pending")
def pending(request: Request) -> Any:
    guard, _ = _authorized(request)
    if guard is not None:
        return guard
    return {"status": "ok", "data": {"pending": _enriched_pending()}}


@router.get("/overview")
def overview(request: Request) -> Any:
    guard, _ = _authorized(request)
    if guard is not None:
        return guard
    return {"status": "ok", "data": _safe_overview()}


def _enriched_pending() -> dict | None:
    """The safe pending summary plus the EXACT machine phrase a one-click confirm sends
    (``confirm_phrase``) — the same derivation the Activity Center uses. Never a draft
    body or a generic 'yes'."""
    from app.core import pending as pending_mod

    summary = pending_mod.get_pending_broker().summary()
    if not summary:
        return None
    return {**summary, "confirm_phrase": pending_mod.confirm_phrase_for(summary)}


def _safe_overview() -> dict:
    """A SAFE snapshot for the desktop's initial paint. Reuses existing safe summaries;
    never returns draft text, message bodies, secrets, or file paths."""
    data: dict = {"pending": _enriched_pending()}
    settings = get_settings()
    if settings.enable_security_profiles:
        try:
            from app.security.service import get_security_service
            data["security"] = get_security_service().status().get("data")
        except Exception:
            data["security"] = None
    else:
        data["security"] = {"enabled": False}
    return data


@router.get("/stream")
async def stream(request: Request):  # pragma: no cover - needs a running event loop
    guard, _ = _authorized(request)
    if guard is not None:
        return guard
    settings = get_settings()
    hb = max(1, settings.desktop_stream_heartbeat_seconds)
    es = EventStream(maxsize=settings.desktop_stream_queue_max).subscribe()

    async def gen():
        try:
            yield format_sse({"domain": "ui", "event_type": "connected", "status": "ok",
                              "severity": "info", "title": "Desktop connected",
                              "error_code": None, "related_id": None}, event_name="hello")
            while True:
                if await request.is_disconnected():
                    break
                event = await asyncio.to_thread(es.get, float(hb))
                if event is None:
                    yield heartbeat()
                else:
                    yield format_sse(event)
        finally:
            es.unsubscribe()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
