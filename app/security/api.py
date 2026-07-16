"""Security API (Phase 6B).

    GET  /security/status            GET  /security/profile
    POST /security/unlock            POST /security/lock
    POST /security/profile/change/prepare   POST /security/profile/change/confirm
    GET  /security/permissions
    POST /security/elevation/prepare POST /security/elevation/confirm
    POST /security/elevation/revoke  GET  /security/audit

Profile changes require "confirmar cambio de perfil" through the shared broker.
Responses never expose password hashes, salts, secrets or raw tokens.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.security.service import get_security_service

router = APIRouter(prefix="/security", tags=["security"])


class UnlockRequest(BaseModel):
    method: str = Field(default="password")
    secret: str = Field(default="")


class ProfileRequest(BaseModel):
    profile: str = Field(default="")


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


class CapabilityRequest(BaseModel):
    capability: str = Field(default="")


class PasswordRequest(BaseModel):
    password: str = Field(default="")


def _svc():
    return get_security_service()


@router.get("/status")
def status() -> dict[str, Any]:
    return _svc().status()


@router.get("/profile")
def profile() -> dict[str, Any]:
    return _svc().profile()


@router.get("/permissions")
def permissions() -> dict[str, Any]:
    return _svc().permissions_view()


@router.post("/password")
def set_password(request: PasswordRequest) -> dict[str, Any]:
    return _svc().set_password(request.password)


@router.post("/unlock")
def unlock(request: UnlockRequest) -> dict[str, Any]:
    return _svc().unlock(request.method, request.secret)


@router.post("/lock")
def lock() -> dict[str, Any]:
    return _svc().lock()


@router.post("/profile/change/prepare")
def profile_change_prepare(request: ProfileRequest) -> dict[str, Any]:
    return _svc().prepare_profile_change(request.profile, get_pending_broker())


@router.post("/profile/change/confirm")
def profile_change_confirm(request: PhraseRequest) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True,
                       confirmation_phrase=request.phrase or "confirmar cambio de perfil"),
        language="es")


@router.post("/elevation/prepare")
def elevation_prepare(request: CapabilityRequest) -> dict[str, Any]:
    return _svc().prepare_elevation(request.capability)


@router.post("/elevation/confirm")
def elevation_confirm(request: CapabilityRequest) -> dict[str, Any]:
    return _svc().confirm_elevation(request.capability)


@router.post("/elevation/revoke")
def elevation_revoke() -> dict[str, Any]:
    return _svc().revoke_elevation()


@router.get("/audit")
def audit(limit: int = 100) -> dict[str, Any]:
    return {"events": _svc().audit(limit=limit)}
