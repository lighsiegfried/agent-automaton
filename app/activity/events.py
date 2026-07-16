"""Event normalization + the neutral-bus sink (Phase 5D).

Subsystems emit safe signals to ``app.core.eventbus``; this module turns each payload
into a redacted :class:`Event` and hands it to the Activity Center store. Everything
here is defensive: the sink checks the feature flag, swallows all errors, and never
blocks or raises into the command path. ``install()`` is idempotent, so re-imports
can't duplicate delivery.
"""

from __future__ import annotations

from app.activity.models import (
    SEV_ERROR, SEV_INFO, SEV_WARNING, Event, safe_metadata,
)
from app.core import eventbus
from app.core.logger import get_logger

log = get_logger(__name__)

_FAIL_STATUSES = {"error", "failed"}
_WARN_STATUSES = {"rejected", "blocked", "cancelled", "skipped", "paused"}

_TITLES = {
    "pending": "Pending action",
    "command": "Command",
    "schedules": "Schedule",
    "voice": "Voice command",
    "runtime": "Runtime",
}


def _severity(payload: dict, status: str) -> str:
    if payload.get("severity") in (SEV_INFO, SEV_WARNING, SEV_ERROR):
        return payload["severity"]
    if status in _FAIL_STATUSES:
        return SEV_ERROR
    if payload.get("error_code") or status in _WARN_STATUSES:
        return SEV_WARNING
    return SEV_INFO


def _humanize(text: str) -> str:
    return (text or "event").replace("_", " ").strip().capitalize()


def _describe(domain: str, event_type: str, status: str, meta: dict) -> str:
    """A generic, safe one-liner — never echoes content, only allowlisted hints."""
    bits: list[str] = []
    if meta.get("recipients"):
        bits.append("for " + ", ".join(meta["recipients"]))
    elif meta.get("recipient"):
        bits.append("for " + str(meta["recipient"]))
    elif meta.get("target"):
        bits.append("→ " + str(meta["target"]))
    if meta.get("count") is not None:
        bits.append(f"{meta['count']} result(s)")
    if meta.get("step_count") is not None:
        bits.append(f"{meta.get('current_step', 0)}/{meta['step_count']} steps")
    if meta.get("recurrence"):
        bits.append("recurring")
    tail = f" — {status}" if status else ""
    return (_humanize(event_type) + (": " + "; ".join(bits) if bits else "") + tail).strip()


def build_event(payload: dict, *, ts: str, event_id: str, owner: str = "local") -> Event:
    domain = payload.get("domain") or "command"
    event_type = payload.get("event_type") or "event"
    status = str(payload.get("status") or "")
    meta = safe_metadata(payload.get("metadata") or {})
    # Nested task/schedule public dicts — flatten only their SAFE scalar fields.
    data = payload.get("metadata") or {}
    if isinstance(data.get("task"), dict):
        t = data["task"]
        meta.setdefault("task_status", t.get("status"))
        meta.setdefault("current_step", t.get("current_step"))
        meta.setdefault("step_count", len(t.get("steps", []) or []))
        meta.setdefault("title", (t.get("title") or "")[:120])
    if isinstance(data.get("schedule"), dict):
        sc = data["schedule"]
        meta.setdefault("schedule_status", sc.get("status"))
        meta.setdefault("next_run_at", sc.get("next_run_at"))
        meta.setdefault("title", (sc.get("title") or "")[:120])
    meta = safe_metadata(meta)
    title = payload.get("title") or _TITLES.get(domain) or _humanize(event_type)
    summary = payload.get("summary") or _describe(domain, event_type, status, meta)
    return Event(
        event_id=event_id, ts=ts, domain=domain, event_type=event_type,
        severity=_severity(payload, status), status=status, title=title, summary=summary,
        related_id=payload.get("related_id") or payload.get("action_id"),
        duration_ms=payload.get("duration_ms"), error_code=payload.get("error_code"),
        metadata=meta, dedup_key=payload.get("dedup_key") or "", owner=owner)


def _sink(payload: dict) -> None:
    """Bus sink — best-effort, flag-gated, never raises or blocks the caller."""
    try:
        from app.config import get_settings

        if not get_settings().enable_activity_center:
            return
        from app.activity.service import get_activity_service

        get_activity_service().ingest(payload)
    except Exception:
        log.debug("activity ingest failed", exc_info=True)


def install() -> None:
    """Idempotently subscribe the activity sink to the neutral event bus."""
    eventbus.subscribe(_sink)
