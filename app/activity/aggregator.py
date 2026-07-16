"""Read-model builders for the Activity Center sections (Phase 5D).

Every section reads from the EXISTING services (task/schedule/memory singletons, the
pending broker, health probes) plus the local event store, and is wrapped so a
disabled, absent, or failing subsystem degrades that ONE section to a safe
``{"status": "degraded"}`` rather than crashing the whole UI. Nothing here mutates a
subsystem; nothing here exposes environment variables or filesystem paths.
"""

from __future__ import annotations

from datetime import timedelta, timezone

from app.core.logger import get_logger

log = get_logger(__name__)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # a subsystem failure degrades only its section
        log.debug("activity section degraded", exc_info=True)
        return {"status": "degraded", "error": type(exc).__name__} if default is None else default


class Aggregator:
    def __init__(self, repo, *, settings_provider, now_fn):
        self.repo = repo
        self._settings = settings_provider
        self._now = now_fn

    # -- service lookups (lazy; never import at module load) ------------------------

    def _tasks(self):
        from app.tasks.service import get_task_service
        return get_task_service()

    def _schedules(self):
        from app.schedules.service import get_schedule_service
        return get_schedule_service()

    def _broker(self):
        from app.core.pending import get_pending_broker
        return get_pending_broker()

    # -- health probes (fast, guarded) ---------------------------------------------

    def _probe_ollama(self, settings) -> dict:
        try:
            from app.llm.ollama_client import OllamaClient
            return {"status": "ok" if OllamaClient().is_available() else "unavailable"}
        except Exception:
            return {"status": "unavailable"}

    def _probe_voice(self, settings) -> dict:
        if not settings.enable_voice:
            return {"status": "disabled"}
        try:
            from app.voice.stt import get_stt_service
            svc = get_stt_service()
            return {"status": "ok" if svc.available() else "unavailable", "loaded": bool(svc.loaded)}
        except Exception:
            return {"status": "unavailable"}

    def _probe_voice_lab(self, settings) -> dict:
        if settings.tts_engine != "voice_lab":
            return {"status": "disabled"}
        try:
            import httpx
            r = httpx.get(f"{settings.voice_lab_url}/health", timeout=1.5)
            return {"status": "ok" if r.status_code == 200 else "unavailable"}
        except Exception:
            return {"status": "unavailable"}

    def services(self) -> dict:
        settings = self._settings()
        return {
            "api": {"status": "ok"},                       # we are serving this request
            "ollama": _safe(lambda: self._probe_ollama(settings), {"status": "unavailable"}),
            "voice": _safe(lambda: self._probe_voice(settings), {"status": "unavailable"}),
            "voice_lab": _safe(lambda: self._probe_voice_lab(settings), {"status": "unavailable"}),
            "wake": {"status": "enabled" if settings.enable_wake_word else "disabled"},
        }

    def _resources(self, settings) -> dict:
        """Best-effort RAM/VRAM summary — never a hard dependency, never a path."""
        ram = None
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram = {"used_pct": round(vm.percent, 1), "total_gb": round(vm.total / 1e9, 1)}
        except Exception:
            ram = None
        return {"ram": ram, "vram": None}                  # VRAM is surfaced via the tray/runtime

    # -- sections ------------------------------------------------------------------

    def pending(self) -> dict:
        from app.core import pending as pending_mod
        summary = self._broker().summary()
        if not summary:
            return {"pending": None}
        domain = summary.get("domain", "")
        target = summary.get("target")
        is_send = domain in ("email_send", "whatsapp_send")
        # An EXACT phrase for the UI confirm button — for recipient-sensitive sends
        # this includes the recipient (which the UI shows), never a generic confirm.
        confirm_phrase = pending_mod.confirm_phrase_for(summary)
        return {"pending": {
            "domain": domain,
            "action_id": summary.get("action_id"),
            "target": target,                         # app name or recipient (safe)
            "recipient": target if is_send else None,
            "required_confirmation_phrase": summary.get("required_confirmation_phrase"),
            "confirm_phrase": confirm_phrase,
            "expires_at": summary.get("expires_at"),
            "status": summary.get("status"),
        }}

    def tasks(self) -> dict:
        settings = self._settings()
        if not settings.enable_multi_step_tasks:
            return {"status": "disabled", "tasks": [], "active": None}
        svc = self._tasks()
        tasks = svc.list(settings, limit=25)
        active = None
        for t in tasks:
            if t["status"] in ("running", "awaiting_confirmation", "paused"):
                detail = svc.get(t["task_id"], settings) or t
                steps = detail.get("steps", [])
                blocked = next((s for s in steps if s["status"] in ("awaiting_confirmation", "failed")), None)
                active = {
                    "task_id": detail["task_id"], "title": detail["title"],
                    "status": detail["status"], "current_step": detail["current_step"],
                    "progress": f"{sum(1 for s in steps if s['status'] == 'completed')}/{len(steps)}",
                    "blocked_reason": (blocked or {}).get("confirmation_phrase") or (blocked or {}).get("error_code"),
                    "audit": [e["event"] for e in svc.audit(detail["task_id"], limit=6)],
                }
                break
        return {"tasks": tasks, "active": active}

    def schedules(self) -> dict:
        settings = self._settings()
        if not settings.enable_schedules:
            return {"status": "disabled", "schedules": [], "next": None}
        svc = self._schedules()
        result = svc.list({}, "es", settings)
        scheds = result.get("data", {}).get("schedules", [])
        active = [s for s in scheds if s["status"] == "active" and s.get("next_run_at")]
        nxt = min(active, key=lambda s: s["next_run_at"]) if active else None
        return {"schedules": scheds, "next": nxt}

    def errors(self, *, limit=25) -> dict:
        settings = self._settings()
        owner = settings.activity_owner
        events, _ = self.repo.query(owner, severity="error", limit=limit)
        warnings, _ = self.repo.query(owner, severity="warning", limit=limit)
        items = [normalize_error(e) for e in [*events, *warnings]][:limit]
        return {"count": len(items), "errors": items}

    def overview(self) -> dict:
        settings = self._settings()
        owner = settings.activity_owner
        services = self.services()
        pending = _safe(lambda: self.pending().get("pending"), None)
        tasks = _safe(lambda: self.tasks(), {"active": None})
        schedules = _safe(lambda: self.schedules(), {"next": None})
        since = (self._now().astimezone(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
        sev = _safe(lambda: self.repo.severity_counts(owner, since=since), {}) or {}
        # Overall state: degraded if any core service is unavailable.
        core_bad = any(services.get(k, {}).get("status") == "unavailable" for k in ("ollama",))
        fifi = "degraded" if core_bad else "ready"
        return {
            "fifi": fifi,
            "interaction_mode": "wake" if settings.enable_wake_word else (
                "ptt" if settings.enable_push_to_talk else "manual"),
            "active_voice": settings.voice_profile,
            "wake": services["wake"]["status"],
            "ptt": "enabled" if settings.enable_push_to_talk else "disabled",
            "services": services,
            "resources": _safe(lambda: self._resources(settings), {"ram": None, "vram": None}),
            "active_task": (tasks or {}).get("active"),
            "next_schedule": (schedules or {}).get("next"),
            "pending": pending,
            "recent_errors": sev.get("error", 0) + sev.get("warning", 0),
        }


# --- error normalization (item 9) --------------------------------------------------

_ERROR_MAP = {
    "FEATURE_DISABLED": ("service unavailable", ["retry_health"]),
    "BROWSER_UNAVAILABLE": ("service unavailable", ["retry_health", "restart_service"]),
    "LOGIN_REQUIRED": ("authentication required", ["open_login"]),
    "ACTION_EXPIRED": ("confirmation expired", ["cancel_pending"]),
    "TARGET_CHANGED": ("target changed", ["cancel_pending"]),
    "CONFIRMATION_MISMATCH": ("confirmation required", ["cancel_pending"]),
    "CONFIRMATION_REQUIRED": ("confirmation required", []),
    "TASK_TIMEOUT": ("task paused", ["cancel_pending"]),
    "STEP_FAILED": ("task paused", []),
    "MODEL_FALLBACK": ("model fallback", ["release_vram", "restart_service"]),
    "MIC_DISCONNECTED": ("microphone disconnected", ["retry_health"]),
}


def normalize_error(event) -> dict:
    code = event.error_code or ""
    label, actions = _ERROR_MAP.get(code, ("issue", ["retry_health"]))
    if event.domain == "schedules" and event.status in ("skipped", "failed"):
        label, actions = "schedule misfire", ["retry_health"]
    return {
        "event_id": event.event_id, "ts": event.ts, "domain": event.domain,
        "severity": event.severity, "label": label, "title": event.title,
        "summary": event.summary, "error_code": code, "recovery_actions": actions,
        "related_id": event.related_id,
    }
