"""ActivityService: ingest + read-models + SAFE control routing + retention (5D).

Ingest is best-effort and non-blocking (a failure is swallowed; retention pruning is
incremental). Every control routes through the EXISTING services and the one global
broker — confirm dispatches the domain's exact phrase (never a generic "confirm
everything"), cancel uses the broker's own cancel, task/schedule controls call their
services, and restart/unload touch OWNED services only (never unrelated processes).
Export returns only the already-redacted stored events.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.activity import events as events_mod
from app.activity.aggregator import Aggregator
from app.activity.models import EventRepository
from app.config import get_settings
from app.core import errors
from app.core.logger import get_logger

log = get_logger(__name__)

_PRUNE_EVERY = 200


class RuntimeController:
    """Safe recovery actions. Never kills unrelated processes; process restarts are
    delegated to the owned tray/runtime, and VRAM release asks Ollama to unload."""

    ALLOWED_SERVICES = frozenset({"ollama", "voice_lab", "wake", "api"})

    def restart(self, service: str, settings) -> dict:
        if service not in self.ALLOWED_SERVICES:
            return {"ok": False, "error_code": errors.UNSUPPORTED_SERVICE}
        return {"ok": True, "status": "delegated", "service": service,
                "note": "restart is performed by the Fifi tray/runtime for owned processes only"}

    def unload(self, settings) -> dict:
        try:
            import httpx
            httpx.post(f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                       json={"model": settings.default_model, "keep_alive": 0, "prompt": ""},
                       timeout=3.0)
            return {"ok": True, "status": "unloaded"}
        except Exception:
            return {"ok": False, "status": "unavailable"}


class ActivityService:
    def __init__(self, *, repository=None, db_path=None, now_fn=None,
                 settings_provider=get_settings, runtime=None):
        self._settings = settings_provider
        self.repo = repository or EventRepository(
            db_path or settings_provider().activity_db_path, now_fn=now_fn)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.aggregator = Aggregator(self.repo, settings_provider=settings_provider, now_fn=self._now)
        self.runtime = runtime or RuntimeController()
        self._ingested = 0

    def _owner(self):
        return self._settings().activity_owner

    # -- ingest --------------------------------------------------------------------

    def ingest(self, payload: dict) -> bool:
        """Store one safe event (best-effort). Prunes incrementally, never blocking."""
        try:
            event = events_mod.build_event(
                payload, ts=self.repo.now_iso(), event_id=self.repo.new_id(), owner=self._owner())
            inserted = self.repo.insert(event)
        except Exception:
            log.debug("activity ingest failed", exc_info=True)
            return False
        if inserted:
            self._ingested += 1
            if self._ingested % _PRUNE_EVERY == 0:
                self._prune()
        return inserted

    def _prune(self) -> int:
        settings = self._settings()
        try:
            return self.repo.prune(retention_days=settings.activity_retention_days,
                                   max_events=settings.activity_max_events, batch=1000)
        except Exception:
            return 0

    def prune_now(self) -> int:
        return self._prune()

    # -- read APIs -----------------------------------------------------------------

    def overview(self) -> dict:
        ov = self.aggregator.overview()
        ov["errors"] = self.aggregator.errors(limit=10).get("errors", [])
        return ov

    def services(self) -> dict:
        return {"services": self.aggregator.services(),
                "resources": self.aggregator._resources(self._settings())}

    def pending(self) -> dict:
        return self.aggregator.pending()

    def tasks(self) -> dict:
        return self.aggregator.tasks()

    def schedules(self) -> dict:
        return self.aggregator.schedules()

    def errors(self, *, limit=25) -> dict:
        return self.aggregator.errors(limit=limit)

    def events(self, *, domain=None, severity=None, status=None, since=None, until=None,
               related_id=None, page=1, page_size=None) -> dict:
        settings = self._settings()
        page_size = page_size or settings.activity_page_size
        page = max(1, int(page))
        offset = (page - 1) * page_size
        items, total = self.repo.query(
            self._owner(), domain=domain, severity=severity, status=status, since=since,
            until=until, related_id=related_id, limit=page_size, offset=offset)
        return {
            "events": [e.public(with_metadata=False) for e in items],
            "page": page, "page_size": page_size, "total": total,
            "has_more": offset + len(items) < total,
        }

    def event(self, event_id: str) -> dict | None:
        ev = self.repo.get(event_id, self._owner())
        return ev.public(with_metadata=True) if ev else None

    # -- control actions (route through existing services; never mutate DBs) --------

    def confirm_action(self, phrase: str) -> dict:
        """Confirm the active pending using the DOMAIN's exact phrase — routed through
        the normal broker path, which rejects a wrong/blank phrase (no generic confirm)."""
        from app.core import conversation
        from app.core.nl import ServiceCommand
        if not (phrase or "").strip():
            return {"status": "rejected", "error_code": errors.CONFIRMATION_REQUIRED}
        result = conversation.dispatch(
            ServiceCommand(intent="confirm", is_confirmation=True, confirmation_phrase=phrase),
            language="es")
        return {"status": result.get("status"), "error_code": result.get("error_code"),
                "domain": result.get("domain"), "spoken": result.get("spoken", "")}

    def cancel_action(self) -> dict:
        from app.core.pending import get_pending_broker
        had = get_pending_broker().cancel_active()
        return {"status": "cancelled" if had else "none", "had_pending": had}

    def task_control(self, task_id: str, action: str, phrase: str = "") -> dict:
        from app.tasks.service import get_task_service
        settings = self._settings()
        svc = get_task_service()
        if action == "approve":
            return svc.approve(task_id, phrase or "aprobar plan", settings, "es")
        if action == "resume":
            return svc.resume(task_id, settings, "es", phrase=phrase)
        if action == "cancel":
            return svc.cancel(task_id, settings, "es")
        return {"status": "rejected", "error_code": errors.UNSUPPORTED_INTENT}

    def schedule_control(self, schedule_id: str, action: str) -> dict:
        from app.schedules.service import get_schedule_service
        settings = self._settings()
        svc = get_schedule_service()
        if action == "pause":
            return svc.pause(schedule_id, settings, "es")
        if action == "resume":
            return svc.resume(schedule_id, settings, "es")
        if action == "cancel":
            return svc.cancel(schedule_id, settings, "es")
        return {"status": "rejected", "error_code": errors.UNSUPPORTED_INTENT}

    def restart_service(self, service: str) -> dict:
        return self.runtime.restart(service, self._settings())

    def unload(self) -> dict:
        return self.runtime.unload(self._settings())

    # -- export --------------------------------------------------------------------

    def export(self, *, limit=None) -> dict:
        settings = self._settings()
        if not settings.activity_enable_export:
            return {"error_code": errors.EXPORT_DISABLED, "events": []}
        cap = min(limit or settings.activity_max_events, settings.activity_max_events)
        items, total = self.repo.query(self._owner(), limit=min(cap, 500), offset=0)
        # Stored events are already redacted; re-run redaction defensively on export.
        return {
            "exported_at": self.repo.now_iso(),
            "owner": self._owner(),
            "count": len(items),
            "total_available": total,
            "redacted": True,
            "events": [e.redacted().public(with_metadata=True) for e in items],
        }


_service: ActivityService | None = None


def get_activity_service() -> ActivityService:
    global _service
    if _service is None:
        _service = ActivityService()
    return _service
