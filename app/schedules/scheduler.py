"""The owned scheduler: due detection, atomic leasing, run policy (Phase 5C, item 6).

``poll_once`` is pure and deterministic (inject ``now``): it finds due schedules,
claims each with an atomic per-run lease, applies the missed-run/backlog policy,
fires reminders or runs task templates through the injected runner, then advances
next_run (recurrence) or completes — persisting before and after. ``run_forever`` is
a thin bounded-interval loop around it; it never busy-loops and, thanks to the lease
plus next_run advancement, never double-runs an occurrence across restarts or clock
changes. Effects are never auto-sent — that is enforced by the runner's autorun cap.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core.logger import get_logger
from app.schedules import models, recurrence
from app.schedules.models import (
    ACTIVE, COMPLETED, ONE_TIME, RECURRING, REMINDER, RUN_DELIVERED, RUN_SKIPPED,
    Schedule, ScheduleRun,
)

log = get_logger(__name__)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Scheduler:
    def __init__(self, repo, runner, *, now_fn=None, notifier=None,
                 settings_provider=get_settings, owner_token=None, sleep=None):
        self.repo = repo
        self.runner = runner            # fn(schedule, now, settings) -> {status, task_id, summary}
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.notifier = notifier or (lambda title, message, **kw: None)
        self._settings = settings_provider
        self.owner_token = owner_token or ("sched_" + uuid.uuid4().hex[:8])
        self._sleep = sleep
        self._running = False

    # -- polling -------------------------------------------------------------------

    def poll_once(self, now: datetime | None = None) -> list[dict]:
        now = now or self._now()
        settings = self._settings()
        out: list[dict] = []
        for schedule in self.repo.due(_iso(now), limit=settings.schedule_max_backlog_runs * 20 + 20):
            lease_ttl = max(60, settings.schedule_poll_seconds * 4)
            lease_until = _iso(now + timedelta(seconds=lease_ttl))
            if not self.repo.acquire_lease(schedule.schedule_id, self.owner_token, _iso(now), lease_until):
                continue                 # another worker owns this run
            try:
                self.repo.audit(models.EV_LEASE, schedule_id=schedule.schedule_id,
                                detail=self.owner_token)
                out.append(self._run(schedule.schedule_id, now, settings))
            finally:
                self.repo.release_lease(schedule.schedule_id)
        return out

    def _run(self, schedule_id: str, now: datetime, settings) -> dict:
        # Re-fetch under the lease so we act on fresh, authoritative state.
        schedule = self.repo.get(schedule_id)
        if schedule is None or schedule.status != ACTIVE or not schedule.next_run_at:
            return {"schedule_id": schedule_id, "status": "skipped", "reason": "not runnable"}
        if schedule.max_runs is not None and schedule.run_count >= schedule.max_runs:
            self._complete(schedule, now)
            return {"schedule_id": schedule_id, "status": "completed", "reason": "max_runs"}

        scheduled_for = schedule.next_run_at
        overdue_s = (now - datetime.fromisoformat(scheduled_for)).total_seconds()
        grace = settings.schedule_misfire_grace_seconds

        if schedule.trigger_type == REMINDER:
            result = self._fire_reminder(schedule, now, scheduled_for, overdue_s > grace)
        elif overdue_s > grace:
            # Missed a task beyond the grace window — never silently run stale work.
            result = self._skip(schedule, now, scheduled_for, "misfire_grace")
        else:
            result = self._fire_task(schedule, now, scheduled_for, settings)

        self._advance(schedule, now, settings, scheduled_for)
        self._emit_run(schedule, result)
        return {"schedule_id": schedule_id, **result}

    def _emit_run(self, schedule, result) -> None:
        """Additive Activity Center hook (Phase 5D) — a scheduled run outcome."""
        try:
            from app.core import eventbus

            status = result.get("status", "")
            eventbus.emit(
                domain="schedules", event_type="schedule_run", status=status,
                related_id=schedule.schedule_id, error_code=result.get("reason"),
                severity="warning" if status in ("skipped", "failed", "paused") else "info",
                title=f"Schedule: {schedule.title}",
                metadata={"title": schedule.title, "trigger_type": schedule.trigger_type,
                          "overdue": bool(result.get("overdue"))})
        except Exception:
            pass

    # -- reminders -----------------------------------------------------------------

    def _fire_reminder(self, schedule: Schedule, now, scheduled_for, overdue: bool) -> dict:
        text = schedule.reminder_text()
        prefix = "⏰ (overdue) " if overdue else "⏰ "
        self.notifier(f"Fifi reminder: {schedule.title}", f"{prefix}{text}",
                      key=f"reminder:{schedule.schedule_id}")
        run = ScheduleRun(run_id=self.repo.new_run_id(), schedule_id=schedule.schedule_id,
                          scheduled_for=scheduled_for, ran_at=_iso(now), status=RUN_DELIVERED,
                          summary=("overdue reminder delivered" if overdue else "reminder delivered"),
                          overdue=overdue)
        self.repo.record_run(run)
        schedule.run_count += 1
        schedule.last_run_at = _iso(now)
        schedule.last_result = run.summary
        self.repo.audit(models.EV_FIRED, schedule_id=schedule.schedule_id,
                        detail=f"reminder overdue={overdue}")
        return {"status": RUN_DELIVERED, "overdue": overdue, "summary": run.summary}

    # -- task templates ------------------------------------------------------------

    def _fire_task(self, schedule: Schedule, now, scheduled_for, settings) -> dict:
        result = self.runner(schedule, now, settings)   # revalidates + runs with autorun cap
        status = result.get("status", models.RUN_FAILED)
        summary = result.get("summary", "")
        run = ScheduleRun(run_id=self.repo.new_run_id(), schedule_id=schedule.schedule_id,
                          scheduled_for=scheduled_for, ran_at=_iso(now), status=status,
                          task_id=result.get("task_id"), summary=summary,
                          overdue=(now - datetime.fromisoformat(scheduled_for)).total_seconds() > 0)
        self.repo.record_run(run)
        schedule.run_count += 1
        schedule.last_run_at = _iso(now)
        schedule.last_result = summary
        self.repo.audit(models.EV_FIRED, schedule_id=schedule.schedule_id,
                        detail=f"task status={status} task_id={result.get('task_id')}")
        # Notify with the title + a SAFE summary (never secrets).
        self.notifier(f"Fifi task: {schedule.title}", summary or status,
                      key=f"schedule_task:{schedule.schedule_id}")
        return {"status": status, "task_id": result.get("task_id"), "summary": summary}

    def _skip(self, schedule: Schedule, now, scheduled_for, reason) -> dict:
        run = ScheduleRun(run_id=self.repo.new_run_id(), schedule_id=schedule.schedule_id,
                          scheduled_for=scheduled_for, ran_at=_iso(now), status=RUN_SKIPPED,
                          summary=f"skipped ({reason})", overdue=True)
        self.repo.record_run(run)
        schedule.last_run_at = _iso(now)
        schedule.last_result = run.summary
        self.repo.audit(models.EV_SKIPPED, schedule_id=schedule.schedule_id, detail=reason)
        return {"status": RUN_SKIPPED, "reason": reason, "summary": run.summary}

    # -- advance / complete --------------------------------------------------------

    def _advance(self, schedule: Schedule, now, settings, scheduled_for) -> None:
        if schedule.max_runs is not None and schedule.run_count >= schedule.max_runs:
            self._complete(schedule, now)
            return
        if schedule.recurrence:            # recurring reminders repeat too
            tz = recurrence.resolve_timezone(schedule.timezone) or timezone.utc
            # Advance to the next occurrence strictly AFTER now — collapsing any
            # missed backlog to a single jump (never a burst of catch-up runs).
            nxt = recurrence.compute_next_run(schedule.recurrence, now, tz)
            missed = 0
            probe = datetime.fromisoformat(scheduled_for)
            while probe < now and missed <= settings.schedule_max_backlog_runs + 1:
                probe = recurrence.compute_next_run(schedule.recurrence, probe, tz)
                missed += 1
            if missed > settings.schedule_max_backlog_runs:
                self.repo.audit(models.EV_BACKLOG, schedule_id=schedule.schedule_id,
                                detail=f"collapsed ~{missed} missed runs")
            schedule.lease_owner = None
            schedule.lease_expires_at = None
            schedule.next_run_at = _iso(nxt)
            self.repo.save(schedule)
        else:
            self._complete(schedule, now)

    def _complete(self, schedule: Schedule, now) -> None:
        schedule.status = COMPLETED
        schedule.next_run_at = None
        schedule.lease_owner = None
        schedule.lease_expires_at = None
        self.repo.save(schedule)
        self.repo.audit(models.EV_COMPLETED, schedule_id=schedule.schedule_id)

    # -- bounded polling loop (single owned process) -------------------------------

    def run_forever(self) -> None:  # pragma: no cover - exercised via poll_once in tests
        import time as _time

        sleep = self._sleep or _time.sleep
        self._running = True
        settings = self._settings()
        log.info("scheduler loop started (token=%s, poll=%ss)", self.owner_token,
                 settings.schedule_poll_seconds)
        while self._running:
            try:
                self.poll_once()
            except Exception:
                log.warning("scheduler poll failed", exc_info=True)
            sleep(max(1, self._settings().schedule_poll_seconds))   # bounded; never busy-loops

    def stop(self) -> None:
        self._running = False
