"""Schedule + run models (Phase 5C). Pure dataclasses, enums, JSON row mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# --- trigger types -----------------------------------------------------------------
REMINDER = "reminder"          # notify only
ONE_TIME = "one_time"          # run a task template once
RECURRING = "recurring"        # run a task template on a recurrence rule
TRIGGER_TYPES = frozenset({REMINDER, ONE_TIME, RECURRING})

# --- schedule status ---------------------------------------------------------------
ACTIVE = "active"
PAUSED = "paused"
COMPLETED = "completed"
CANCELLED = "cancelled"
SCHEDULE_STATUSES = frozenset({ACTIVE, PAUSED, COMPLETED, CANCELLED})

# --- run status --------------------------------------------------------------------
RUN_DELIVERED = "delivered"    # reminder notified
RUN_COMPLETED = "completed"    # task ran to completion
RUN_PAUSED = "paused"          # task paused for a confirmation (effect prepared, not sent)
RUN_FAILED = "failed"
RUN_SKIPPED = "skipped"        # missed beyond grace / backlog cap / busy

# --- audit events ------------------------------------------------------------------
EV_PREPARED = "schedule_prepared"
EV_CREATED = "schedule_created"
EV_UPDATED = "schedule_updated"
EV_PAUSED = "schedule_paused"
EV_RESUMED = "schedule_resumed"
EV_CANCELLED = "schedule_cancelled"
EV_FIRED = "schedule_fired"
EV_LEASE = "lease_acquired"
EV_SKIPPED = "run_skipped"
EV_BACKLOG = "backlog_collapsed"
EV_COMPLETED = "schedule_completed"


@dataclass
class Schedule:
    schedule_id: str
    owner: str
    title: str
    timezone: str
    trigger_type: str
    next_run_at: str | None = None          # UTC iso
    recurrence: dict | None = None
    task_template: dict = field(default_factory=dict)   # {steps, reminder_text, ...}
    status: str = ACTIVE
    language: str = "es"
    last_run_at: str | None = None
    last_result: str = ""
    run_count: int = 0
    max_runs: int | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_row(self) -> dict:
        return {
            "schedule_id": self.schedule_id, "owner": self.owner, "title": self.title,
            "timezone": self.timezone, "trigger_type": self.trigger_type,
            "next_run_at": self.next_run_at,
            "recurrence": json.dumps(self.recurrence) if self.recurrence else None,
            "task_template": json.dumps(self.task_template, ensure_ascii=False),
            "status": self.status, "language": self.language,
            "last_run_at": self.last_run_at, "last_result": self.last_result,
            "run_count": self.run_count, "max_runs": self.max_runs,
            "lease_owner": self.lease_owner, "lease_expires_at": self.lease_expires_at,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row) -> "Schedule":
        d = dict(row)
        return cls(
            schedule_id=d["schedule_id"], owner=d["owner"], title=d["title"],
            timezone=d["timezone"], trigger_type=d["trigger_type"],
            next_run_at=d["next_run_at"],
            recurrence=json.loads(d["recurrence"]) if d["recurrence"] else None,
            task_template=json.loads(d["task_template"] or "{}"), status=d["status"],
            language=d.get("language", "es"), last_run_at=d["last_run_at"],
            last_result=d["last_result"] or "", run_count=d["run_count"],
            max_runs=d["max_runs"], lease_owner=d["lease_owner"],
            lease_expires_at=d["lease_expires_at"], created_at=d["created_at"],
            updated_at=d["updated_at"])

    def steps(self) -> list[dict]:
        return list(self.task_template.get("steps", []) or [])

    def reminder_text(self) -> str:
        return self.task_template.get("reminder_text", "") or self.title

    def public(self) -> dict:
        return {
            "schedule_id": self.schedule_id, "title": self.title, "timezone": self.timezone,
            "trigger_type": self.trigger_type, "next_run_at": self.next_run_at,
            "recurrence": self.recurrence, "status": self.status,
            "last_run_at": self.last_run_at, "last_result": self.last_result,
            "run_count": self.run_count, "max_runs": self.max_runs,
            "step_count": len(self.steps()),
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass
class ScheduleRun:
    run_id: str
    schedule_id: str
    scheduled_for: str | None
    ran_at: str
    status: str
    task_id: str | None = None
    summary: str = ""
    overdue: bool = False

    def to_row(self) -> dict:
        return {
            "run_id": self.run_id, "schedule_id": self.schedule_id,
            "scheduled_for": self.scheduled_for, "ran_at": self.ran_at,
            "status": self.status, "task_id": self.task_id, "summary": self.summary,
            "overdue": 1 if self.overdue else 0,
        }

    @classmethod
    def from_row(cls, row) -> "ScheduleRun":
        d = dict(row)
        return cls(run_id=d["run_id"], schedule_id=d["schedule_id"],
                   scheduled_for=d["scheduled_for"], ran_at=d["ran_at"], status=d["status"],
                   task_id=d["task_id"], summary=d["summary"] or "", overdue=bool(d["overdue"]))

    def public(self) -> dict:
        return {"run_id": self.run_id, "scheduled_for": self.scheduled_for, "ran_at": self.ran_at,
                "status": self.status, "task_id": self.task_id, "summary": self.summary,
                "overdue": self.overdue}
