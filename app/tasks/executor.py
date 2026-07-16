"""Deterministic, resumable task execution (Phase 5B, items 5-8).

The executor advances a task by dispatching each step through the SAME
``conversation.dispatch`` path a direct command uses — so every existing safety
gate (the one global broker, per-domain exact confirmations, recipient/page/target/
hash revalidation) is reused, never bypassed. Read-only and local steps run
automatically after approval; every external effect pauses in the broker and
requires its own domain phrase (a plain "sí", a wake, and plan approval can never
confirm an effect). One task executes effects at a time. State is persisted after
every transition; completed effects are protected by an idempotency guard and never
run twice; cancellation stops future steps without pretending a done effect rolled
back; timeouts/failures pause with a stable error; only read-only steps retry.
"""

from __future__ import annotations

from datetime import datetime

from app.core import conversation, errors
from app.core.logger import get_logger
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.tasks import models
from app.tasks.models import (
    APPROVED, AWAITING_CONFIRMATION, CANCELLED, COMPLETED, EFFECT, FAILED, PAUSED,
    PLANNED, READ_ONLY, RISK_RANK, RUNNING, STEP_AWAITING, STEP_CANCELLED,
    STEP_COMPLETED, STEP_FAILED, STEP_PENDING, Task,
)

# The interactive default: auto-run every risk level, pausing only at each step's
# own confirmation. A scheduled run passes a lower cap (Phase 5C).
AUTORUN_ALL = RISK_RANK[EFFECT]

log = get_logger(__name__)

# Approval phrases (task-level; NOT broker confirmations — approval never confirms
# an effect).
APPROVAL_PHRASES = frozenset({"aprobar plan", "ejecutar este plan", "approve plan",
                              "ejecutar el plan", "run this plan"})

_SUCCESS_STATUS = {"executed", "simulated", "filled", "sent", "placed", "ok", "completed"}
# States that count a task as holding the single effect slot.
_ACTIVE_STATES = (RUNNING, AWAITING_CONFIRMATION, PAUSED)


def is_approval_phrase(text: str) -> bool:
    return " ".join((text or "").strip().lower().split()) in APPROVAL_PHRASES


def _out(situation, *, code=None, task=None, phrase="", data=None):
    return {"situation": situation, "code": code,
            "task": task.public() if task else None,
            "phrase": phrase, "data": data or {}}


class TaskExecutor:
    def __init__(self, repo):
        self.repo = repo

    # -- helpers -------------------------------------------------------------------

    def _broker(self):
        return get_pending_broker()

    def _now(self) -> datetime:
        return self.repo._now()

    def _other_active_task(self, task: Task) -> str | None:
        for other in self.repo.list_tasks(task.owner, statuses=_ACTIVE_STATES, limit=50):
            if other.task_id != task.task_id:
                return other.task_id
        return None

    def _timed_out(self, task: Task, settings) -> bool:
        if not task.started_at:
            return False
        started = datetime.fromisoformat(task.started_at)
        elapsed = (self._now().astimezone(started.tzinfo) - started).total_seconds()
        return elapsed > settings.task_max_runtime_minutes * 60

    def _next_runnable(self, task: Task):
        done = {s.position for s in task.steps if s.status == STEP_COMPLETED}
        for step in sorted(task.steps, key=lambda s: s.position):
            if step.status != STEP_PENDING:
                continue
            if all(dep in done for dep in step.dependencies):
                return step
        return None

    def _result_summary(self, step, result) -> str:
        # The services guarantee ``spoken`` carries no secrets — safe to store.
        spoken = (result.get("spoken") or "").strip()
        base = spoken[:160] if spoken else result.get("status", "")
        return f"{step.intent}: {base}".strip()

    def _dispatch_step(self, step, task, settings) -> dict:
        cmd = ServiceCommand(intent=step.intent, arguments=dict(step.arguments))
        return conversation.dispatch(cmd, language=task.language, settings=settings)

    @staticmethod
    def _classify(result) -> str:
        status = result.get("status")
        state = result.get("state")
        if status == "needs_confirmation" or state == errors.AWAITING_CONFIRMATION:
            return "pause"
        if status in _SUCCESS_STATUS or state == errors.COMPLETED:
            return "done"
        return "fail"

    # -- approve -------------------------------------------------------------------

    def approve(self, task: Task, phrase: str, settings) -> dict:
        if task.status == models.BLOCKED:
            return _out("plan_blocked", code=errors.INVALID_TASK_STATE, task=task)
        if task.status not in (PLANNED,):
            return _out("invalid_state", code=errors.INVALID_TASK_STATE, task=task)
        if settings.task_require_plan_approval and not is_approval_phrase(phrase):
            return _out("approval_mismatch", code=errors.APPROVAL_MISMATCH, task=task)
        task.status = APPROVED
        task.approved_at = self.repo.now_iso()
        self.repo.save_task(task)
        self.repo.audit(models.EV_PLAN_APPROVED, task_id=task.task_id,
                        detail=f"phrase={phrase!r}")
        return _out("approved", task=task)

    # -- run / resume core ---------------------------------------------------------

    def run(self, task: Task, settings, *, max_autorun_risk: int = AUTORUN_ALL) -> dict:
        if task.status in (COMPLETED, CANCELLED):
            return _out("already_final", code=errors.TASK_ALREADY_COMPLETED, task=task)
        if task.status == FAILED:
            return _out("invalid_state", code=errors.INVALID_TASK_STATE, task=task)
        if settings.task_require_plan_approval and task.status == PLANNED:
            return _out("not_approved", code=errors.PLAN_NOT_APPROVED, task=task)
        busy = self._other_active_task(task)
        if busy:
            return _out("busy", code=errors.TASK_BUSY, task=task, data={"active_task": busy})
        if not task.started_at:
            task.started_at = self.repo.now_iso()
        if task.status in (APPROVED, PLANNED, PAUSED, RUNNING):
            task.status = RUNNING
            self.repo.save_task(task)
        return self._advance(task, settings, max_autorun_risk=max_autorun_risk)

    def resume(self, task: Task, settings, *, phrase: str = "") -> dict:
        if task.status in (COMPLETED, CANCELLED, FAILED):
            return _out("already_final", code=errors.INVALID_TASK_STATE, task=task)
        busy = self._other_active_task(task)
        if busy:
            return _out("busy", code=errors.TASK_BUSY, task=task, data={"active_task": busy})

        if task.status == AWAITING_CONFIRMATION:
            awaiting = next((s for s in task.steps if s.status == STEP_AWAITING), None)
            active = self._broker().active()
            if awaiting is None:
                task.status = RUNNING
            elif active is None or active.get("action_id") != awaiting.action_id:
                # The pending was lost (e.g. a restart cleared the memory-only
                # broker). Re-prepare THIS step — completed steps stay 'completed',
                # so nothing already done repeats.
                awaiting.status = STEP_PENDING
                awaiting.action_id = None
                self.repo.save_step(awaiting)
                task.status = RUNNING
                self.repo.save_task(task)
                return self._advance(task, settings)
            elif phrase:
                # Route the domain confirmation through the normal path; the
                # effect listener marks the step, then we reload and advance.
                conversation.dispatch(
                    ServiceCommand(intent="confirm", is_confirmation=True,
                                   confirmation_phrase=phrase),
                    language=task.language, settings=settings)
                task = self.repo.get_task(task.task_id)
                refreshed = next((s for s in task.steps if s.step_id == awaiting.step_id), None)
                if refreshed and refreshed.status != STEP_COMPLETED:
                    # Wrong/expired phrase — stay paused (a plain yes never confirms).
                    return _out("needs_confirmation_phrase", code=errors.CONFIRMATION_REQUIRED,
                                task=task, phrase=refreshed.confirmation_phrase)
            else:
                # A bare "continuar" cannot confirm an effect.
                return _out("needs_confirmation_phrase", code=errors.CONFIRMATION_REQUIRED,
                            task=task, phrase=awaiting.confirmation_phrase)

        if task.status in (PAUSED, AWAITING_CONFIRMATION, RUNNING, APPROVED):
            task.status = RUNNING
            self.repo.save_task(task)
        return self._advance(task, settings)

    def _advance(self, task: Task, settings, *, max_autorun_risk: int = AUTORUN_ALL) -> dict:
        while True:
            if self._timed_out(task, settings):
                return self._fail(task, errors.TASK_TIMEOUT, models.EV_TASK_TIMEOUT)

            step = self._next_runnable(task)
            if step is None:
                return self._complete(task)

            # Autorun cap (Phase 5C): a scheduled run defers any step above its cap
            # WITHOUT dispatching it — the effect is never even prepared, let alone
            # sent. The user later resumes interactively (cap = AUTORUN_ALL).
            if RISK_RANK.get(step.risk_level, 0) > max_autorun_risk:
                step.status = STEP_AWAITING
                step.action_id = None
                self.repo.save_step(step)
                task.status = AWAITING_CONFIRMATION
                task.current_step = step.position
                self.repo.save_task(task)
                self.repo.audit(models.EV_STEP_PAUSED, task_id=task.task_id, step_id=step.step_id,
                                detail=f"autorun_cap:{step.risk_level}")
                return _out("paused_for_confirmation", task=task,
                            phrase=step.confirmation_phrase, data={"step": step.public(),
                                                                   "deferred": True})

            max_attempts = (1 + settings.task_max_step_retries
                            if step.risk_level == READ_ONLY else 1)
            step.attempt_count += 1
            step.started_at = step.started_at or self.repo.now_iso()
            self.repo.audit(models.EV_STEP_STARTED, task_id=task.task_id, step_id=step.step_id,
                            detail=f"{step.intent} attempt={step.attempt_count}")
            result = self._dispatch_step(step, task, settings)
            kind = self._classify(result)

            if kind == "pause":
                broker_summary = self._broker().summary() or {}
                step.status = STEP_AWAITING
                step.action_id = result.get("action_id") or broker_summary.get("action_id")
                step.confirmation_phrase = (broker_summary.get("required_confirmation_phrase")
                                            or step.confirmation_phrase)
                step.result_summary = self._result_summary(step, result)
                self.repo.save_step(step)
                task.status = AWAITING_CONFIRMATION
                task.current_step = step.position
                self.repo.save_task(task)
                self.repo.audit(models.EV_STEP_PAUSED, task_id=task.task_id, step_id=step.step_id,
                                detail=step.confirmation_phrase)
                return _out("paused_for_confirmation", task=task, phrase=step.confirmation_phrase,
                            data={"step": step.public()})

            if kind == "done":
                step.status = STEP_COMPLETED
                step.result_summary = self._result_summary(step, result)
                step.completed_at = self.repo.now_iso()
                self.repo.save_step(step)
                task.current_step = step.position + 1
                self.repo.save_task(task)
                self.repo.audit(models.EV_STEP_COMPLETED, task_id=task.task_id,
                                step_id=step.step_id, detail=step.result_summary[:120])
                continue

            # failure
            if step.attempt_count < max_attempts:
                self.repo.audit(models.EV_STEP_RETRIED, task_id=task.task_id, step_id=step.step_id,
                                detail=result.get("error_code") or "retry")
                self.repo.save_step(step)
                continue   # read-only retry (re-dispatch same step)
            step.status = STEP_FAILED
            step.error_code = result.get("error_code") or errors.STEP_FAILED
            step.result_summary = self._result_summary(step, result)
            self.repo.save_step(step)
            return self._fail(task, step.error_code, models.EV_STEP_FAILED, step=step)

    # -- effect-completion listener (external confirmations) -----------------------

    def on_effect_complete(self, result, action_id) -> None:
        """Called after ANY confirmation resolves. Reconciles the confirmed pending
        with the task step that was waiting on it (idempotent)."""
        if not action_id:
            return
        step = self.repo.step_by_action_id(action_id)
        if step is None:
            return   # not a task-owned effect
        task = self.repo.get_task(step.task_id)
        if task is None or task.status not in (AWAITING_CONFIRMATION, RUNNING):
            return
        live = next((s for s in task.steps if s.step_id == step.step_id), None)
        if live is None or live.status != STEP_AWAITING:
            return

        status = result.get("status")
        code = result.get("error_code")
        if status in _SUCCESS_STATUS:
            live.status = STEP_COMPLETED
            live.completed_at = self.repo.now_iso()
            live.action_id = None
            live.result_summary = f"{live.intent}: confirmed ({status})"
            self.repo.save_step(live)
            task.status = PAUSED               # ready to resume; user says "continuar"
            task.current_step = live.position + 1
            self.repo.save_task(task)
            self.repo.audit(models.EV_STEP_CONFIRMED, task_id=task.task_id, step_id=live.step_id)
            self.repo.audit(models.EV_EFFECT_EXECUTED, task_id=task.task_id, step_id=live.step_id,
                            detail=status)
        elif code == errors.ACTION_EXPIRED:
            live.status = STEP_FAILED
            live.error_code = errors.ACTION_EXPIRED
            live.action_id = None
            self.repo.save_step(live)
            self._fail(task, errors.ACTION_EXPIRED, models.EV_STEP_FAILED, step=live)
        # A mismatch (wrong phrase / plain "sí") leaves the step awaiting.

    # -- cancel --------------------------------------------------------------------

    def cancel(self, task: Task, settings) -> dict:
        if task.status in (COMPLETED, CANCELLED, FAILED):
            return _out("already_final", code=errors.INVALID_TASK_STATE, task=task)
        broker = self._broker()
        active = broker.active()
        for step in task.steps:
            if step.status == STEP_AWAITING and active and active.get("action_id") == step.action_id:
                broker.cancel_active()     # release the effect slot this task holds
            if step.status in (STEP_PENDING, STEP_AWAITING):
                step.status = STEP_CANCELLED
                step.action_id = None
                self.repo.save_step(step)
        task.status = CANCELLED
        task.completed_at = self.repo.now_iso()
        self.repo.save_task(task)
        self.repo.audit(models.EV_TASK_CANCELLED, task_id=task.task_id)
        # Completed effects are NOT rolled back — cancellation only stops the future.
        return _out("cancelled", task=task)

    # -- terminal transitions ------------------------------------------------------

    def _complete(self, task: Task) -> dict:
        completed = [s for s in task.steps if s.status == STEP_COMPLETED]
        task.status = COMPLETED
        task.completed_at = self.repo.now_iso()
        task.final_summary = "; ".join(s.result_summary for s in completed if s.result_summary)[:1000]
        self.repo.save_task(task)
        self.repo.audit(models.EV_TASK_COMPLETED, task_id=task.task_id,
                        detail=f"steps={len(completed)}")
        return _out("task_completed", task=task)

    def _fail(self, task: Task, code, event, *, step=None) -> dict:
        task.status = FAILED
        task.error_code = code
        task.completed_at = self.repo.now_iso()
        self.repo.save_task(task)
        self.repo.audit(event, task_id=task.task_id,
                        step_id=step.step_id if step else None, detail=code)
        if event != models.EV_STEP_FAILED:
            self.repo.audit(models.EV_TASK_FAILED, task_id=task.task_id, detail=code)
        return _out("task_failed", code=code, task=task)
