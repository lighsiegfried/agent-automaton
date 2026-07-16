"""TaskService: plan → approve → run → pause/confirm → resume → complete (Phase 5B).

Ties the planner, validator, repository, and executor together, maps executor
outcomes to safe bilingual replies + a normalized dispatcher result, and registers
the task-level intents (plan/approve/resume/cancel), NL patterns, and the effect-
completion listener with the core dispatcher. Importing the domain services here
guarantees their intents/schemas are registered so validation can run.
"""

from __future__ import annotations

import time
import uuid

from app.config import get_settings
from app.core import conversation, errors, nl
from app.core.logger import get_logger
from app.tasks import executor as executor_mod
from app.tasks import models, planner, validator
from app.tasks.executor import TaskExecutor
from app.tasks.models import Task
from app.tasks.repository import TaskRepository

# Importing the domain services registers their intents + argument schemas so the
# validator can check them. They have no dependency on app.tasks (no import cycle).
import app.memory.service  # noqa: F401,E402
import app.text.service  # noqa: F401,E402
import app.browser.service  # noqa: F401,E402
import app.integrations.whatsapp.service  # noqa: F401,E402
import app.integrations.email_web.service  # noqa: F401,E402

log = get_logger(__name__)

# situation -> (normalized status, command state, spoken key)
_SITUATIONS = {
    "plan_ready": ("ok", errors.AWAITING_CONFIRMATION, "plan_ready"),
    "plan_blocked": ("rejected", errors.BLOCKED, "plan_blocked"),
    "approved": ("ok", errors.EXECUTING, "approved"),
    "paused_for_confirmation": ("needs_confirmation", errors.AWAITING_CONFIRMATION, "paused"),
    "step_confirmed": ("ok", errors.EXECUTING, "step_confirmed"),
    "needs_confirmation_phrase": ("needs_confirmation", errors.AWAITING_CONFIRMATION, "need_phrase"),
    "task_completed": ("executed", errors.COMPLETED, "completed"),
    "task_failed": ("error", errors.FAILED, "failed"),
    "cancelled": ("cancelled", errors.COMPLETED, "cancelled"),
    "busy": ("rejected", errors.BLOCKED, "busy"),
    "not_approved": ("rejected", errors.BLOCKED, "not_approved"),
    "approval_mismatch": ("rejected", errors.BLOCKED, "approval_mismatch"),
    "invalid_state": ("rejected", errors.BLOCKED, "invalid_state"),
    "already_final": ("rejected", errors.BLOCKED, "already_final"),
    "disabled": ("rejected", errors.BLOCKED, "disabled"),
    "not_found": ("rejected", errors.BLOCKED, "not_found"),
    "no_active_task": ("rejected", errors.BLOCKED, "no_active_task"),
}

_SPOKEN = {
    "plan_ready": ("I prepared a {count}-step plan. Say 'aprobar plan' to approve — I'll only "
                   "run the safe steps and pause before anything is sent.",
                   "Preparé un plan de {count} paso(s). Di 'aprobar plan' para aprobarlo — solo "
                   "ejecutaré los pasos seguros y me detendré antes de enviar nada."),
    "plan_blocked": ("I can't run that plan — some steps aren't allowed.",
                     "No puedo ejecutar ese plan — algunos pasos no están permitidos."),
    "approved": ("Plan approved. Running the safe steps now.",
                 "Plan aprobado. Ejecutando los pasos seguros."),
    "paused": ("I paused before a sensitive step. Say '{phrase}' to proceed — approval alone "
               "won't do it.",
               "Me detuve antes de un paso sensible. Di '{phrase}' para continuar — la aprobación "
               "sola no basta."),
    "step_confirmed": ("Done. Say 'continuar tarea' to keep going.",
                       "Listo. Di 'continuar tarea' para seguir."),
    "need_phrase": ("That step needs its exact confirmation: '{phrase}'. A plain 'yes' won't do it.",
                    "Ese paso necesita su confirmación exacta: '{phrase}'. Un simple 'sí' no basta."),
    "completed": ("The task is complete.", "La tarea está completa."),
    "failed": ("The task stopped: {code}.", "La tarea se detuvo: {code}."),
    "cancelled": ("Task cancelled. Steps already done were not undone.",
                  "Tarea cancelada. Los pasos ya realizados no se deshacen."),
    "busy": ("Another task is currently running its effects. Finish or cancel it first.",
             "Otra tarea está ejecutando sus efectos. Termínala o cancélala primero."),
    "not_approved": ("Approve the plan first with 'aprobar plan'.",
                     "Aprueba el plan primero con 'aprobar plan'."),
    "approval_mismatch": ("To approve, say exactly 'aprobar plan'.",
                          "Para aprobar, di exactamente 'aprobar plan'."),
    "invalid_state": ("That task isn't in a state I can do that to.",
                      "Esa tarea no está en un estado donde pueda hacer eso."),
    "already_final": ("That task is already finished.", "Esa tarea ya terminó."),
    "disabled": ("Multi-step tasks are turned off.", "Las tareas de varios pasos están desactivadas."),
    "not_found": ("I couldn't find that task.", "No encontré esa tarea."),
    "no_active_task": ("There's no active task.", "No hay ninguna tarea activa."),
    "failed_generic": ("That didn't work.", "Eso no funcionó."),
}


def _say(key, language, **ctx):
    en, es = _SPOKEN.get(key, _SPOKEN["failed_generic"])
    template = en if str(language).lower().startswith("en") else es
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


class TaskService:
    def __init__(self, *, repository=None, db_path=None, now_fn=None, id_factory=None,
                 clock=time.monotonic):
        if repository is None:
            repository = TaskRepository(
                db_path or get_settings().tasks_db_path, now_fn=now_fn, id_factory=id_factory)
        self.repo = repository
        self._executor = TaskExecutor(repository)

    # -- result shaping ------------------------------------------------------------

    def _owner(self, settings):
        return settings.task_owner

    def _result(self, outcome, language):
        situation = outcome["situation"]
        status, state, spoken_key = _SITUATIONS.get(
            situation, ("error", errors.FAILED, "failed_generic"))
        task = outcome.get("task") or {}
        phrase = outcome.get("phrase", "")
        code = outcome.get("code") or (task.get("error_code") if isinstance(task, dict) else None)
        spoken = _say(spoken_key, language, phrase=phrase,
                      count=len(task.get("steps", [])) if isinstance(task, dict) else 0,
                      code=code or "")
        return {"status": status, "error_code": code, "domain": "tasks",
                "action_id": task.get("task_id") if isinstance(task, dict) else None,
                "state": state, "spoken": spoken, "required_phrase": phrase,
                "data": {"task": task, "required_phrase": phrase, **outcome.get("data", {})}}

    def _disabled(self, settings, language):
        if not settings.enable_multi_step_tasks:
            return self._result(_o("disabled", code=errors.TASKS_DISABLED), language)
        return None

    def _load(self, task_id, language):
        task = self.repo.get_task(task_id)
        if task is None:
            return None, self._result(_o("not_found", code=errors.TASK_NOT_FOUND), language)
        return task, None

    # -- plan ----------------------------------------------------------------------

    def plan(self, args, language, settings, broker=None):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        request = (args.get("request") or args.get("user_request") or "").strip()
        explicit = args.get("steps")
        proposed = planner.propose(request, settings, explicit_steps=explicit,
                                   title=args.get("title", ""))
        task_id = self.repo.new_task_id()
        result = validator.validate_plan(proposed.raw_steps, task_id=task_id,
                                         max_steps=settings.task_max_steps)

        for step in result.steps:
            step.step_id = self.repo.new_step_id()
            step.idempotency_key = step.compute_idempotency_key()
        task = Task(task_id=task_id, owner=owner, user_request=request,
                    title=proposed.title or request[:80] or "task",
                    status=models.PLANNED if result.ok else models.BLOCKED,
                    language="en" if str(language).lower().startswith("en") else "es",
                    max_steps=settings.task_max_steps, memory_refs=proposed.memory_refs,
                    steps=result.steps)
        self.repo.create_task(task)
        self.repo.audit(models.EV_PLAN_CREATED, task_id=task_id,
                        detail=f"source={proposed.source} steps={len(result.steps)} ok={result.ok}")
        if proposed.memory_refs:
            self.repo.audit(models.EV_MEMORY_CONTEXT, task_id=task_id,
                            detail=f"memory_refs={len(proposed.memory_refs)}")

        preview = self._plan_preview(task, result)
        if not result.ok:
            self.repo.audit(models.EV_PLAN_BLOCKED, task_id=task_id, detail=result.error_code)
            out = _o("plan_blocked", code=result.error_code, task=task, data=preview)
        else:
            out = _o("plan_ready", task=task, data=preview)
        return self._result(out, language)

    def _plan_preview(self, task, result) -> dict:
        return {
            "plan": [
                {"position": s.position, "intent": s.intent, "domain": s.domain,
                 "classification": s.risk_level,
                 "sensitive": s.risk_level in (models.LOCAL, models.EFFECT),
                 "requires_confirmation": s.requires_confirmation,
                 "expected_confirmation": s.confirmation_phrase,
                 "dependencies": s.dependencies}
                for s in task.steps
            ],
            "blocked_steps": [
                {"position": b.position, "intent": b.intent, "code": b.code, "reason": b.reason}
                for b in result.blocked
            ],
            "memory_refs": list(task.memory_refs),
            "reason": result.reason,
        }

    # -- lifecycle (explicit task_id; used by API) ---------------------------------

    def approve(self, task_id, phrase, settings, language):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task, err = self._load(task_id, language)
        if err:
            return err
        return self._result(self._executor.approve(task, phrase, settings), language)

    def run(self, task_id, settings, language):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task, err = self._load(task_id, language)
        if err:
            return err
        return self._result(self._executor.run(task, settings), language)

    def resume(self, task_id, settings, language, *, phrase=""):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task, err = self._load(task_id, language)
        if err:
            return err
        return self._result(self._executor.resume(task, settings, phrase=phrase), language)

    def cancel(self, task_id, settings, language):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task, err = self._load(task_id, language)
        if err:
            return err
        return self._result(self._executor.cancel(task, settings), language)

    def run_from_template(self, steps, *, request, language, settings, owner,
                          max_autorun_risk, memory_refs=None, title=""):
        """Create a fresh task from a validated template and run it with an autorun
        cap (Phase 5C scheduler). The plan is REVALIDATED here every time, so a
        template that references a now-unsupported intent is rejected at run time.
        Pre-approved (the schedule creation was confirmed) — but effects still pause.
        Returns {ok, code?, task_id?, outcome?}."""
        task_id = self.repo.new_task_id()
        result = validator.validate_plan(steps, task_id=task_id, max_steps=settings.task_max_steps)
        if not result.ok:
            return {"ok": False, "code": result.error_code, "reason": result.reason,
                    "task_id": None}
        for step in result.steps:
            step.step_id = self.repo.new_step_id()
            step.idempotency_key = step.compute_idempotency_key()
        task = Task(
            task_id=task_id, owner=owner, user_request=request,
            title=title or request[:80] or "scheduled task", status=models.APPROVED,
            language="en" if str(language).lower().startswith("en") else "es",
            max_steps=settings.task_max_steps, memory_refs=memory_refs or [],
            steps=result.steps, approved_at=self.repo.now_iso())
        self.repo.create_task(task)
        self.repo.audit(models.EV_PLAN_CREATED, task_id=task_id,
                        detail=f"source=scheduled steps={len(result.steps)}")
        self.repo.audit(models.EV_PLAN_APPROVED, task_id=task_id, detail="scheduled")
        outcome = self._executor.run(task, settings, max_autorun_risk=max_autorun_risk)
        return {"ok": True, "task_id": task_id, "outcome": outcome,
                "task": self.repo.get_task(task_id).public()}

    def get(self, task_id, settings):
        task = self.repo.get_task(task_id)
        return task.public() if task else None

    def list(self, settings, *, limit=50):
        owner = self._owner(settings)
        return [t.public(include_steps=False) for t in self.repo.list_tasks(owner, limit=limit)]

    def audit(self, task_id, *, limit=100):
        return self.repo.audit_recent(task_id, limit=limit)

    # -- voice helpers (operate on the active task) --------------------------------

    def _active(self, settings, statuses):
        owner = self._owner(settings)
        tasks = self.repo.list_tasks(owner, statuses=statuses, limit=1)
        return self.repo.get_task(tasks[0].task_id) if tasks else None

    def voice_approve(self, args, language, settings, broker=None):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task = self._active(settings, [models.PLANNED])
        if task is None:
            return self._result(_o("no_active_task"), language)
        approved = self._executor.approve(task, args.get("phrase", "aprobar plan"), settings)
        if approved["situation"] != "approved":
            return self._result(approved, language)
        # Approval lets the safe steps begin immediately (item 5).
        task = self.repo.get_task(task.task_id)
        return self._result(self._executor.run(task, settings), language)

    def voice_resume(self, args, language, settings, broker=None):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task = self._active(settings, [models.AWAITING_CONFIRMATION, models.PAUSED,
                                       models.RUNNING, models.APPROVED])
        if task is None:
            return self._result(_o("no_active_task"), language)
        return self._result(self._executor.resume(task, settings, phrase=args.get("phrase", "")),
                            language)

    def voice_cancel(self, args, language, settings, broker=None):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        task = self._active(settings, [models.AWAITING_CONFIRMATION, models.PAUSED,
                                       models.RUNNING, models.APPROVED, models.PLANNED])
        if task is None:
            return self._result(_o("no_active_task"), language)
        return self._result(self._executor.cancel(task, settings), language)


def _o(situation, *, code=None, task=None, phrase="", data=None):
    return {"situation": situation, "code": code,
            "task": task.public() if isinstance(task, Task) else task,
            "phrase": phrase, "data": data or {}}


_service: TaskService | None = None


def get_task_service() -> TaskService:
    global _service
    if _service is None:
        _service = TaskService()
    return _service


def _svc():
    return get_task_service()


def _effect_listener(result, action_id) -> None:
    # Only reconcile once a task service actually exists — never create the tasks DB
    # for an unrelated (text/browser/email) confirmation.
    if _service is None:
        return
    try:
        _service._executor.on_effect_complete(result, action_id)
    except Exception:
        log.warning("task effect reconciliation failed", exc_info=True)


def _register() -> None:
    conversation.register_effect_listener(_effect_listener)

    conversation.register_service_intent(
        "task_plan", {"request", "user_request", "steps", "title"},
        lambda a, l, s, b: _svc().plan(a, l, s, b))
    conversation.register_service_intent(
        "task_approve", {"phrase"}, lambda a, l, s, b: _svc().voice_approve(a, l, s, b))
    conversation.register_service_intent(
        "task_resume", {"phrase"}, lambda a, l, s, b: _svc().voice_resume(a, l, s, b))
    conversation.register_service_intent(
        "task_cancel", set(), lambda a, l, s, b: _svc().voice_cancel(a, l, s, b))

    import re
    nl.register_patterns([
        (re.compile(r"\b(?:prepara|haz|arma|crea)\s+un\s+plan\s+(?:para|de)\s+(.+)", re.I),
         "task_plan", lambda m: {"request": m.group(1).strip()}),
        (re.compile(r"\bplanifica\s+(.+)", re.I),
         "task_plan", lambda m: {"request": m.group(1).strip()}),
        (re.compile(r"\b(?:prepare|make|create)\s+a\s+plan\s+(?:to|for)\s+(.+)", re.I),
         "task_plan", lambda m: {"request": m.group(1).strip()}),
        (re.compile(r"\b(?:aprobar|aprueba)\s+(?:el\s+|este\s+)?plan\b", re.I),
         "task_approve", lambda m: {"phrase": "aprobar plan"}),
        (re.compile(r"\bejecuta(?:r)?\s+(?:este|el)\s+plan\b", re.I),
         "task_approve", lambda m: {"phrase": "ejecutar este plan"}),
        (re.compile(r"\bapprove\s+(?:the\s+)?plan\b", re.I),
         "task_approve", lambda m: {"phrase": "approve plan"}),
        (re.compile(r"\brun\s+this\s+plan\b", re.I),
         "task_approve", lambda m: {"phrase": "run this plan"}),
        (re.compile(r"\b(?:continuar|contin[uú]a|reanuda(?:r)?)\s+(?:con\s+)?(?:la\s+)?tarea\b", re.I),
         "task_resume", lambda m: {}),
        (re.compile(r"\b(?:continue|resume)\s+(?:the\s+)?task\b", re.I),
         "task_resume", lambda m: {}),
        (re.compile(r"\bcancela(?:r)?\s+(?:la\s+)?tarea\b", re.I),
         "task_cancel", lambda m: {}),
        (re.compile(r"\bcancel\s+(?:the\s+)?task\b", re.I),
         "task_cancel", lambda m: {}),
    ])


_register()
