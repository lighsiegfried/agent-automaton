"""ScheduleService: sensitive create/confirm + lifecycle + the scheduler runner (5C).

Creating or modifying a schedule is a sensitive write: ``prepare`` resolves the
absolute date (refusing ambiguous ones), validates the task template, and registers
a pending broker action; only the exact phrase ("confirmar programación" /
"crear recordatorio" / "confirm schedule") — never a plain "sí" or a wake — persists
it. The runner the scheduler calls REVALIDATES the plan and recomputes bounded memory
context at every run, then runs it through the Phase-5B executor with an autorun cap
so read-only/local steps auto-run while every external effect pauses in the broker
for a fresh domain confirmation.

Importing this module registers the schedule intents, NL patterns, and broker domain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core import conversation, errors, nl, pending
from app.core.logger import get_logger
from app.schedules import models, recurrence
from app.schedules.models import (
    ACTIVE, CANCELLED, ONE_TIME, PAUSED, RECURRING, REMINDER, Schedule)
from app.schedules.repository import ScheduleRepository
from app.schedules.scheduler import Scheduler
from app.tasks.models import EFFECT, LOCAL, READ_ONLY, RISK_RANK

# Registering task/domain services makes their intent schemas available so the
# template validator can run (no import cycle — they don't import app.schedules).
import app.tasks.service  # noqa: F401,E402

log = get_logger(__name__)

WRITE_DOMAIN = "schedule_write"
CONFIRM_PHRASES = frozenset({"confirmar programación", "confirmar programacion",
                             "crear recordatorio", "confirm schedule", "crear programación"})

_SPOKEN = {
    "schedule_ready": (
        "I'll schedule \"{title}\" for {when}. Say 'confirmar programación' to set it — "
        "I won't send anything without a fresh confirmation at run time.",
        "Programaré «{title}» para {when}. Di 'confirmar programación' para fijarlo — "
        "no enviaré nada sin una confirmación nueva al ejecutarse."),
    "scheduled": ("Scheduled \"{title}\" for {when}.", "Programé «{title}» para {when}."),
    "ambiguous": ("That date is ambiguous — {reason}.", "Esa fecha es ambigua — {reason}."),
    "invalid_date": ("I couldn't resolve that date — {reason}.",
                     "No pude resolver esa fecha — {reason}."),
    "invalid_recurrence": ("I couldn't understand that recurrence.",
                           "No entendí esa recurrencia."),
    "invalid_timezone": ("That timezone isn't valid.", "Esa zona horaria no es válida."),
    "empty": ("There's nothing to schedule.", "No hay nada que programar."),
    "wake_blocked": ("A wake word can't confirm a schedule — say the confirmation yourself.",
                     "Una palabra de activación no puede confirmar una programación — dila tú."),
    "mismatch": ("That's not the confirmation I need for the pending schedule.",
                 "Esa no es la confirmación que necesito para la programación pendiente."),
    "nothing_pending": ("There's no pending schedule to confirm.",
                        "No hay programación pendiente que confirmar."),
    "expired": ("That schedule proposal expired. Please ask again.",
                "Esa propuesta de programación expiró. Pídemelo de nuevo."),
    "paused": ("Paused \"{title}\".", "Pausé «{title}»."),
    "resumed": ("Resumed \"{title}\" — next run {when}.", "Reanudé «{title}» — próxima {when}."),
    "cancelled": ("Cancelled \"{title}\".", "Cancelé «{title}»."),
    "list_summary": ("You have {count} schedule(s).", "Tienes {count} programación(es)."),
    "not_found": ("I couldn't find that schedule.", "No encontré esa programación."),
    "disabled": ("Schedules are turned off.", "Las programaciones están desactivadas."),
    "failed": ("That didn't work.", "Eso no funcionó."),
}


def _say(key, language, **ctx):
    en, es = _SPOKEN.get(key, _SPOKEN["failed"])
    template = en if str(language).lower().startswith("en") else es
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def _r(status, *, state, spoken, code=None, action_id=None, data=None):
    return {"status": status, "error_code": code, "domain": "schedules",
            "action_id": action_id, "state": state, "spoken": spoken, "data": data or {}}


def _autorun_cap(settings) -> int:
    cap = -1
    if settings.schedule_allow_read_only_autorun:
        cap = max(cap, RISK_RANK[READ_ONLY])
    if settings.schedule_allow_local_draft_autorun:
        cap = max(cap, RISK_RANK[LOCAL])
    if settings.schedule_allow_external_effect_autorun:
        cap = max(cap, RISK_RANK[EFFECT])
    return cap


class ScheduleService:
    def __init__(self, *, repository=None, db_path=None, now_fn=None, id_factory=None,
                 task_service_provider=None, memory_service_provider=None, notifier=None):
        self.repo = repository or ScheduleRepository(
            db_path or get_settings().schedules_db_path, now_fn=now_fn, id_factory=id_factory)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._task_service = task_service_provider
        self._memory_service = memory_service_provider
        self.notifier = notifier or self._default_notifier
        self._pending: Schedule | None = None
        self._pending_expires: str | None = None
        self.scheduler = Scheduler(self.repo, self._run_template, now_fn=self._now,
                                   notifier=self._notify)

    # -- helpers -------------------------------------------------------------------

    def _owner(self, settings):
        return settings.schedule_owner

    def _disabled(self, settings, language):
        if not settings.enable_schedules:
            return _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULES_DISABLED,
                      spoken=_say("disabled", language))
        return None

    def _default_notifier(self, title, message, **kw):
        log.info("schedule notify: %s — %s", title, message)

    def _notify(self, title, message, **kw):
        try:
            self.notifier(title, message, **kw)
        except Exception:
            log.warning("schedule notifier failed", exc_info=True)

    def _tasks(self):
        if self._task_service:
            return self._task_service()
        from app.tasks.service import get_task_service
        return get_task_service()

    def _memory(self):
        if self._memory_service:
            return self._memory_service()
        from app.memory.service import get_memory_service
        return get_memory_service()

    def _clear_pending(self):
        self._pending = None
        self._pending_expires = None

    # -- prepare (sensitive) -------------------------------------------------------

    def prepare(self, args, language, settings, broker):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        owner = self._owner(settings)
        tz_name = args.get("timezone") or settings.schedule_timezone
        if recurrence.resolve_timezone(tz_name) is None:
            return _r("rejected", state=errors.BLOCKED, code=errors.INVALID_TIMEZONE,
                      spoken=_say("invalid_timezone", language))
        kind = args.get("kind") or ("reminder" if args.get("reminder_text") else "task")

        # Resolve the when + title/action.
        title = args.get("title", "").strip()
        recurrence_rule = args.get("recurrence")
        run_at = args.get("run_at")            # explicit UTC iso (API)
        local_display = ""
        if run_at is None:
            parsed = recurrence.parse_when(args.get("text", "") or args.get("when", ""),
                                           tz_name=tz_name, now_utc=self._now())
            if not parsed.ok:
                key = {errors.AMBIGUOUS_DATE: "ambiguous",
                       errors.INVALID_RECURRENCE: "invalid_recurrence",
                       errors.INVALID_TIMEZONE: "invalid_timezone"}.get(parsed.code, "invalid_date")
                return _r("rejected", state=errors.BLOCKED, code=parsed.code,
                          spoken=_say(key, language, reason=parsed.reason))
            run_at = parsed.run_at_utc
            recurrence_rule = recurrence_rule or parsed.recurrence
            local_display = parsed.local_display
            title = title or parsed.remainder
        local_display = local_display or recurrence.format_local(run_at, tz_name)
        title = title or "recordatorio"

        # Build template + validate task steps (so the preview lists which steps will
        # need confirmation later, and a bad template is rejected up front).
        steps = args.get("steps")
        template: dict = {"language": "en" if str(language).lower().startswith("en") else "es"}
        planned: list[dict] = []
        if kind == "reminder" and not steps:
            trigger_type = RECURRING if recurrence_rule else REMINDER
            template["reminder_text"] = args.get("reminder_text") or title
        else:
            if steps is None:
                from app.tasks import planner
                steps = planner.heuristic_steps(title)
            from app.tasks import validator
            result = validator.validate_plan(steps, task_id="tmpl", max_steps=settings.task_max_steps)
            if not result.ok:
                return _r("rejected", state=errors.BLOCKED, code=result.error_code,
                          spoken=_say("failed", language),
                          data={"blocked_steps": [b.__dict__ for b in result.blocked],
                                "reason": result.reason})
            if not result.steps:
                return _r("rejected", state=errors.BLOCKED, code=errors.EMPTY_SCHEDULE,
                          spoken=_say("empty", language))
            template["steps"] = [{"intent": s.intent, "arguments": s.arguments,
                                  "dependencies": s.dependencies} for s in result.steps]
            planned = [{"position": s.position, "intent": s.intent, "domain": s.domain,
                        "classification": s.risk_level,
                        "requires_confirmation": s.requires_confirmation,
                        "expected_confirmation": s.confirmation_phrase} for s in result.steps]
            trigger_type = RECURRING if recurrence_rule else ONE_TIME

        schedule = Schedule(
            schedule_id=args.get("target_id") or self.repo.new_schedule_id(), owner=owner,
            title=title[:200], timezone=tz_name, trigger_type=trigger_type, next_run_at=run_at,
            recurrence=recurrence_rule, task_template=template,
            language=template["language"], max_runs=args.get("max_runs"))
        self._pending = schedule
        self._pending_expires = (self._now() + timedelta(
            seconds=settings.memory_action_expires_seconds)).isoformat(timespec="seconds")
        broker.register(domain=WRITE_DOMAIN, action_id=schedule.schedule_id,
                        target=title, expires_at=self._pending_expires, cancel=self._clear_pending)
        self.repo.audit(models.EV_PREPARED, schedule_id=schedule.schedule_id,
                        detail=f"{trigger_type} at {run_at}")
        preview = {
            "schedule_id": schedule.schedule_id, "title": title, "trigger_type": trigger_type,
            "timezone": tz_name, "when_local": local_display, "next_run_at": run_at,
            "recurrence": recurrence_rule, "planned_steps": planned,
            "steps_requiring_confirmation": [p for p in planned if p["requires_confirmation"]],
            "max_runs": schedule.max_runs,
        }
        return _r("needs_confirmation", state=errors.AWAITING_CONFIRMATION,
                  action_id=schedule.schedule_id,
                  spoken=_say("schedule_ready", language, title=title, when=local_display),
                  data=preview)

    def confirm_create(self, action_id, phrase, language, broker, wake=False):
        if wake:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_MISMATCH,
                      spoken=_say("wake_blocked", language))
        sched = self._pending
        if sched is None or sched.schedule_id != action_id:
            return _r("rejected", state=errors.BLOCKED, code=errors.CONFIRMATION_REQUIRED,
                      spoken=_say("nothing_pending", language))
        if self._pending_expires and self._now().isoformat() >= self._pending_expires:
            self._clear_pending(); broker.clear(action_id)
            return _r("rejected", state=errors.BLOCKED, code=errors.ACTION_EXPIRED,
                      spoken=_say("expired", language))
        existing = self.repo.get(sched.schedule_id)
        if existing is not None:
            # Update in place — historical runs (separate rows) are never mutated.
            sched.run_count = existing.run_count
            sched.created_at = existing.created_at
            self.repo.save(sched)
            self.repo.audit(models.EV_UPDATED, schedule_id=sched.schedule_id, detail=sched.trigger_type)
        else:
            self.repo.create(sched)
            self.repo.audit(models.EV_CREATED, schedule_id=sched.schedule_id, detail=sched.trigger_type)
        self._clear_pending(); broker.clear(action_id)
        when = recurrence.format_local(sched.next_run_at, sched.timezone) if sched.next_run_at else ""
        return _r("executed", state=errors.COMPLETED, action_id=sched.schedule_id,
                  spoken=_say("scheduled", language, title=sched.title, when=when),
                  data={"schedule": sched.public()})

    # -- lifecycle -----------------------------------------------------------------

    def list(self, args, language, settings, broker=None):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        scheds = self.repo.list(self._owner(settings), statuses=(ACTIVE, PAUSED))
        return _r("ok", state=errors.COMPLETED,
                  spoken=_say("list_summary", language, count=len(scheds)),
                  data={"count": len(scheds), "schedules": [s.public() for s in scheds]})

    def get(self, schedule_id, settings):
        s = self.repo.get(schedule_id)
        if s is None or s.owner != self._owner(settings):
            return None
        data = s.public()
        data["runs"] = [r.public() for r in self.repo.runs(schedule_id, limit=20)]
        return data

    def _lifecycle(self, schedule_id, settings, language, action):
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        s = self.repo.get(schedule_id)
        if s is None or s.owner != self._owner(settings):
            return _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULE_NOT_FOUND,
                      spoken=_say("not_found", language))
        return action(s)

    def pause(self, schedule_id, settings, language):
        def _do(s):
            if s.status not in (ACTIVE,):
                return _r("rejected", state=errors.BLOCKED, code=errors.INVALID_SCHEDULE_STATE,
                          spoken=_say("failed", language))
            s.status = PAUSED
            self.repo.save(s)
            self.repo.audit(models.EV_PAUSED, schedule_id=s.schedule_id)
            return _r("ok", state=errors.COMPLETED, spoken=_say("paused", language, title=s.title),
                      data={"schedule": s.public()})
        return self._lifecycle(schedule_id, settings, language, _do)

    def resume(self, schedule_id, settings, language):
        def _do(s):
            if s.status != PAUSED:
                return _r("rejected", state=errors.BLOCKED, code=errors.INVALID_SCHEDULE_STATE,
                          spoken=_say("failed", language))
            s.status = ACTIVE
            # If the next run is in the past (missed while paused), roll a recurring
            # schedule forward to the next future occurrence.
            if s.recurrence and s.next_run_at and datetime.fromisoformat(s.next_run_at) < self._now():
                tz = recurrence.resolve_timezone(s.timezone) or timezone.utc
                s.next_run_at = recurrence.compute_next_run(s.recurrence, self._now(), tz)\
                    .astimezone(timezone.utc).isoformat(timespec="seconds")
            self.repo.save(s)
            self.repo.audit(models.EV_RESUMED, schedule_id=s.schedule_id)
            when = recurrence.format_local(s.next_run_at, s.timezone) if s.next_run_at else ""
            return _r("ok", state=errors.COMPLETED,
                      spoken=_say("resumed", language, title=s.title, when=when),
                      data={"schedule": s.public()})
        return self._lifecycle(schedule_id, settings, language, _do)

    def cancel(self, schedule_id, settings, language):
        def _do(s):
            s.status = CANCELLED
            s.next_run_at = None
            self.repo.save(s)
            self.repo.audit(models.EV_CANCELLED, schedule_id=s.schedule_id)
            return _r("cancelled", state=errors.COMPLETED,
                      spoken=_say("cancelled", language, title=s.title),
                      data={"schedule": s.public()})
        return self._lifecycle(schedule_id, settings, language, _do)

    def update_prepare(self, schedule_id, args, language, settings, broker):
        """Modifying a schedule is sensitive too — re-prepare with the SAME id so
        confirmation updates it in place while its run history is preserved."""
        blocked = self._disabled(settings, language)
        if blocked:
            return blocked
        existing = self.repo.get(schedule_id)
        if existing is None or existing.owner != self._owner(settings):
            return _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULE_NOT_FOUND,
                      spoken=_say("not_found", language))
        merged = dict(args)
        merged["target_id"] = schedule_id
        merged.setdefault("timezone", existing.timezone)
        return self.prepare(merged, language, settings, broker)

    def runs(self, schedule_id, settings):
        return [r.public() for r in self.repo.runs(schedule_id, limit=50)]

    def audit(self, schedule_id=None, *, limit=100):
        return self.repo.audit_recent(schedule_id, limit=limit)

    def run_now(self, schedule_id, settings, language):
        """Force a due-run of one schedule immediately (tray 'Run now')."""
        def _do(s):
            self.scheduler._run(s.schedule_id, self._now(), settings)
            return _r("ok", state=errors.COMPLETED, spoken="",
                      data={"schedule": self.get(s.schedule_id, settings)})
        return self._lifecycle(schedule_id, settings, language, _do)

    # -- the scheduler's task runner ------------------------------------------------

    def _run_template(self, schedule: Schedule, now, settings) -> dict:
        """Called by the scheduler for a task schedule. Revalidates the plan and
        recomputes bounded memory context at RUN time, then runs with an autorun cap
        (read-only/local auto-run; effects pause for a fresh domain confirmation)."""
        cap = _autorun_cap(settings)
        memory_refs: list[str] = []
        if settings.enable_memory:
            try:
                bundle = self._memory().context_bundle(
                    schedule.title, settings, max_items=settings.task_context_max_memories,
                    max_chars=settings.task_context_max_chars)
                memory_refs = [m["id"] for m in bundle.get("memories", [])]
            except Exception:
                memory_refs = []
        result = self._tasks().run_from_template(
            schedule.steps(), request=schedule.title, language=schedule.language,
            settings=settings, owner=settings.task_owner, max_autorun_risk=cap,
            memory_refs=memory_refs, title=schedule.title)
        if not result.get("ok"):
            return {"status": models.RUN_FAILED, "summary": f"plan rejected: {result.get('code')}"}
        outcome = result["outcome"]
        task = result.get("task") or {}
        sit = outcome.get("situation")
        status = {"task_completed": models.RUN_COMPLETED,
                  "paused_for_confirmation": models.RUN_PAUSED,
                  "busy": models.RUN_SKIPPED,
                  "task_failed": models.RUN_FAILED}.get(sit, models.RUN_FAILED)
        steps = task.get("steps", [])
        done = sum(1 for s in steps if s["status"] == "completed")
        summary = f"{task.get('status', sit)} ({done}/{len(steps)} steps)"
        return {"status": status, "task_id": result.get("task_id"), "summary": summary}


_service: ScheduleService | None = None


def get_schedule_service() -> ScheduleService:
    global _service
    if _service is None:
        _service = ScheduleService()
    return _service


def _svc():
    return get_schedule_service()


def _register() -> None:
    pending.register_domain(WRITE_DOMAIN, phrases=CONFIRM_PHRASES,
                            primary="confirmar programación / confirm schedule")
    conversation.register_confirm_handler(
        WRITE_DOMAIN,
        lambda aid, phrase, lang, b, wake=False: _svc().confirm_create(aid, phrase, lang, b, wake=wake))

    for intent, keys, method in [
        ("schedule_prepare",
         {"text", "when", "title", "kind", "steps", "reminder_text", "recurrence",
          "run_at", "timezone", "max_runs"}, "_dispatch_prepare"),
        ("schedule_list", {"query"}, "list"),
        ("schedule_pause", {"schedule_id"}, "_dispatch_pause"),
        ("schedule_resume", {"schedule_id"}, "_dispatch_resume"),
        ("schedule_cancel", {"schedule_id"}, "_dispatch_cancel"),
    ]:
        conversation.register_service_intent(
            intent, keys, (lambda m: lambda a, l, s, b: getattr(_svc(), m)(a, l, s, b))(method))

    import re
    nl.register_patterns([
        (re.compile(r"\b(?:recu[eé]rdame|recu[eé]rdame\s+que|recordarme)\s+(.+)", re.I),
         "schedule_prepare", lambda m: {"text": m.group(1).strip(), "kind": "reminder"}),
        (re.compile(r"\b((?:cada|todos\s+los|todas\s+las|every|each)\s+.+)", re.I),
         "schedule_prepare", lambda m: {"text": m.group(1).strip(), "kind": "task"}),
        (re.compile(r"\b(?:pausar|pausa)\s+(?:el|la)?\s*(?:recordatorio|programaci[oó]n|tarea\s+programada)\b", re.I),
         "schedule_pause", lambda m: {}),
        (re.compile(r"\bpause\s+(?:the\s+)?(?:reminder|schedule)\b", re.I),
         "schedule_pause", lambda m: {}),
        (re.compile(r"\b(?:reanuda(?:r)?|contin[uú]a)\s+(?:el|la)?\s*(?:recordatorio|programaci[oó]n)\b", re.I),
         "schedule_resume", lambda m: {}),
        (re.compile(r"\bresume\s+(?:the\s+)?(?:reminder|schedule)\b", re.I),
         "schedule_resume", lambda m: {}),
        (re.compile(r"\bcancela(?:r)?\s+(?:el|la)?\s*(?:recordatorio|programaci[oó]n)\b", re.I),
         "schedule_cancel", lambda m: {}),
        (re.compile(r"\bcancel\s+(?:the\s+)?(?:reminder|schedule)\b", re.I),
         "schedule_cancel", lambda m: {}),
        (re.compile(r"\b(?:qu[eé]\s+tengo\s+programado|mis\s+(?:recordatorios|programaciones)|"
                    r"what(?:'s|\s+is)\s+scheduled|what\s+do\s+i\s+have\s+scheduled)\b", re.I),
         "schedule_list", lambda m: {}),
    ])


# --- voice dispatch helpers (operate on the active schedule) -----------------------

def _bind_voice():
    def _dispatch_prepare(self, args, language, settings, broker):
        return self.prepare(args, language, settings, broker)

    def _active(self, settings):
        return self.repo.latest(self._owner(settings), statuses=(ACTIVE, PAUSED))

    def _dispatch_pause(self, args, language, settings, broker):
        if args.get("schedule_id"):
            return self.pause(args["schedule_id"], settings, language)
        s = _active(self, settings)
        return self.pause(s.schedule_id, settings, language) if s else \
            _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULE_NOT_FOUND,
               spoken=_say("not_found", language))

    def _dispatch_resume(self, args, language, settings, broker):
        if args.get("schedule_id"):
            return self.resume(args["schedule_id"], settings, language)
        s = self.repo.latest(self._owner(settings), statuses=(PAUSED,))
        return self.resume(s.schedule_id, settings, language) if s else \
            _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULE_NOT_FOUND,
               spoken=_say("not_found", language))

    def _dispatch_cancel(self, args, language, settings, broker):
        if args.get("schedule_id"):
            return self.cancel(args["schedule_id"], settings, language)
        s = _active(self, settings)
        return self.cancel(s.schedule_id, settings, language) if s else \
            _r("rejected", state=errors.BLOCKED, code=errors.SCHEDULE_NOT_FOUND,
               spoken=_say("not_found", language))

    ScheduleService._dispatch_prepare = _dispatch_prepare
    ScheduleService._dispatch_pause = _dispatch_pause
    ScheduleService._dispatch_resume = _dispatch_resume
    ScheduleService._dispatch_cancel = _dispatch_cancel


_bind_voice()
_register()
