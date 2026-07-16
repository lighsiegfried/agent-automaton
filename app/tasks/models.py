"""Task + step models and the authoritative step registry (Phase 5B).

Pure dataclasses, enums, and the intent → (domain, risk) registry that says which
compositions are supported and how each step is classified. No SQLite, no service
logic — so the validator, planner, executor, and tests all import this without a
cycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# --- risk classification (item 3) --------------------------------------------------
READ_ONLY = "read_only"      # pure reads; may auto-run on approval; retried once
LOCAL = "local"              # local drafting/composition; auto-runs, then pauses
EFFECT = "effect"            # external effect (typing/sending); always pauses

RISK_LEVELS = (READ_ONLY, LOCAL, EFFECT)
# Ordered severity — a scheduled run auto-executes steps up to a configured cap and
# defers anything above it (Phase 5C). Interactive runs use the top rank (EFFECT).
RISK_RANK = {READ_ONLY: 0, LOCAL: 1, EFFECT: 2}

# --- task / step lifecycle ---------------------------------------------------------
# task status
PLANNED = "planned"                    # validated, awaiting approval
BLOCKED = "blocked"                    # plan contains unsupported/invalid steps
APPROVED = "approved"                  # approved, not yet run
RUNNING = "running"                    # advancing (transient / ready to continue)
AWAITING_CONFIRMATION = "awaiting_confirmation"   # paused at an effect step
PAUSED = "paused"                      # paused, ready to resume (no confirmation needed)
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

TASK_STATUSES = frozenset({PLANNED, BLOCKED, APPROVED, RUNNING, AWAITING_CONFIRMATION,
                           PAUSED, COMPLETED, FAILED, CANCELLED})
TASK_TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})

# step status
STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_AWAITING = "awaiting_confirmation"
STEP_COMPLETED = "completed"
STEP_FAILED = "failed"
STEP_SKIPPED = "skipped"
STEP_CANCELLED = "cancelled"

# --- audit events ------------------------------------------------------------------
EV_PLAN_CREATED = "plan_created"
EV_PLAN_BLOCKED = "plan_blocked"
EV_PLAN_APPROVED = "plan_approved"
EV_MEMORY_CONTEXT = "memory_context"
EV_STEP_STARTED = "step_started"
EV_STEP_COMPLETED = "step_completed"
EV_STEP_PAUSED = "step_paused"
EV_STEP_CONFIRMED = "step_confirmed"
EV_STEP_FAILED = "step_failed"
EV_STEP_SKIPPED = "step_skipped"
EV_STEP_RETRIED = "step_retried"
EV_EFFECT_EXECUTED = "effect_executed"
EV_TASK_COMPLETED = "task_completed"
EV_TASK_FAILED = "task_failed"
EV_TASK_CANCELLED = "task_cancelled"
EV_TASK_TIMEOUT = "task_timeout"

# --- authoritative step registry ---------------------------------------------------
# intent -> (domain, risk). ONLY these intents may appear in a plan. Everything
# else — memory writes, form fill, purchases, deletion, attachments, shell, account
# changes, confirmations themselves — is rejected by the validator. Confirmations
# are user actions through the broker, never plan steps.
SUPPORTED_INTENTS: dict[str, tuple[str, str]] = {
    # read-only
    "memory_search": ("memory", READ_ONLY),
    "memory_list": ("memory", READ_ONLY),
    "browser_open": ("browser", READ_ONLY),
    "browser_search": ("browser", READ_ONLY),
    "browser_read": ("browser", READ_ONLY),
    "browser_summarize": ("browser", READ_ONLY),
    "browser_find": ("browser", READ_ONLY),
    "browser_scroll": ("browser", READ_ONLY),
    "browser_open_link": ("browser", READ_ONLY),
    "browser_close": ("browser", READ_ONLY),
    "email_status": ("email", READ_ONLY),
    "email_search": ("email", READ_ONLY),
    "email_open_thread": ("email", READ_ONLY),
    "email_summarize_thread": ("email", READ_ONLY),
    "email_close": ("email", READ_ONLY),
    "whatsapp_status": ("whatsapp", READ_ONLY),
    "whatsapp_find_contact": ("whatsapp", READ_ONLY),
    "whatsapp_open_chat": ("whatsapp", READ_ONLY),
    "whatsapp_close": ("whatsapp", READ_ONLY),
    # local drafting / composition (auto-runs on approval, then pauses for its
    # placement/insertion confirmation — never an external effect on its own)
    "draft_text": ("text", LOCAL),
    "rewrite_text": ("text", LOCAL),
    "whatsapp_draft_message": ("whatsapp", LOCAL),
    "email_draft_new": ("email", LOCAL),
    "email_draft_reply": ("email", LOCAL),
    # external effect (insertion / send preparation — pauses for the final
    # domain-specific confirmation; approval never pre-confirms these)
    "type_text": ("text", EFFECT),
    "append_text": ("text", EFFECT),
    "replace_selected_text": ("text", EFFECT),
    "whatsapp_prepare_send": ("whatsapp", EFFECT),
    "email_prepare_send": ("email", EFFECT),
}


def is_supported(intent: str) -> bool:
    return intent in SUPPORTED_INTENTS


def domain_of(intent: str) -> str:
    return SUPPORTED_INTENTS.get(intent, ("", ""))[0]


def risk_of(intent: str) -> str:
    return SUPPORTED_INTENTS.get(intent, ("", ""))[1]


def requires_confirmation(intent: str) -> bool:
    return risk_of(intent) in (LOCAL, EFFECT)


def _recipients(args: dict) -> str:
    value = args.get("to") or args.get("recipient") or args.get("recipients") or ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def expected_confirmation_phrase(intent: str, args: dict) -> str:
    """The domain phrase the user will be asked for at pause time (best-effort;
    the executor uses the broker's actual required phrase when it pauses)."""
    domain, risk = SUPPORTED_INTENTS.get(intent, ("", ""))
    if risk == READ_ONLY:
        return ""
    if domain == "text":
        return "confirmar escritura / insert text"
    if intent in ("whatsapp_draft_message", "email_draft_new", "email_draft_reply"):
        return "colocar borrador / place draft"
    if intent in ("whatsapp_prepare_send", "email_prepare_send"):
        who = _recipients(args)
        return f"confirmar envío a {who}" if who else "confirmar envío a <destinatario>"
    return ""


@dataclass
class Step:
    step_id: str
    task_id: str
    position: int
    intent: str
    arguments: dict = field(default_factory=dict)
    domain: str = ""
    risk_level: str = READ_ONLY
    dependencies: list[int] = field(default_factory=list)   # positions of prior steps
    status: str = STEP_PENDING
    attempt_count: int = 0
    idempotency_key: str = ""
    result_summary: str = ""
    requires_confirmation: bool = False
    confirmation_phrase: str = ""
    action_id: str | None = None          # broker action id while awaiting confirmation
    error_code: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def compute_idempotency_key(self) -> str:
        basis = json.dumps({"t": self.task_id, "p": self.position, "i": self.intent,
                            "a": self.arguments}, sort_keys=True, ensure_ascii=False)
        return "idem_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]

    def to_row(self) -> dict:
        return {
            "step_id": self.step_id, "task_id": self.task_id, "position": self.position,
            "intent": self.intent, "arguments": json.dumps(self.arguments, ensure_ascii=False),
            "domain": self.domain, "risk_level": self.risk_level,
            "dependencies": json.dumps(self.dependencies), "status": self.status,
            "attempt_count": self.attempt_count, "idempotency_key": self.idempotency_key,
            "result_summary": self.result_summary,
            "requires_confirmation": 1 if self.requires_confirmation else 0,
            "confirmation_phrase": self.confirmation_phrase, "action_id": self.action_id,
            "error_code": self.error_code, "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_row(cls, row) -> "Step":
        d = dict(row)
        return cls(
            step_id=d["step_id"], task_id=d["task_id"], position=d["position"],
            intent=d["intent"], arguments=json.loads(d["arguments"] or "{}"),
            domain=d["domain"], risk_level=d["risk_level"],
            dependencies=json.loads(d["dependencies"] or "[]"), status=d["status"],
            attempt_count=d["attempt_count"], idempotency_key=d["idempotency_key"],
            result_summary=d["result_summary"] or "",
            requires_confirmation=bool(d["requires_confirmation"]),
            confirmation_phrase=d["confirmation_phrase"] or "", action_id=d["action_id"],
            error_code=d["error_code"], started_at=d["started_at"],
            completed_at=d["completed_at"],
        )

    def public(self) -> dict:
        return {
            "step_id": self.step_id, "position": self.position, "intent": self.intent,
            "domain": self.domain, "risk_level": self.risk_level,
            "dependencies": list(self.dependencies), "status": self.status,
            "attempt_count": self.attempt_count,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_phrase": self.confirmation_phrase,
            "result_summary": self.result_summary, "error_code": self.error_code,
        }


@dataclass
class Task:
    task_id: str
    owner: str
    user_request: str
    title: str
    status: str = PLANNED
    language: str = "es"
    created_at: str = ""
    updated_at: str = ""
    current_step: int = 0
    max_steps: int = 12
    memory_refs: list[str] = field(default_factory=list)   # context snapshot references
    final_summary: str = ""
    error_code: str | None = None
    approved_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    steps: list[Step] = field(default_factory=list)        # not persisted inline

    def to_row(self) -> dict:
        return {
            "task_id": self.task_id, "owner": self.owner, "user_request": self.user_request,
            "title": self.title, "status": self.status, "language": self.language,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "current_step": self.current_step, "max_steps": self.max_steps,
            "memory_refs": json.dumps(self.memory_refs), "final_summary": self.final_summary,
            "error_code": self.error_code, "approved_at": self.approved_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
        }

    @classmethod
    def from_row(cls, row) -> "Task":
        d = dict(row)
        return cls(
            task_id=d["task_id"], owner=d["owner"], user_request=d["user_request"],
            title=d["title"], status=d["status"], language=d.get("language", "es"),
            created_at=d["created_at"], updated_at=d["updated_at"],
            current_step=d["current_step"], max_steps=d["max_steps"],
            memory_refs=json.loads(d["memory_refs"] or "[]"),
            final_summary=d["final_summary"] or "", error_code=d["error_code"],
            approved_at=d["approved_at"], started_at=d["started_at"],
            completed_at=d["completed_at"],
        )

    def public(self, *, include_steps=True) -> dict:
        data = {
            "task_id": self.task_id, "user_request": self.user_request, "title": self.title,
            "status": self.status, "created_at": self.created_at, "updated_at": self.updated_at,
            "current_step": self.current_step, "max_steps": self.max_steps,
            "memory_refs": list(self.memory_refs), "final_summary": self.final_summary,
            "error_code": self.error_code,
        }
        if include_steps:
            data["steps"] = [s.public() for s in self.steps]
        return data
