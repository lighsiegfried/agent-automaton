"""Pydantic models shared across the API, router, safety layer and tools."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intent(str, Enum):
    OPEN_FOLDER = "open_folder"
    OPEN_APP = "open_app"
    TYPE_TEXT = "type_text"
    SEARCH_WEB = "search_web"
    SPEAK = "speak"
    UNKNOWN = "unknown"

    # Phase 4B.1 conversational service intents. These are dispatched to the
    # TextActionService / BrowserService through the deterministic dispatcher
    # (app/core/conversation.py) — never through the tool registry. TYPE_TEXT
    # above is reused for the text-insertion service route.
    DRAFT_TEXT = "draft_text"
    REWRITE_TEXT = "rewrite_text"
    APPEND_TEXT = "append_text"
    REPLACE_SELECTED_TEXT = "replace_selected_text"
    CANCEL_TEXT_ACTION = "cancel_text_action"
    BROWSER_OPEN = "browser_open"
    BROWSER_SEARCH = "browser_search"
    BROWSER_READ = "browser_read"
    BROWSER_SUMMARIZE = "browser_summarize"
    BROWSER_FIND = "browser_find"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_OPEN_LINK = "browser_open_link"
    BROWSER_PREPARE_FORM = "browser_prepare_form"
    BROWSER_CONFIRM_FORM = "browser_confirm_form"
    BROWSER_CANCEL_FORM = "browser_cancel_form"
    BROWSER_CLOSE = "browser_close"
    CONFIRM = "confirm"  # a bare confirmation phrase; routed by pending domain

    # Phase 4C — WhatsApp Web (dispatched to the WhatsApp integration).
    WHATSAPP_STATUS = "whatsapp_status"
    WHATSAPP_FIND_CONTACT = "whatsapp_find_contact"
    WHATSAPP_OPEN_CHAT = "whatsapp_open_chat"
    WHATSAPP_DRAFT_MESSAGE = "whatsapp_draft_message"
    WHATSAPP_PLACE_DRAFT = "whatsapp_place_draft"
    WHATSAPP_PREPARE_SEND = "whatsapp_prepare_send"
    WHATSAPP_CONFIRM_SEND = "whatsapp_confirm_send"
    WHATSAPP_CANCEL = "whatsapp_cancel"
    WHATSAPP_CLOSE = "whatsapp_close"

    # Phase 4D — web email (dispatched to the email integration).
    EMAIL_STATUS = "email_status"
    EMAIL_SEARCH = "email_search"
    EMAIL_OPEN_THREAD = "email_open_thread"
    EMAIL_SUMMARIZE_THREAD = "email_summarize_thread"
    EMAIL_DRAFT_NEW = "email_draft_new"
    EMAIL_DRAFT_REPLY = "email_draft_reply"
    EMAIL_PLACE_DRAFT = "email_place_draft"
    EMAIL_PREPARE_SEND = "email_prepare_send"
    EMAIL_CONFIRM_SEND = "email_confirm_send"
    EMAIL_CANCEL = "email_cancel"
    EMAIL_CLOSE = "email_close"

    # Phase 5A — explicit, local, auditable personal memory (dispatched to the
    # memory service). Storing/updating/forgetting are sensitive writes routed
    # through the global pending broker; a plain "sí" or a wake never confirms.
    MEMORY_PROPOSE = "memory_propose"
    MEMORY_CONFIRM = "memory_confirm"
    MEMORY_SEARCH = "memory_search"
    MEMORY_LIST = "memory_list"
    MEMORY_UPDATE = "memory_update"
    MEMORY_FORGET = "memory_forget"
    MEMORY_CANCEL = "memory_cancel"

    # Phase 5B — deterministic multi-step tasks (dispatched to the task service).
    # Planning/approval/resume/cancel are task-level; each sensitive STEP still
    # pauses for its own domain confirmation through the pending broker. Plan
    # approval never pre-confirms an effect.
    TASK_PLAN = "task_plan"
    TASK_APPROVE = "task_approve"
    TASK_RESUME = "task_resume"
    TASK_CANCEL = "task_cancel"

    # Phase 5C — persistent reminders + scheduled tasks (dispatched to the schedule
    # service). Creating/modifying a schedule is a sensitive write confirmed through
    # the pending broker; scheduled runs still pause for each effect's own domain
    # confirmation.
    SCHEDULE_PREPARE = "schedule_prepare"
    SCHEDULE_CONFIRM = "schedule_confirm"
    SCHEDULE_LIST = "schedule_list"
    SCHEDULE_PAUSE = "schedule_pause"
    SCHEDULE_RESUME = "schedule_resume"
    SCHEDULE_CANCEL = "schedule_cancel"

    # Phase 6A — local document Knowledge Vault (dispatched to the knowledge service).
    # Document content is untrusted; deleting a document is confirmed through the
    # pending broker. Answers are citation-grounded and never stored to memory
    # automatically.
    KNOWLEDGE_IMPORT = "knowledge_import"
    KNOWLEDGE_LIST = "knowledge_list"
    KNOWLEDGE_SEARCH = "knowledge_search"
    KNOWLEDGE_ASK = "knowledge_ask"
    KNOWLEDGE_REINDEX = "knowledge_reindex"
    KNOWLEDGE_ARCHIVE = "knowledge_archive"
    KNOWLEDGE_DELETE = "knowledge_delete"


class SafetyLevel(str, Enum):
    SAFE = "safe"  # can be executed/simulated freely
    SENSITIVE = "sensitive"  # requires explicit confirmation
    DESTRUCTIVE = "destructive"  # blocked by default


class ExecutionStatus(str, Enum):
    SIMULATED = "simulated"  # tool ran in simulation mode
    EXECUTED = "executed"  # tool performed the real action (real mode only)
    REJECTED = "rejected"  # tool refused its input (allowlist, bad path, ...)
    NEEDS_CONFIRMATION = "needs_confirmation"  # re-send with confirm=true
    BLOCKED = "blocked"  # refused by the safety layer
    NOT_HANDLED = "not_handled"  # no tool for this intent


class CommandRequest(BaseModel):
    text: str = Field(min_length=1, description="Natural-language command")
    confirm: bool = Field(
        default=False,
        description="Set true to approve a sensitive action that asked for confirmation",
    )
    language: str | None = Field(
        default=None,
        description="Command language hint ('en'/'es'), e.g. from voice detection. "
        "Auto-guessed when omitted.",
    )


class PlannerSource(str, Enum):
    RULE_ROUTER = "rule_router"  # ENABLE_LLM_PLANNER=false (default)
    LLM_PLANNER = "llm_planner"  # validated LLM plan was used
    FALLBACK_ROUTER = "fallback_router"  # LLM failed/was rejected; rules used


class PlanInfo(BaseModel):
    """Short, safe summary of an accepted LLM plan (no chain-of-thought)."""

    confidence: float
    reasoning_summary: str
    language: str


class CommandResponse(BaseModel):
    input_text: str
    intent: Intent
    tool: str | None = None
    safety_level: SafetyLevel | None = None
    status: ExecutionStatus
    message: str
    result: dict[str, Any] | None = None
    planner: PlannerSource = PlannerSource.RULE_ROUTER
    plan: PlanInfo | None = None
    # Short natural-language reply for the user (bilingual; template-based
    # unless ENABLE_RESPONSE_GENERATOR=true). Suitable for TTS.
    assistant_message: str = ""
