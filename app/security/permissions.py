"""Permission model + the intent→(domain, effect) registry + the profile matrix (6B).

A permission is domain + action + resource scope + effect level + confirmation policy +
allowed profiles + temp-elevation-allowed + audit requirement. Effect levels are
ordered by severity; each profile has a ceiling. Owner-resource domains are hidden
from guest entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.security.profiles import (
    DEVELOPER, GUEST, LOCKED, STANDARD, TRUSTED,
)

# --- effect levels (ordered) -------------------------------------------------------
READ = "read"
LOCAL_WRITE = "local_write"
EXTERNAL_PREPARE = "external_prepare"
EXTERNAL_EFFECT = "external_effect"
DESTRUCTIVE = "destructive"
ADMINISTRATIVE = "administrative"
EFFECT_RANK = {READ: 0, LOCAL_WRITE: 1, EXTERNAL_PREPARE: 2, EXTERNAL_EFFECT: 3,
               DESTRUCTIVE: 4, ADMINISTRATIVE: 5}

# --- domains -----------------------------------------------------------------------
DOMAINS = ("runtime", "voice", "memory", "knowledge", "browser", "text", "whatsapp",
           "email", "tasks", "schedules", "activity", "settings")
# Domains that expose the owner's personal resources — never available to guest.
OWNER_RESOURCE_DOMAINS = frozenset({"memory", "knowledge", "tasks", "schedules",
                                    "whatsapp", "email"})

# --- profile ceilings (highest effect level each profile may perform) --------------
PROFILE_CEILING = {
    LOCKED: -1,                       # nothing beyond an explicit status allowlist
    GUEST: EFFECT_RANK[READ],         # read only, and never owner resources
    STANDARD: EFFECT_RANK[EXTERNAL_PREPARE],   # + simulated external effects
    TRUSTED: EFFECT_RANK[DESTRUCTIVE],
    DEVELOPER: EFFECT_RANK[ADMINISTRATIVE],
}
ELEVATABLE_TO = {STANDARD: TRUSTED}   # standard may request temporary trusted access

# Runtime read-only actions available even while locked.
STATUS_ACTIONS = frozenset({"status", "health", "overview"})


@dataclass
class Permission:
    domain: str
    action: str
    effect_level: str
    resource_scope: str = "owner"           # owner | collection | public
    confirmation_policy: str = "domain"     # domain | none
    allowed_profiles: tuple = ()
    temp_elevation_allowed: bool = True
    audit_required: bool = True


# intent -> (domain, effect_level, resource_scope). Confirmations complete an
# already-authorized action and are never re-gated here.
INTENT_POLICY: dict[str, tuple[str, str, str]] = {
    # memory
    "memory_search": ("memory", READ, "owner"), "memory_list": ("memory", READ, "owner"),
    "memory_propose": ("memory", LOCAL_WRITE, "owner"), "memory_update": ("memory", LOCAL_WRITE, "owner"),
    "memory_forget": ("memory", LOCAL_WRITE, "owner"), "memory_confirm": ("memory", READ, "owner"),
    "memory_cancel": ("memory", READ, "owner"),
    # knowledge
    "knowledge_search": ("knowledge", READ, "owner"), "knowledge_ask": ("knowledge", READ, "owner"),
    "knowledge_list": ("knowledge", READ, "owner"), "knowledge_import": ("knowledge", LOCAL_WRITE, "owner"),
    "knowledge_reindex": ("knowledge", LOCAL_WRITE, "owner"),
    "knowledge_archive": ("knowledge", LOCAL_WRITE, "owner"),
    "knowledge_delete": ("knowledge", DESTRUCTIVE, "owner"),
    # browser
    "browser_open": ("browser", READ, "public"), "browser_search": ("browser", READ, "public"),
    "browser_read": ("browser", READ, "public"), "browser_summarize": ("browser", READ, "public"),
    "browser_find": ("browser", READ, "public"), "browser_scroll": ("browser", READ, "public"),
    "browser_open_link": ("browser", READ, "public"), "browser_close": ("browser", READ, "public"),
    "browser_prepare_form": ("browser", EXTERNAL_EFFECT, "public"),
    # text
    "draft_text": ("text", LOCAL_WRITE, "owner"), "rewrite_text": ("text", LOCAL_WRITE, "owner"),
    "type_text": ("text", EXTERNAL_EFFECT, "owner"), "append_text": ("text", EXTERNAL_EFFECT, "owner"),
    "replace_selected_text": ("text", EXTERNAL_EFFECT, "owner"),
    # whatsapp
    "whatsapp_status": ("whatsapp", READ, "owner"), "whatsapp_find_contact": ("whatsapp", READ, "owner"),
    "whatsapp_open_chat": ("whatsapp", READ, "owner"), "whatsapp_close": ("whatsapp", READ, "owner"),
    "whatsapp_draft_message": ("whatsapp", EXTERNAL_PREPARE, "owner"),
    "whatsapp_prepare_send": ("whatsapp", EXTERNAL_EFFECT, "owner"),
    # email
    "email_status": ("email", READ, "owner"), "email_search": ("email", READ, "owner"),
    "email_open_thread": ("email", READ, "owner"), "email_summarize_thread": ("email", READ, "owner"),
    "email_close": ("email", READ, "owner"), "email_draft_new": ("email", EXTERNAL_PREPARE, "owner"),
    "email_draft_reply": ("email", EXTERNAL_PREPARE, "owner"),
    "email_prepare_send": ("email", EXTERNAL_EFFECT, "owner"),
    # tasks
    "task_plan": ("tasks", LOCAL_WRITE, "owner"), "task_approve": ("tasks", LOCAL_WRITE, "owner"),
    "task_resume": ("tasks", LOCAL_WRITE, "owner"), "task_cancel": ("tasks", LOCAL_WRITE, "owner"),
    # schedules
    "schedule_prepare": ("schedules", LOCAL_WRITE, "owner"), "schedule_list": ("schedules", READ, "owner"),
    "schedule_pause": ("schedules", LOCAL_WRITE, "owner"), "schedule_resume": ("schedules", LOCAL_WRITE, "owner"),
    "schedule_cancel": ("schedules", LOCAL_WRITE, "owner"),
}

# Effect levels whose REAL execution requires trusted (standard gets simulated only).
_REAL_EFFECT_FLAGS = {
    "text": "enable_real_text_input", "whatsapp": "enable_real_whatsapp_send",
    "email": "enable_real_email_send", "browser": "enable_browser_automation",
}


def real_effect_enabled(domain: str, settings) -> bool:
    flag = _REAL_EFFECT_FLAGS.get(domain)
    return bool(getattr(settings, flag, False)) if flag else False


def policy_for(intent: str):
    """(domain, effect_level, resource_scope) or None for un-gated intents."""
    return INTENT_POLICY.get(intent)
