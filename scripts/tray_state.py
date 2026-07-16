"""Pure state + menu model for the Fifi system-tray app (Phase 3E).

No pystray, Pillow, or Windows APIs here — just the state machine, the per-state
icon style, and which menu items are enabled in each state. Keeping this pure
makes the whole tray behaviour unit-testable without a display or tray backend.
"""

from __future__ import annotations

# --- states ------------------------------------------------------------------------
STOPPED = "stopped"
STARTING = "starting"
WARMING = "warming"
READY = "ready"
LISTENING = "listening"
MUTED = "muted"
PROCESSING = "processing"
DEGRADED = "degraded"
ERROR = "error"

STATES = [
    STOPPED, STARTING, WARMING, READY, LISTENING, MUTED, PROCESSING, DEGRADED, ERROR,
]

# Transient states the controller passes through while (re)starting.
TRANSIENT = {STARTING, WARMING, PROCESSING}
# States in which the host API + services are considered up.
RUNNING = {WARMING, READY, LISTENING, MUTED, PROCESSING, DEGRADED}


def is_running(state: str) -> bool:
    return state in RUNNING


def is_busy(state: str) -> bool:
    """True while a transition is in flight (menu actions should be inert)."""
    return state in TRANSIENT


# --- icon style (color + tooltip text) --------------------------------------------
# RGB tuples so the Pillow adapter can render a status dot without any assets.
ICON_STYLE = {
    STOPPED: {"color": (128, 128, 128), "text": "Fifi: stopped"},
    STARTING: {"color": (230, 190, 0), "text": "Fifi: starting…"},
    WARMING: {"color": (230, 190, 0), "text": "Fifi: warming…"},
    READY: {"color": (40, 170, 70), "text": "Fifi: ready"},
    LISTENING: {"color": (40, 200, 90), "text": "Fifi: listening"},
    MUTED: {"color": (60, 120, 210), "text": "Fifi: muted"},
    PROCESSING: {"color": (0, 180, 200), "text": "Fifi: processing…"},
    DEGRADED: {"color": (230, 130, 20), "text": "Fifi: degraded"},
    ERROR: {"color": (210, 50, 50), "text": "Fifi: error"},
}


def icon_style(state: str) -> dict:
    return ICON_STYLE.get(state, ICON_STYLE[STOPPED])


# --- menu model --------------------------------------------------------------------
# Item keys (stable identifiers the controller dispatches on) + display labels.
START = "start"
STOP = "stop"
RESTART = "restart"
WAKE_MODE = "wake_mode"
PTT_MODE = "ptt_mode"
MUTE = "mute"
UNMUTE = "unmute"
VOICE_PROFILE = "voice_profile"
RUNTIME_MODE = "runtime_mode"
OPEN_DESIGNER = "open_designer"
OPEN_LOGS = "open_logs"
MODEL_STATUS = "model_status"
RELEASE_VRAM = "release_vram"
# Personal memory (Phase 5A). Read-only status/search/viewer plus explicit
# confirm/cancel of a pending memory — a menu click IS an explicit confirmation
# (wake never reaches these).
MEMORY_STATUS = "memory_status"
MEMORY_PENDING = "memory_pending"
MEMORY_CONFIRM = "memory_confirm"
MEMORY_CANCEL = "memory_cancel"
MEMORY_SEARCH = "memory_search"
MEMORY_VIEWER = "memory_viewer"
# Multi-step tasks (Phase 5B). Inspect/approve/resume/cancel the current task and
# view its plan, pending confirmation, and audit — all API-backed and read-only
# except the deliberate approve/resume/cancel clicks.
TASK_CURRENT = "task_current"
TASK_PLAN_VIEW = "task_plan_view"
TASK_APPROVE = "task_approve"
TASK_RESUME = "task_resume"
TASK_CANCEL = "task_cancel"
TASK_PENDING = "task_pending"
TASK_AUDIT = "task_audit"
# Schedules + reminders (Phase 5C). Inspect upcoming schedules, pause/resume, run
# now, view the pending scheduled effect and recent results — API-backed.
SCHEDULE_UPCOMING = "schedule_upcoming"
SCHEDULE_CREATE = "schedule_create"
SCHEDULE_PAUSE = "schedule_pause"
SCHEDULE_RESUME = "schedule_resume"
SCHEDULE_RUN_NOW = "schedule_run_now"
SCHEDULE_PENDING = "schedule_pending"
SCHEDULE_RESULTS = "schedule_results"
# Activity Center (Phase 5D). Open the localhost UI and show a summary consistent
# with it (pending count, active task progress, next reminder, recent errors).
ACTIVITY_OPEN = "activity_open"
ACTIVITY_STATUS = "activity_status"
# Knowledge Vault (Phase 6A). Open the vault, import/search documents, and view the
# index status — API-backed.
KNOWLEDGE_OPEN = "knowledge_open"
KNOWLEDGE_IMPORT = "knowledge_import"
KNOWLEDGE_SEARCH = "knowledge_search"
KNOWLEDGE_STATUS = "knowledge_status"
# Security profiles (Phase 6B). Show the current profile, lock/unlock (a menu click is
# an explicit local action — wake never reaches these), request a capability-scoped
# temporary elevation, and revoke it. Profile CHANGES stay behind the spoken/localhost
# "confirmar cambio de perfil" flow — the tray never changes the base profile.
SECURITY_STATUS = "security_status"
SECURITY_UNLOCK = "security_unlock"
SECURITY_LOCK = "security_lock"
SECURITY_ELEVATE = "security_elevate"
SECURITY_REVOKE = "security_revoke"
SETTINGS = "settings"
EXIT = "exit"

MENU_ITEMS = [
    (START, "Start Fifi"),
    (STOP, "Stop Fifi"),
    (RESTART, "Restart Fifi"),
    (WAKE_MODE, "Wake mode"),
    (PTT_MODE, "Push-to-talk mode"),
    (MUTE, "Mute microphone"),
    (UNMUTE, "Unmute microphone"),
    (VOICE_PROFILE, "Voice profile"),
    (RUNTIME_MODE, "Runtime mode"),
    (OPEN_DESIGNER, "Open Voice Designer"),
    (OPEN_LOGS, "Open logs"),
    (MODEL_STATUS, "Model status"),
    (RELEASE_VRAM, "Release VRAM"),
    (MEMORY_STATUS, "Memory status"),
    (MEMORY_PENDING, "View pending memory"),
    (MEMORY_CONFIRM, "Confirm pending memory"),
    (MEMORY_CANCEL, "Cancel pending memory"),
    (MEMORY_SEARCH, "Search memories"),
    (MEMORY_VIEWER, "Open memory viewer"),
    (TASK_CURRENT, "Current task"),
    (TASK_PLAN_VIEW, "View plan"),
    (TASK_APPROVE, "Approve plan"),
    (TASK_RESUME, "Resume task"),
    (TASK_CANCEL, "Cancel task"),
    (TASK_PENDING, "Pending confirmation"),
    (TASK_AUDIT, "Audit summary"),
    (SCHEDULE_UPCOMING, "Upcoming schedules"),
    (SCHEDULE_CREATE, "Create reminder"),
    (SCHEDULE_PAUSE, "Pause schedule"),
    (SCHEDULE_RESUME, "Resume schedule"),
    (SCHEDULE_RUN_NOW, "Run schedule now"),
    (SCHEDULE_PENDING, "Pending scheduled effect"),
    (SCHEDULE_RESULTS, "Recent results"),
    (ACTIVITY_OPEN, "Open Activity Center"),
    (ACTIVITY_STATUS, "Activity status"),
    (KNOWLEDGE_OPEN, "Open Knowledge Vault"),
    (KNOWLEDGE_IMPORT, "Import document"),
    (KNOWLEDGE_SEARCH, "Search documents"),
    (KNOWLEDGE_STATUS, "Index status"),
    (SECURITY_STATUS, "Security status"),
    (SECURITY_UNLOCK, "Unlock (Windows Hello)"),
    (SECURITY_LOCK, "Lock now"),
    (SECURITY_ELEVATE, "Request elevation"),
    (SECURITY_REVOKE, "Revoke elevation"),
    (SETTINGS, "Settings"),
    (EXIT, "Exit"),
]

RUNTIME_MODES = ["daily", "low-memory", "designer"]
# Capabilities a temporary elevation can be scoped to (never a global "*"). These are
# the domains whose REAL external effects require a trusted profile.
ELEVATION_CAPABILITIES = ["email", "whatsapp", "text", "browser", "knowledge"]


def menu_enabled(state: str) -> dict:
    """Which menu items are actionable in ``state`` (True = enabled).

    Actions that touch running services are disabled when stopped or mid-transition;
    Mute/Unmute mirror the mic state; Start only offers itself when down.
    """
    running = is_running(state)
    busy = is_busy(state)
    return {
        START: state in (STOPPED, ERROR) and not busy,
        STOP: state != STOPPED and not busy,
        RESTART: state != STOPPED and not busy,
        WAKE_MODE: running and not busy,
        PTT_MODE: running and not busy,
        MUTE: state in (LISTENING, READY) and not busy,
        UNMUTE: state == MUTED,
        VOICE_PROFILE: running and not busy,
        RUNTIME_MODE: running and not busy,
        OPEN_DESIGNER: not busy,
        OPEN_LOGS: True,          # logs are always viewable
        MODEL_STATUS: True,       # read-only, always available
        RELEASE_VRAM: running and not busy,
        # Memory is served by the local API (read-only status/search/viewer, plus
        # explicit confirm/cancel that no-op when nothing is pending) — always
        # available so the user can inspect or confirm regardless of mic state.
        MEMORY_STATUS: True,
        MEMORY_PENDING: True,
        MEMORY_CONFIRM: True,
        MEMORY_CANCEL: True,
        MEMORY_SEARCH: True,
        MEMORY_VIEWER: True,
        # Task controls are API-backed (read-only inspect + deliberate
        # approve/resume/cancel) and available regardless of mic state.
        TASK_CURRENT: True,
        TASK_PLAN_VIEW: True,
        TASK_APPROVE: True,
        TASK_RESUME: True,
        TASK_CANCEL: True,
        TASK_PENDING: True,
        TASK_AUDIT: True,
        # Schedule controls are API-backed and available regardless of mic state.
        SCHEDULE_UPCOMING: True,
        SCHEDULE_CREATE: True,
        SCHEDULE_PAUSE: True,
        SCHEDULE_RESUME: True,
        SCHEDULE_RUN_NOW: True,
        SCHEDULE_PENDING: True,
        SCHEDULE_RESULTS: True,
        # The Activity Center is a localhost read-only UI — always openable.
        ACTIVITY_OPEN: True,
        ACTIVITY_STATUS: True,
        # Knowledge Vault controls are API-backed and always available.
        KNOWLEDGE_OPEN: True,
        KNOWLEDGE_IMPORT: True,
        KNOWLEDGE_SEARCH: True,
        KNOWLEDGE_STATUS: True,
        # Security controls are API-backed; the API enforces the real decision (a
        # lock/unlock is idempotent and safe regardless of the runtime state).
        SECURITY_STATUS: True,
        SECURITY_UNLOCK: True,
        SECURITY_LOCK: True,
        SECURITY_ELEVATE: True,
        SECURITY_REVOKE: True,
        SETTINGS: True,
        EXIT: True,               # exit must never be locked out
    }
