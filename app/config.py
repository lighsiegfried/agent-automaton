"""Application settings, loaded from environment variables / .env."""

import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agent-automaton"
    host: str = "127.0.0.1"
    port: int = 8000

    # Local persistent runtime (Phase 3B.7). scripts/local_runtime.py starts the
    # FastAPI server on the Windows host and keeps Dockerized Ollama (GPU-first)
    # running as a support service. Defaults are safe and local-only: the API
    # binds to loopback, GPU is required unless CPU is explicitly opted in
    # (ALLOW_CPU_OLLAMA), and stopping the runtime never touches Docker volumes
    # or the Ollama container unless asked. These settings change *where* things
    # run, never *what* is permitted — the safety layer is untouched.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    runtime_auto_start_ollama: bool = True
    runtime_require_gpu: bool = True
    runtime_keep_ollama_running: bool = True

    # Persistent warm runtime (Phase 3D.1). start/restart preload the LLM into
    # VRAM (keep_alive=-1) and the STT model onto the GPU so the first command
    # is never cold. The GPU check is strict: a CPU-offloaded model fails the
    # warmup unless explicitly permitted (ALLOW_CPU_OLLAMA or
    # OLLAMA_REQUIRE_FULL_GPU=false). Warmup never changes tool permissions.
    runtime_prewarm_llm: bool = True
    runtime_prewarm_stt: bool = True
    ollama_require_full_gpu: bool = True
    ollama_keep_model_loaded: bool = True
    model_keeper_check_seconds: int = 60

    # Identity (Phase 3B.5). The assistant persona is named Fifi; the project
    # and repository stay agent-automaton. The name is presentation-only and
    # never influences safety decisions. wake_word is future metadata for
    # Phase 3C — no wake word listening exists yet.
    agent_name: str = "Fifi"
    wake_word: str = "fifi"

    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:7b"

    # LLM command planner (Phase 2). When false, only the rule-based router
    # runs. When true, the planner proposes a plan that is strictly validated
    # and then goes through the exact same safety layer as rule-routed
    # commands; any planner failure falls back to the rule-based router.
    enable_llm_planner: bool = False
    llm_planner_model: str = "qwen2.5:7b"
    llm_planner_timeout_seconds: float = 20.0

    # Assistant response generator (Phase 3B). When false, short deterministic
    # template responses are used (bilingual). When true, Ollama phrases the
    # response — templates remain the fallback for any failure.
    enable_response_generator: bool = False
    response_model: str = "qwen2.5:7b"
    response_timeout_seconds: float = 15.0

    # Voice (Phase 3A). Disabled by default; voice endpoints answer with a
    # "disabled" status until ENABLE_VOICE=true. Voice commands feed into the
    # exact same /command flow — they never bypass the safety layer.
    enable_voice: bool = False
    stt_model: str = "small"
    # STT device / compute type (Phase 3C.1). Defaults are Blackwell-safe:
    # cuda + float16 works on RTX 50 (sm_120), where INT8 cuBLAS variants raise
    # CUBLAS_STATUS_NOT_SUPPORTED. "auto" is resolved at load time (see
    # app/voice/stt.py:resolve_stt_config) — on Blackwell it becomes float16 and
    # never an INT8 variant. CPU inference happens only when explicitly allowed.
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_allow_cpu_fallback: bool = False
    voice_language: str = "auto"  # "auto" lets whisper detect es/en
    # "windows" (SAPI via pyttsx3) | "voice_lab" (Phase 3D.0: neural voices via
    # the isolated local worker) | "simulated"
    tts_engine: str = "windows"

    # Voice Lab (Phase 3D.0). An embedded-but-isolated TTS laboratory under
    # voice_lab/ with its own venv/models/cache. The main API NEVER loads
    # neural TTS models; with TTS_ENGINE=voice_lab it only POSTs text to the
    # loopback worker, which reads the active profile from
    # config/voices/active.json. When the worker is unavailable (and fallback
    # is allowed), speech falls back to the existing Windows TTS — a TTS
    # failure never affects /health, command execution, or safety.
    voice_lab_url: str = "http://127.0.0.1:8766"
    voice_profile: str = "fifi_warm"
    voice_lab_timeout_seconds: float = 60.0
    voice_lab_allow_fallback: bool = True
    # Phase 3D.0.1: the runtime can optionally bring the Voice Lab up with
    # `start` and prewarm the active voice. Off by default — the Voice Lab
    # stays optional and Fifi degrades to Windows TTS without it.
    runtime_auto_start_voice_lab: bool = False
    runtime_prewarm_tts: bool = True
    voice_record_seconds: float = 5.0
    voice_sample_rate: int = 16000
    # Speak the assistant_message out loud after /voice/command (needs TTS).
    voice_speak_command_response: bool = False

    # Push-to-talk desktop client (Phase 3C). A host-only client
    # (scripts/fifi_ptt.py) that records while a global hotkey is held and sends
    # the clip to /voice/command. The API is unchanged; these settings only
    # configure the client and never affect the safety layer. No always-on mic
    # and no wake word — recording is an explicit, held-key action.
    # Wake-word runtime (Phase 3D.1). A host-only listener (scripts/fifi_wake.py)
    # that detects "Fifi" with a custom openWakeWord ONNX model gated by Silero
    # VAD, then captures one utterance and sends it through the exact same
    # /voice/command pipeline as push-to-talk. Disabled by default and never
    # auto-started; the custom model is NEVER downloaded or replaced — when it
    # is missing the wake service reports unavailable. Idle audio is never
    # retained and temporary WAVs are deleted unless WAKE_WORD_STORE_AUDIO=true.
    enable_wake_word: bool = False
    wake_word_model_path: str = "models/wake_words/fifi.onnx"
    wake_word_threshold: float = 0.55
    wake_word_cooldown_seconds: float = 2.0
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_silence_ms: int = 900
    vad_max_command_seconds: float = 20.0
    wake_word_store_audio: bool = False
    wake_word_input_device: str = "default"
    wake_word_audio_feedback: bool = True
    wake_word_mute_hotkey: str = "ctrl+alt+m"
    # Daily wake-mode TTS latency guard (Phase 3D.1). A hands-free spoken reply
    # must never keep the user waiting: if the active (possibly Qwen) voice can't
    # synthesize within this budget, the reply falls back to Kokoro — the command
    # itself already executed. VoiceDesign is NEVER loaded for a wake reply.
    wake_max_tts_wait_seconds: float = 20.0

    # Optional speaker verifier (Phase 3D.2). Prepared but DISABLED by default —
    # wake detection works fully without it. When enabled AND the verifier file
    # exists, the wake service adds an openWakeWord custom-verifier gate trained
    # from the authorized user's own samples (wake_training/). It only ever makes
    # detection stricter (a second gate); it never bypasses VAD, confirmation, or
    # tool permissions, and a missing verifier simply means "not used".
    enable_wake_word_verifier: bool = False
    wake_word_verifier_path: str = "models/wake_words/fifi_verifier.pkl"
    wake_word_verifier_threshold: float = 0.3

    enable_push_to_talk: bool = False
    ptt_hotkey: str = "ctrl+alt+space"
    ptt_max_seconds: float = 20.0
    ptt_min_seconds: float = 0.4
    ptt_confirm_window_seconds: float = 30.0
    ptt_audio_feedback: bool = True
    ptt_input_device: str = "default"

    # Controlled text insertion (Phase 4A). Drafting/rewriting text is always
    # safe; inserting it into a Windows app is SENSITIVE and simulated unless
    # ENABLE_REAL_TEXT_INPUT=true. Real insertion additionally requires an
    # explicit spoken/typed confirmation phrase, never types into an unverified
    # window, never presses Enter/submit, never types into password or secret
    # fields, and never uses the clipboard unless TEXT_ALLOW_CLIPBOARD_FALLBACK
    # is set (and then restores the previous clipboard). One pending action at a
    # time; it is memory-only and expires.
    enable_real_text_input: bool = False
    text_input_require_confirmation: bool = True
    text_action_expires_seconds: int = 45
    text_max_characters: int = 5000
    text_allowed_apps: str = "notepad,wordpad,winword,chrome,edge"
    text_allow_clipboard_fallback: bool = False
    text_block_password_fields: bool = True

    @property
    def text_allowed_apps_list(self) -> list[str]:
        return [a.strip().lower() for a in self.text_allowed_apps.split(",") if a.strip()]

    # Safe browser automation (Phase 4B). Reading/searching/scrolling/navigating
    # are safe; filling a form field is SENSITIVE and requires an explicit
    # confirmation phrase. Submitting/purchasing/publishing/uploading/deleting/
    # logging in are blocked in this phase. A dedicated, isolated Chromium
    # profile is used (never the user's real browser profile); cookies are never
    # imported and never exposed to the LLM.
    enable_browser_automation: bool = False
    browser_engine: str = "chromium"
    browser_headless: bool = False
    browser_profile_mode: str = "isolated"          # isolated | (future) authenticated
    browser_allow_existing_profile: bool = False
    browser_allowed_schemes: str = "https,http"
    browser_block_private_networks: bool = True
    browser_require_confirmation_for_form_fill: bool = True
    browser_allow_form_submit: bool = False
    browser_max_page_text_chars: int = 50000
    browser_action_timeout_seconds: int = 30
    browser_session_idle_seconds: int = 600

    @property
    def browser_allowed_schemes_list(self) -> list[str]:
        return [s.strip().lower() for s in self.browser_allowed_schemes.split(",") if s.strip()]

    # WhatsApp Web automation (Phase 4C). Reuses the controlled browser with a
    # DEDICATED persistent profile (never the user's normal browser); login is
    # manual by QR and cookies/tokens are never read or logged. Drafting is safe;
    # placing a draft and sending are two separate confirmations, and a send
    # requires naming the recipient. Simulated unless ENABLE_REAL_WHATSAPP_SEND.
    # No groups, attachments, bulk, or unofficial APIs.
    enable_whatsapp_automation: bool = False
    enable_real_whatsapp_send: bool = False
    whatsapp_profile_dir: str = "storage/browser/whatsapp"
    whatsapp_action_expires_seconds: int = 45
    whatsapp_max_message_chars: int = 4000
    whatsapp_send_cooldown_seconds: int = 5
    whatsapp_require_recipient_in_confirmation: bool = True

    # Web-email automation (Phase 4D). Reuses the controlled browser with
    # per-provider dedicated profiles (storage/browser/email/{gmail,outlook});
    # manual login only; cookies/tokens are never read or logged. Email content
    # is UNTRUSTED data — summarized, never executed; in-email instructions are
    # ignored. Drafting is safe; placing a draft and sending are two separate
    # confirmations and a send must name all recipients. Reply-all and BCC are
    # blocked this phase. Simulated unless ENABLE_REAL_EMAIL_SEND.
    enable_email_web_automation: bool = False
    enable_real_email_send: bool = False
    email_allowed_providers: str = "gmail,outlook"
    email_action_expires_seconds: int = 60
    email_max_body_chars: int = 10000
    email_max_recipients: int = 3
    email_allow_cc: bool = True
    email_allow_bcc: bool = False
    email_allow_attachments: bool = False
    email_send_cooldown_seconds: int = 10

    @property
    def email_allowed_providers_list(self) -> list[str]:
        return [p.strip().lower() for p in self.email_allowed_providers.split(",") if p.strip()]

    # Personal memory (Phase 5A). Explicit, local, auditable memory for Fifi.
    # Storage is a local SQLite file (WAL) with schema migrations — no cloud
    # dependency. Nothing is ever stored merely because it appeared in
    # conversation: creating/updating/forgetting a memory requires an explicit
    # user intent ("recuerda que…") that becomes a pending proposal, and then an
    # EXACT confirmation phrase — a plain "sí", a wake detection, or email/web
    # content never store memory. Sensitive data (passwords, OTPs, API keys,
    # cards, private keys, cookies, precise addresses, sensitive personal data)
    # is blocked by default and never inferred. Retrieval is deterministic and
    # returns only a small bounded bundle — never the whole database.
    enable_memory: bool = True
    memory_db_path: Path = PROJECT_ROOT / "storage" / "memory" / "fifi_memory.db"
    memory_owner: str = "local"                     # owner scope, single-user default
    memory_action_expires_seconds: int = 120
    memory_max_title_chars: int = 200
    memory_max_content_chars: int = 4000
    memory_default_confidence: float = 0.8
    # Bounded retrieval context injected into the planner (never the full DB).
    memory_context_max_items: int = 5
    memory_context_max_chars: int = 1500
    memory_search_max_results: int = 10
    memory_list_max_results: int = 50
    memory_block_sensitive: bool = True             # never weakened by a flag

    # Multi-step tasks (Phase 5B). Deterministic, resumable plans over the EXISTING
    # services with per-effect human confirmation — never unrestricted autonomy,
    # background agents, or arbitrary loops. An LLM may PROPOSE a plan, but
    # deterministic code validates every intent/argument against the service
    # registry (no arbitrary tools/Python/shell/selectors/unknown URLs), rejects
    # cycles and over-long plans, and classifies each step as read-only / local /
    # external effect. Plan approval ("aprobar plan") only lets read-only and
    # local-safe steps begin; it NEVER pre-confirms typing, form-fill, WhatsApp or
    # email sending — each effect still pauses in the one global broker and needs
    # its own exact domain phrase. One task executes effects at a time; nothing
    # auto-resumes after a restart. Stored in a separate local SQLite DB (WAL).
    enable_multi_step_tasks: bool = False
    tasks_db_path: Path = PROJECT_ROOT / "storage" / "tasks" / "fifi_tasks.db"
    task_owner: str = "local"
    task_max_steps: int = 12
    task_max_runtime_minutes: int = 20
    task_max_step_retries: int = 1                  # read-only steps only; effects never retry
    task_require_plan_approval: bool = True
    task_allow_background_execution: bool = False
    task_auto_resume: bool = False
    task_context_max_memories: int = 5
    task_context_max_chars: int = 2000

    # Persistent reminders + scheduled tasks (Phase 5C). Schedules fire reminders
    # and run validated Phase-5B task templates at a time/recurrence, in a separate
    # local SQLite DB (WAL). Instants are stored in UTC and resolved through the
    # configured IANA timezone. Creating/modifying a schedule is sensitive (a
    # pending broker action needing "confirmar programación" — a plain "sí"/wake
    # never confirms). One owned scheduler polls on a bounded interval and acquires
    # an atomic per-run lease so a restart or clock change can't double-run. At run
    # time the plan is REVALIDATED and memory context recomputed; read-only and
    # local-draft steps may auto-run, but every external effect (WhatsApp/email
    # send, text insertion, form fill) pauses in the global broker and needs a fresh
    # domain-specific confirmation — schedule approval never pre-confirms an effect,
    # and a paused task is never auto-resumed. Off by default.
    enable_schedules: bool = False
    schedules_db_path: Path = PROJECT_ROOT / "storage" / "schedules" / "fifi_schedules.db"
    schedule_owner: str = "local"
    schedule_timezone: str = "America/Guatemala"
    schedule_poll_seconds: int = 15
    schedule_misfire_grace_seconds: int = 300
    schedule_max_backlog_runs: int = 3
    schedule_max_active_runs: int = 1
    schedule_allow_read_only_autorun: bool = True
    schedule_allow_local_draft_autorun: bool = True
    schedule_allow_external_effect_autorun: bool = False   # never auto-sends

    # Local Activity Center (Phase 5D). A read-optimized, localhost-only observability
    # + safe-control UI that AGGREGATES safe events from every subsystem via additive
    # event hooks (subsystems never depend on the UI) into a separate local SQLite DB
    # (WAL). It stores only redacted, allowlisted metadata — never secrets, tokens,
    # cookies, form values, or full email/message/draft bodies. Every control it
    # offers (confirm/cancel/approve/resume/pause/restart/unload) routes through the
    # EXISTING services and the one global broker — it never mutates a subsystem DB
    # directly, never offers a generic "confirm everything", and never kills unrelated
    # processes. Event writes are best-effort and never block command execution.
    enable_activity_center: bool = True
    activity_db_path: Path = PROJECT_ROOT / "storage" / "activity" / "fifi_activity.db"
    activity_owner: str = "local"
    activity_retention_days: int = 30
    activity_max_events: int = 50000
    activity_redact_content: bool = True                   # never weakened for content
    activity_ui_host: str = "127.0.0.1"                    # loopback only
    activity_ui_port: int = 8770
    activity_enable_export: bool = True
    activity_page_size: int = 50                           # default timeline pagination

    # Local document Knowledge Vault (Phase 6A) — citation-grounded RAG over locally
    # ingested documents. Document content is UNTRUSTED data: it can inform answers
    # but never authorizes an action or modifies memory. Embeddings are generated by
    # a LOCAL model (deterministic hash fallback; optional sentence-transformers) —
    # documents are never sent to an external API. Retrieval is hybrid (FTS5 + local
    # vector similarity), bounded, and every document-derived answer cites
    # filename + page/section or states insufficient evidence. Deleting a document is
    # sensitive (a broker confirmation); OCR is disabled and never runs automatically.
    enable_knowledge_vault: bool = False
    knowledge_db_path: Path = PROJECT_ROOT / "storage" / "knowledge" / "knowledge.db"
    knowledge_files_dir: Path = PROJECT_ROOT / "storage" / "knowledge" / "files"
    knowledge_indexes_dir: Path = PROJECT_ROOT / "storage" / "knowledge" / "indexes"
    knowledge_owner: str = "local"
    knowledge_max_file_mb: int = 50
    knowledge_max_pages: int = 1000
    knowledge_chunk_size: int = 800
    knowledge_chunk_overlap: int = 120
    knowledge_max_results: int = 8
    knowledge_context_max_chars: int = 10000
    knowledge_embedding_device: str = "auto"               # auto | cpu | cuda
    knowledge_allow_ocr: bool = False                      # never runs automatically

    # Deterministic profiles, permissions and trust boundaries (Phase 6B). Profiles
    # (locked/guest/standard/trusted/developer) are explicit policy bundles — not AI
    # scores. A single central authorize() gates every service dispatch path with
    # default-deny; NO profile (including developer) may bypass the mandatory
    # invariants: recipient-specific confirmation, changed-target/hash revalidation,
    # secret blocking, the document prompt-injection boundary, no automatic
    # purchases/deletion/account changes, wake-cannot-confirm, and duplicate-effect
    # prevention. The authenticated Windows user is the identity boundary; optional
    # local unlock (Windows Hello or a SALTED password verifier — never plaintext, and
    # never in .env/logs). Temporary elevation is capability-scoped, visibly expiring,
    # audited, and cleared on reboot. Off by default (backward compatible); when
    # enabled it starts locked.
    enable_security_profiles: bool = False
    security_db_path: Path = PROJECT_ROOT / "storage" / "security" / "fifi_security.db"
    security_owner: str = "local"
    security_default_profile: str = "standard"
    security_lock_on_start: bool = True
    security_idle_lock_minutes: int = 15
    security_allow_windows_hello: bool = True
    security_allow_local_password: bool = True
    security_temp_elevation_minutes: int = 10
    security_max_failed_unlocks: int = 5
    security_lockout_minutes: int = 5

    # Native desktop app bridge (Phase 6C). The PySide6 desktop client is a THIN
    # presentation layer that talks ONLY to this localhost API — it holds no business
    # logic and writes no database. These endpoints mint a short-lived UI session token
    # and stream safe activity events (SSE). The token is in-memory (cleared on restart)
    # and gates the stream/overview endpoints. Bind host is forced to loopback. Off by
    # default (backward compatible); the desktop app enables it for its own process.
    enable_desktop_bridge: bool = False
    desktop_bind_host: str = "127.0.0.1"                   # loopback only — never 0.0.0.0
    desktop_session_ttl_minutes: int = 30                  # short-lived UI session token
    desktop_stream_heartbeat_seconds: int = 15
    desktop_stream_queue_max: int = 1000                   # bounded; drops if a client stalls

    # Safety
    require_confirmation: bool = True

    # Real execution (Phase 1.5). When false, every tool is simulated.
    # When true, only the safe real actions run: open_folder (validated),
    # open_app (allowlist), search_web (default browser).
    enable_real_windows_tools: bool = False

    # Comma-separated canonical app names that open_app may launch.
    allowed_apps: str = "notepad,calculator,chrome,edge,explorer"

    @property
    def allowed_apps_list(self) -> list[str]:
        return [a.strip().lower() for a in self.allowed_apps.split(",") if a.strip()]

    # Local storage
    storage_dir: Path = PROJECT_ROOT / "storage"
    db_path: Path = PROJECT_ROOT / "storage" / "assistant.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def real_windows_tools_enabled() -> bool:
    """Real execution requires both the opt-in flag and a Windows host.

    Inside Docker (Linux) this is always false — containers have no access to
    the user's desktop session (see docs/ARCHITECTURE.md).
    """
    return get_settings().enable_real_windows_tools and sys.platform == "win32"
