"""Voice Lab settings — loaded from voice_lab/.env, NEVER the main project .env.

Isolation rules (Phase 3D.0):
- All caches, models, previews, and logs live under voice_lab/ — the main
  project's storage/ and the Ollama Docker service are never touched.
- HF_HOME / TORCH_HOME / XDG_CACHE_HOME are pinned into voice_lab/ so neural
  engines can never write into the user profile or the main project.
- The worker binds to loopback only.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Runtime modes (Phase 3D.0.5). The mode decides which engines the coordinator
# is willing to keep warm at once:
#   daily      — Ollama + Whisper + ONLY the active voice profile's TTS engine.
#   designer   — daily, but VoiceDesign is prioritized while creating identities
#                (and may temporarily free Ollama when explicitly configured).
#   low-memory — Kokoro / Windows fallback only; heavy Qwen models are refused.
VOICE_LAB_MODES = ("daily", "designer", "low-memory")
DEFAULT_MODE = "daily"


def normalize_mode(raw: str | None) -> str:
    """Fold any spelling (low_memory / 'low memory' / LOW-MEMORY) to a canonical
    mode; anything unknown falls back to the safe default rather than raising."""
    value = (raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    if value in ("low-memory", "lowmemory", "low"):
        return "low-memory"
    return value if value in VOICE_LAB_MODES else DEFAULT_MODE

# THE canonical repository-root resolver (Phase 3D.0.3a). Every shared path —
# config/voices, voice_lab storage/models/logs, previews, identity references —
# derives from these two constants, which resolve from THIS FILE's location.
# The process CWD is never consulted, so the worker, the CLI, and the tests
# see the same profile store no matter where they were launched from.
VOICE_LAB_ROOT = Path(__file__).resolve().parent.parent  # voice_lab/
PROJECT_ROOT = VOICE_LAB_ROOT.parent  # agent-automaton/


def repo_root() -> Path:
    """The agent-automaton checkout this worker belongs to (CWD-independent)."""
    return PROJECT_ROOT

MODELS_DIR = VOICE_LAB_ROOT / "models"
STORAGE_DIR = VOICE_LAB_ROOT / "storage"
PREVIEWS_DIR = STORAGE_DIR / "previews"
CACHE_DIR = STORAGE_DIR / "cache"
LOGS_DIR = STORAGE_DIR / "logs"
# Lightweight per-job metadata (atomic JSON, no tensors/model objects) so a
# crashed/restarted worker can mark in-flight jobs as interrupted instead of
# pretending to resume an in-memory GPU operation.
JOBS_META_DIR = STORAGE_DIR / "jobs"
# Coordinator runtime state (Phase 3D.0.5): the live mode override, persisted so
# it survives a worker restart. Small JSON only — never models or tensors.
RUNTIME_STATE_DIR = STORAGE_DIR / "runtime"

# Shared integration files — the ONLY place voice_lab and the main project meet.
VOICES_CONFIG_DIR = PROJECT_ROOT / "config" / "voices"
PROFILES_DIR = VOICES_CONFIG_DIR / "profiles"
ACTIVE_PROFILE_PATH = VOICES_CONFIG_DIR / "active.json"


class VoiceLabSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=VOICE_LAB_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="VOICE_LAB_",
        extra="ignore",
    )

    # Loopback only — the worker must never be reachable from the network.
    host: str = "127.0.0.1"
    port: int = 8766

    default_profile: str = "fifi_warm"
    # The exact comparison sentence (Phase 3D.0.1) — every preview uses it so
    # profiles are compared apples-to-apples.
    preview_text: str = (
        "Hola, soy Fifi. El sistema está listo y puedo ayudarte con tus tareas. "
        "¿En qué trabajaremos hoy?"
    )

    # Neural engine defaults; individual profiles can override.
    device: str = "cuda"
    dtype: str = "float16"

    # Play synthesized audio on the worker host (the CLI relies on this).
    playback_enabled: bool = True

    # Resource policy (Phase 3D.0.4) — 16 GB system RAM and 16 GB dedicated
    # VRAM on the target machine. All comparisons happen in BYTES inside
    # app/resources.py; these thresholds are the display/config surface.
    # The 3D.0.3a values (3 GB RAM / 3 GB VRAM) caused FALSE rejections on
    # this machine, where free RAM normally hovers near 2-5 GB.
    #
    # VOICE_LAB_MIN_FREE_RAM_GB is accepted as an alias for the system-RAM
    # threshold (Phase 3D.0.5 config surface); the older
    # VOICE_LAB_MIN_FREE_SYSTEM_RAM_GB still works.
    min_free_system_ram_gb: float = Field(
        default=2.0,
        validation_alias=AliasChoices(
            "VOICE_LAB_MIN_FREE_SYSTEM_RAM_GB", "VOICE_LAB_MIN_FREE_RAM_GB"
        ),
    )
    min_free_vram_gb: float = 2.0
    min_free_disk_gb: float = 5.0
    heavy_job_concurrency: int = 1
    # A running job whose heartbeat is older than this (while /health is still
    # alive) is reported as stalled.
    job_stall_seconds: float = 30.0

    # Exclusive design mode: before a heavy VoiceDesign job, unload unused
    # Voice Lab models (never Ollama, never the main API) to maximize headroom.
    exclusive_design_mode: bool = True

    # --- RAM/VRAM-aware orchestration (Phase 3D.0.5) --------------------------------
    # Runtime mode; the coordinator may override it live via POST /mode, but this
    # is the value the worker boots with (and returns to on restart when no
    # persisted override exists).
    mode: str = DEFAULT_MODE
    # Idle heavy models are released after this many seconds with no synthesis
    # AND no active job (0 disables idle unloading). Cached weights are never
    # deleted — only the in-memory model/VRAM is freed, so the next use reloads
    # from the local cache with zero downloads.
    idle_unload_seconds: float = 300.0
    # How many HEAVY (Qwen) models may be resident at once. Loading one more than
    # this makes the coordinator unload the least-recently-used unused heavy
    # model first (release refs -> gc -> clear CUDA cache).
    max_loaded_heavy_models: int = 1
    # In designer mode ONLY, may a heavy VoiceDesign load temporarily free
    # Ollama's model without per-job UI consent? Default false: Ollama is left
    # alone and the job returns a clear, actionable message instead.
    designer_auto_unload_ollama: bool = False
    # Include Kokoro in the response fallback chain. When false, responses skip
    # Kokoro and fall back straight to Windows host TTS.
    allow_kokoro_fallback: bool = True

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: str) -> str:
        return normalize_mode(value)

    # Soft timeouts (Phase 3D.0.4): a heavy job stuck past these is marked
    # failed (timeout) — the UI never waits forever. The underlying torch call
    # cannot be force-killed; the job record is truthful about that.
    model_load_timeout_seconds: float = 300.0
    generation_timeout_seconds: float = 180.0


def ensure_isolated_dirs() -> None:
    for path in (MODELS_DIR, PREVIEWS_DIR, CACHE_DIR, LOGS_DIR, JOBS_META_DIR,
                 RUNTIME_STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def pin_cache_env() -> None:
    """Force every model/cache download into voice_lab/ (isolation guarantee).

    setdefault, not overwrite: an explicit operator override still wins, but the
    default can never leak into the user profile or the main project.
    """
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "hf" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))


@lru_cache
def get_settings() -> VoiceLabSettings:
    return VoiceLabSettings()
