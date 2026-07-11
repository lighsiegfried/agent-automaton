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
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    voice_language: str = "auto"  # "auto" lets whisper detect es/en
    tts_engine: str = "windows"  # "windows" (SAPI via pyttsx3) | "simulated"
    voice_record_seconds: float = 5.0
    voice_sample_rate: int = 16000
    # Speak the assistant_message out loud after /voice/command (needs TTS).
    voice_speak_command_response: bool = False

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
