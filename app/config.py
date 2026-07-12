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

    enable_push_to_talk: bool = False
    ptt_hotkey: str = "ctrl+alt+space"
    ptt_max_seconds: float = 20.0
    ptt_min_seconds: float = 0.4
    ptt_confirm_window_seconds: float = 30.0
    ptt_audio_feedback: bool = True
    ptt_input_device: str = "default"

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
