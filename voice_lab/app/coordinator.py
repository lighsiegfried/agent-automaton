"""Model coordinator (Phase 3D.0.5) — one brain for RAM/VRAM-aware loading.

The Voice Lab shares a single 16 GB GPU and ~16 GB of system RAM with Ollama
(the planner LLM) and faster-whisper (STT). Warming several neural TTS engines
at once starves both and makes synthesis crawl. This coordinator enforces a
simple daily-use policy so that only what the ACTIVE voice needs is ever
resident:

- Kokoro profile           -> load Kokoro only.
- Frozen Qwen clone        -> load Qwen Base (clone) only.
- VoiceDesign              -> loaded ONLY while creating identities, never for
                              normal Fifi responses; released after idle.

It is the single place that knows:
- the runtime MODE (daily / designer / low-memory) and what each mode permits;
- which engine/model is loaded and when each was last used (for LRU eviction);
- how many HEAVY (Qwen) models may be resident at once;
- how to make room BEFORE a heavy load — release references, garbage-collect,
  clear the CUDA cache — WITHOUT ever deleting cached weights;
- when an idle heavy model should be unloaded to give RAM/VRAM back.

It never loads models itself (the engines do that lazily on first synthesis);
it decides what may load and unloads what should not stay. Every probe is
delegated to app.resources / app.gpu (byte-exact, best-effort, never raises),
so the coordinator adds policy, not new measurement.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import (
    RUNTIME_STATE_DIR,
    get_settings,
    normalize_mode,
)
from app.engines.qwen3_tts import (
    CLONE_MODEL_ID,
    CUSTOM_VOICE_MODEL_ID,
    DEFAULT_MODEL_ID,
    is_clone_base,
    is_voice_design,
)

HEAVY_ENGINE = "qwen3_tts"
KOKORO_ENGINE = "kokoro"
WINDOWS_ENGINE = "windows_sapi"


def model_kind(model_id: str) -> str:
    """The VRAM-estimate bucket for a Qwen model id: voice_design | clone | custom_voice."""
    if is_voice_design(model_id):
        return "voice_design"
    if is_clone_base(model_id):
        return "clone"
    return "custom_voice"


def kind_target_model(kind: str) -> str:
    """The canonical Qwen model id a heavy job of this kind loads."""
    if kind == "clone":
        return CLONE_MODEL_ID
    if kind == "custom_voice":
        return CUSTOM_VOICE_MODEL_ID
    return DEFAULT_MODEL_ID  # voice_design


class ModelCoordinator:
    """Shared, thread-safe model/resource policy for all Voice Lab jobs."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._state_dir = Path(state_dir) if state_dir else RUNTIME_STATE_DIR
        # last_activity drives idle unloading; per-model timestamps drive LRU
        # eviction when more heavy models are resident than the limit allows.
        self._last_activity = time.monotonic()
        self._model_last_used: dict[str, float] = {}
        self._mode = self._load_persisted_mode()
        # The idle unloader queries these; bind_jobs wires them to the real
        # JobManager so a heavy job in flight is never unloaded out from under.
        self._active_jobs_fn: Callable[[], int] = lambda: 0
        self._active_heavy_fn: Callable[[], dict[str, Any] | None] = lambda: None
        self._idle_thread: threading.Thread | None = None
        self._idle_stop = threading.Event()

    # -- mode ---------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, raw: str) -> str:
        """Switch runtime mode (daily/designer/low-memory) and persist it.

        Switching does NOT itself load or unload anything — the caller decides
        whether to also run enforce_policy(); a mode switch alone only changes
        what future loads are permitted."""
        mode = normalize_mode(raw)
        with self._lock:
            self._mode = mode
            self._persist_mode(mode)
        return mode

    def _state_file(self) -> Path:
        return self._state_dir / "coordinator.json"

    def _load_persisted_mode(self) -> str:
        try:
            data = json.loads(self._state_file().read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("mode"):
                return normalize_mode(data["mode"])
        except (OSError, ValueError):
            pass
        return normalize_mode(get_settings().mode)

    def _persist_mode(self, mode: str) -> None:
        """Best-effort: a persisted mode survives a worker/container restart."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._state_file().write_text(
                json.dumps({
                    "mode": mode,
                    "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }),
                encoding="utf-8",
            )
        except OSError:
            pass  # persistence is a convenience, never a hard requirement

    # -- activity / last-used -----------------------------------------------------------

    def note_activity(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def note_engine_use(self, engine: str, model: str = "") -> None:
        """Record a synthesis. Resets the idle clock and, for heavy engines,
        stamps the model's last-used time so LRU eviction is meaningful."""
        with self._lock:
            self._last_activity = time.monotonic()
            if engine == HEAVY_ENGINE and model:
                self._model_last_used[model] = self._last_activity

    def seconds_idle(self, now: float | None = None) -> float:
        with self._lock:
            return max(0.0, (now if now is not None else time.monotonic()) - self._last_activity)

    # -- policy: which engine an active profile needs -----------------------------------

    def effective_fallback_chain(self, profile: Any) -> list[str]:
        """Preferred synthesis order for THIS profile under the current mode.

        Base order is selected engine -> Kokoro -> profile fallback -> Windows.
        - low-memory mode drops the heavy Qwen engine entirely, so normal
          responses transparently use Kokoro (or Windows) — never a refusal.
        - allow_kokoro_fallback=false removes Kokoro, falling straight back to
          the Windows host TTS.
        Duplicates collapse while order is preserved; Windows host TTS is always
        the final safety net.
        """
        settings = get_settings()
        candidates = [
            profile.engine,
            KOKORO_ENGINE if settings.allow_kokoro_fallback else "",
            profile.fallback_engine,
            WINDOWS_ENGINE,
        ]
        chain: list[str] = []
        for name in candidates:
            if not name:
                continue
            if self.mode == "low-memory" and name == HEAVY_ENGINE:
                continue  # heavy models never load in low-memory mode
            if name not in chain:
                chain.append(name)
        return chain or [WINDOWS_ENGINE]

    def required_engine(self, profile: Any) -> str:
        """The single engine that should be warm for this profile right now —
        the first entry of the effective fallback chain."""
        return self.effective_fallback_chain(profile)[0]

    def profile_is_heavy(self, profile: Any) -> bool:
        """Will serving this profile actually load a heavy Qwen model? (False in
        low-memory mode, where the effective engine is Kokoro/Windows.)"""
        return self.required_engine(profile) == HEAVY_ENGINE

    # -- policy: mode gates for heavy loads ---------------------------------------------

    def mode_allows_heavy(self, kind: str) -> tuple[bool, str]:
        """May a heavy (Qwen) model load right now? low-memory mode says no."""
        if self.mode == "low-memory":
            return False, (
                "El modo de baja memoria está activo (VOICE_LAB_MODE=low-memory): "
                "no se cargan modelos Qwen. Cambia a modo daily o designer "
                "para crear o usar voces neuronales pesadas."
            )
        return True, ""

    def designer_ollama_consent(self, kind: str) -> bool:
        """In designer mode only, may a heavy load free Ollama automatically?

        Governed by VOICE_LAB_DESIGNER_AUTO_UNLOAD_OLLAMA (default false). This
        is the mode-driven consent; a per-job UI consent still works everywhere.
        """
        return (
            self.mode == "designer"
            and kind == "voice_design"
            and get_settings().designer_auto_unload_ollama
        )

    # -- resident heavy models ----------------------------------------------------------

    @staticmethod
    def _qwen_instance():
        """The live Qwen engine instance if one was ever built, else None.
        Read-only peek — never constructs the engine (which would import torch)."""
        from app.engines.base import _instances

        return _instances.get(HEAVY_ENGINE)

    def resident_heavy_models(self) -> list[str]:
        return list(getattr(self._qwen_instance(), "_models", {}) or {})

    def _lru_order(self, models: list[str]) -> list[str]:
        """Oldest-first: never-used models (no timestamp) sort as oldest."""
        with self._lock:
            return sorted(models, key=lambda m: self._model_last_used.get(m, 0.0))

    def unused_heavy_models(self, target: str) -> list[str]:
        """Resident heavy models other than the target, least-recently-used first."""
        return self._lru_order([m for m in self.resident_heavy_models() if m != target])

    def estimated_vram_gb(self, model_id: str) -> float:
        from app.gpu import VRAM_ESTIMATES_GB

        return VRAM_ESTIMATES_GB.get(model_kind(model_id), 0.0)

    # -- making room --------------------------------------------------------------------

    def prepare_for_heavy_load(self, target: str) -> dict[str, Any]:
        """Enforce max_loaded_heavy_models BEFORE loading `target`.

        Keeps at most (max_loaded_heavy_models - 1) OTHER heavy models resident,
        evicting the least-recently-used first so the new load fits. Each
        eviction releases references, garbage-collects, and clears the CUDA
        cache (Qwen3TtsEngine.unload_model) — cached weight FILES are never
        touched, so a re-load needs no download.
        """
        qwen = self._qwen_instance()
        if qwen is None:
            return {"unloaded": [], "kept": []}
        limit = max(0, get_settings().max_loaded_heavy_models - 1)
        others = self.unused_heavy_models(target)
        to_evict = others[: max(0, len(others) - limit)] if limit else others
        unloaded: list[str] = []
        for model_id in to_evict:
            if qwen.unload_model(model_id):  # ref release -> gc -> empty_cache
                unloaded.append(model_id)
            with self._lock:
                self._model_last_used.pop(model_id, None)
        kept = [m for m in self.resident_heavy_models() if m != target]
        return {"unloaded": unloaded, "kept": kept}

    def enforce_policy(self, profile: Any | None, reason: str = "optimize") -> dict[str, Any]:
        """Immediately make the loaded set match the daily-use policy.

        Unloads every engine/model NOT required by the active profile: all heavy
        Qwen models except the one the active profile needs, and Kokoro when it
        is not the active engine. Windows SAPI holds no GPU state. Cached weights
        are never deleted. This is what `model-optimize` and (a stronger form of)
        idle unloading run.
        """
        from app.engines.base import loaded_engines, unload_all

        required = self.required_engine(profile) if profile is not None else None
        keep_model = ""
        if required == HEAVY_ENGINE and profile is not None:
            keep_model = profile.model or DEFAULT_MODEL_ID

        unloaded_models: list[str] = []
        qwen = self._qwen_instance()
        if qwen is not None:
            for model_id in list(getattr(qwen, "_models", {})):
                if model_id == keep_model:
                    continue
                if qwen.unload_model(model_id):
                    unloaded_models.append(model_id)
                with self._lock:
                    self._model_last_used.pop(model_id, None)

        unloaded_engines: list[str] = []
        loaded = loaded_engines()
        if required != KOKORO_ENGINE and loaded.get(KOKORO_ENGINE):
            unloaded_engines += unload_all(KOKORO_ENGINE)
        return {
            "reason": reason,
            "kept_engine": required,
            "kept_model": keep_model,
            "unloaded_engines": unloaded_engines,
            "unloaded_models": unloaded_models,
        }

    # -- idle unloading -----------------------------------------------------------------

    def idle_unload_tick(
        self,
        now: float | None = None,
        active_jobs: int | None = None,
        idle_seconds: float | None = None,
    ) -> list[str]:
        """One idle check. Unloads ALL resident heavy models once the worker has
        been idle past the threshold with no active job. Returns what it freed.

        Kokoro is intentionally kept — it is cheap and usually the daily voice /
        fallback. Heavy Qwen models (VoiceDesign ~5 GB, clone ~2 GB) are the
        memory hogs, so those are what idle time reclaims. A re-load reads the
        local cache; no download ever happens.
        """
        threshold = get_settings().idle_unload_seconds if idle_seconds is None else idle_seconds
        if threshold <= 0:
            return []
        active = self._active_jobs_fn() if active_jobs is None else active_jobs
        if active > 0:
            self.note_activity()  # work in flight is not idleness
            return []
        if self.seconds_idle(now) < threshold:
            return []
        qwen = self._qwen_instance()
        if qwen is None or not getattr(qwen, "_models", None):
            return []
        unloaded: list[str] = []
        for model_id in self._lru_order(self.resident_heavy_models()):
            if qwen.unload_model(model_id):
                unloaded.append(model_id)
            with self._lock:
                self._model_last_used.pop(model_id, None)
        # Reset the clock so a still-idle worker does not re-probe every tick.
        self.note_activity()
        return unloaded

    def bind_jobs(self, jobs: Any) -> None:
        """Wire the idle unloader + status to the live JobManager."""
        self._active_jobs_fn = lambda: len(jobs.active())
        self._active_heavy_fn = jobs.active_heavy

    def start_idle_thread(self) -> None:
        """Start the background idle unloader (once). No-op when disabled."""
        with self._lock:
            if self._idle_thread is not None or get_settings().idle_unload_seconds <= 0:
                return
            self._idle_stop.clear()

            def _loop() -> None:
                # React within a fraction of the timeout without busy-looping.
                interval = max(5.0, min(60.0, get_settings().idle_unload_seconds / 6))
                while not self._idle_stop.wait(interval):
                    try:
                        self.idle_unload_tick()
                    except Exception:
                        pass  # an idle probe must never take the worker down

            self._idle_thread = threading.Thread(
                target=_loop, daemon=True, name="voicelab-idle-unload"
            )
            self._idle_thread.start()

    def stop_idle_thread(self) -> None:
        self._idle_stop.set()

    # -- status (Phase 3D.0.5, requirement 7) -------------------------------------------

    def status(self, profile: Any | None = None, jobs: Any = None) -> dict[str, Any]:
        """The full orchestration picture for `model-status` / the UI.

        Composed from the canonical byte-exact resource snapshot plus mode,
        active-voice policy, loaded models, and the current heavy job. No paths,
        no secrets. Shared GPU memory is reported separately and NEVER counted
        as dedicated VRAM.
        """
        from app import resources
        from app.engines.base import loaded_engines

        settings = get_settings()
        snap = resources.snapshot()
        heavy_job = None
        active_jobs = self._active_jobs_fn()
        if jobs is not None:
            heavy_job = jobs.active_heavy()
            active_jobs = len(jobs.active())
        elif self._active_heavy_fn is not None:
            heavy_job = self._active_heavy_fn()

        resident = self.resident_heavy_models()
        ollama_loaded = snap.get("ollama", {}).get("loaded") or []
        whisper = snap.get("whisper") or {}

        required = self.required_engine(profile) if profile is not None else None
        chain = self.effective_fallback_chain(profile) if profile is not None else []
        engines_loaded = snap.get("engines_loaded") or loaded_engines()

        return {
            "mode": self.mode,
            "active_voice": getattr(profile, "name", None),
            "required_tts_engine": required,
            "fallback_chain": chain,
            "fallback_status": self._fallback_status(profile, required),
            "loaded_voice_lab_model": (resident[0] if resident else None),
            "resident_heavy_models": resident,
            "engines_loaded": engines_loaded,
            "kokoro_loaded": bool(engines_loaded.get(KOKORO_ENGINE)),
            "ollama_loaded": [m.get("name") for m in ollama_loaded],
            "ollama": {"loaded": ollama_loaded},
            "whisper_loaded": bool(whisper.get("loaded")),
            "whisper": whisper,
            "system_ram_available_gb": snap.get("system_ram_available_gb"),
            "dedicated_vram_free_gb": snap.get("dedicated_vram_free_gb"),
            "shared_gpu_memory": snap.get("shared_gpu_memory"),  # never VRAM
            "gpu_name": snap.get("gpu_name"),
            "vram_source": snap.get("vram_source"),
            "current_heavy_job": heavy_job,
            "active_jobs": active_jobs,
            "seconds_idle": round(self.seconds_idle(), 1),
            "estimated_vram_gb": {m: self.estimated_vram_gb(m) for m in resident},
            "thresholds": {
                "min_free_ram_gb": settings.min_free_system_ram_gb,
                "min_free_vram_gb": settings.min_free_vram_gb,
                "idle_unload_seconds": settings.idle_unload_seconds,
                "max_loaded_heavy_models": settings.max_loaded_heavy_models,
                "allow_kokoro_fallback": settings.allow_kokoro_fallback,
                "designer_auto_unload_ollama": settings.designer_auto_unload_ollama,
            },
        }

    def _fallback_status(self, profile: Any | None, required: str | None) -> str:
        if profile is None or required is None:
            return "sin voz activa"
        if required != profile.engine:
            return (
                f"fallback activo: se usará {required} en lugar de {profile.engine} "
                f"(modo {self.mode})"
            )
        return f"motor principal {required} (sin fallback)"


# -- module singleton -------------------------------------------------------------------

_COORDINATOR: ModelCoordinator | None = None
_singleton_lock = threading.Lock()


def get_coordinator() -> ModelCoordinator:
    global _COORDINATOR
    with _singleton_lock:
        if _COORDINATOR is None:
            _COORDINATOR = ModelCoordinator()
        return _COORDINATOR


def reset_coordinator(state_dir: Path | None = None) -> ModelCoordinator:
    """Replace the singleton — for tests and a clean worker startup."""
    global _COORDINATOR
    with _singleton_lock:
        _COORDINATOR = ModelCoordinator(state_dir=state_dir)
        return _COORDINATOR
