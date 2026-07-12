"""Pluggable TTS engine contract + registry.

Rules every engine must obey (Phase 3D.0):
- LAZY: importing an engine module is cheap; heavy imports (torch, kokoro,
  transformers) happen only inside load(), on first synthesis.
- SAFE: engines synthesize audio and nothing else — they never execute
  commands, never touch the safety layer, and never raise out of synthesize();
  failures return a structured error so the worker can fall back.
- ISOLATED: model downloads/caches land under voice_lab/ (config.pin_cache_env).
- QUIET: error messages must not leak absolute filesystem paths.
"""

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.profiles.schema import VoiceProfile


class EngineUnavailable(Exception):
    """Dependencies/model missing — carries a user-safe install hint."""


class TtsEngine(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self._loaded = False
        self._lock = threading.Lock()  # one synthesis at a time per engine

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(ok, reason) — dependency check only; must be cheap and never raise."""

    @abstractmethod
    def _load(self, profile: VoiceProfile) -> None:
        """Heavy one-time load (models onto device). May raise EngineUnavailable."""

    @abstractmethod
    def _synthesize(self, text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        """Write a WAV to out_path; return engine metadata (sample_rate etc.)."""

    def synthesize(self, text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        """Load lazily, synthesize, and NEVER raise — errors come back structured."""
        started = time.monotonic()
        ok, reason = self.available()
        if not ok:
            return {"status": "unavailable", "engine": self.name, "message": reason}
        try:
            with self._lock:
                if not self._loaded:
                    self._load(profile)
                    self._loaded = True
                meta = self._synthesize(text, profile, out_path)
        except EngineUnavailable as exc:
            return {"status": "unavailable", "engine": self.name, "message": str(exc)}
        except Exception as exc:
            from app.progress import Cancelled

            if isinstance(exc, Cancelled):
                raise  # user cancellation is not an engine failure — no fallback
            return {
                "status": "error",
                "engine": self.name,
                "message": f"{self.name} synthesis failed: {type(exc).__name__}: {exc}",
            }
        return {
            "status": "ok",
            "engine": self.name,
            "seconds": round(time.monotonic() - started, 2),
            **meta,
        }

    def unload(self) -> bool:
        """Release models/VRAM. Returns whether anything was actually unloaded."""
        with self._lock:
            was_loaded = self._loaded
            self._unload()
            self._loaded = False
        return was_loaded

    def _unload(self) -> None:  # engines with GPU state override this
        pass


# --- registry (lazy singletons) ----------------------------------------------------

_instances: dict[str, TtsEngine] = {}
_registry_lock = threading.Lock()


def _build(name: str) -> TtsEngine:
    # Imports stay inside so merely importing base.py pulls in nothing heavy.
    if name == "windows_sapi":
        from app.engines.windows_sapi import WindowsSapiEngine

        return WindowsSapiEngine()
    if name == "kokoro":
        from app.engines.kokoro import KokoroEngine

        return KokoroEngine()
    if name == "qwen3_tts":
        from app.engines.qwen3_tts import Qwen3TtsEngine

        return Qwen3TtsEngine()
    raise EngineUnavailable(f"Unknown engine {name!r}")


def get_engine(name: str) -> TtsEngine:
    with _registry_lock:
        if name not in _instances:
            _instances[name] = _build(name)
        return _instances[name]


def loaded_engines() -> dict[str, bool]:
    with _registry_lock:
        return {name: engine.loaded for name, engine in _instances.items()}


def unload_all(only: str | None = None) -> list[str]:
    """Unload one or all engines; returns the names that were actually loaded."""
    with _registry_lock:
        targets = (
            [_instances[only]] if only and only in _instances else
            [] if only else list(_instances.values())
        )
    return [engine.name for engine in targets if engine.unload()]
