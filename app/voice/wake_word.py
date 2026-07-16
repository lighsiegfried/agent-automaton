"""Wake word detection service (Phase 3D.1) — Windows-host processing only.

Detects the wake word ("Fifi") in 16 kHz mono PCM int16 frames using a CUSTOM
openWakeWord ONNX model, gated by Silero VAD to reduce false positives (frames
without speech are never scored as wake words).

Boundaries:
- The API server never instantiates this service. Only the explicit host-side
  listener (scripts/fifi_wake.py) does — there is no always-on microphone in
  the API and this module never opens one itself: callers feed PCM frames.
- The custom model is expected at WAKE_WORD_MODEL_PATH (models/wake_words/
  fifi.onnx). It is NEVER downloaded or replaced automatically; when it is
  missing (or the optional dependencies are), load() reports "unavailable".
- Detection only ever leads to the normal /voice/command pipeline — the wake
  word cannot bypass the safety layer, confirmations, or tool permissions.

Dependencies (optional, host-only):  pip install -r requirements-wakeword.txt
"""

from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings

SAMPLE_RATE = 16000
# Silero VAD scores fixed 512-sample windows at 16 kHz.
VAD_WINDOW_SAMPLES = 512

INSTALL_HINT = (
    "wake-word dependencies are not installed. Install them on the Windows host: "
    "pip install -r requirements-wakeword.txt"
)

_REQUIRED_MODULES = ("openwakeword", "onnxruntime", "silero_vad", "numpy")


def check_wake_deps() -> tuple[bool, list[str]]:
    """(ok, missing) for the optional wake-word dependencies. Never raises."""
    missing: list[str] = []
    for module in _REQUIRED_MODULES:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    return (not missing, missing)


def resolve_model_path(path: str | Path | None = None) -> Path:
    raw = Path(path or get_settings().wake_word_model_path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


class WakeWordService:
    """Scores PCM frames for the wake word; VAD-gated. Feed-only, no microphone."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float | None = None,
        vad_threshold: float | None = None,
        verifier_enabled: bool | None = None,
        verifier_path: str | Path | None = None,
        verifier_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = resolve_model_path(model_path)
        self.threshold = settings.wake_word_threshold if threshold is None else threshold
        self.vad_threshold = (
            settings.vad_threshold if vad_threshold is None else vad_threshold
        )
        # Optional speaker verifier (Phase 3D.2). Off by default; only consulted
        # when explicitly enabled AND the verifier file exists at load().
        self.verifier_enabled = (
            settings.enable_wake_word_verifier
            if verifier_enabled is None
            else verifier_enabled
        )
        self.verifier_path = resolve_model_path(
            verifier_path or settings.wake_word_verifier_path
        )
        self.verifier_threshold = (
            settings.wake_word_verifier_threshold
            if verifier_threshold is None
            else verifier_threshold
        )
        self._oww = None
        self._vad = None
        # Last wake score seen by detect() — for logging/threshold calibration.
        self.last_score: float = 0.0

    @property
    def verifier_active(self) -> bool:
        """True only when the verifier is enabled AND its file exists on disk."""
        return bool(self.verifier_enabled and self.verifier_path.is_file())

    @property
    def model_present(self) -> bool:
        return self.model_path.is_file()

    @property
    def loaded(self) -> bool:
        return self._oww is not None and self._vad is not None

    def status(self) -> dict[str, Any]:
        deps_ok, missing = check_wake_deps()
        return {
            "dependencies_ok": deps_ok,
            "missing": missing,
            "model_path": str(self.model_path),
            "model_present": self.model_present,
            "available": deps_ok and self.model_present,
            "loaded": self.loaded,
            "threshold": self.threshold,
            "vad_threshold": self.vad_threshold,
            "verifier_enabled": self.verifier_enabled,
            "verifier_active": self.verifier_active,
            "verifier_path": str(self.verifier_path),
        }

    def load(self) -> dict[str, Any]:
        """Load the custom wake model + Silero VAD. NEVER downloads anything.

        Returns {"status": "ok"} or {"status": "unavailable", "message": ...}.
        """
        deps_ok, missing = check_wake_deps()
        if not deps_ok:
            return {
                "status": "unavailable",
                "message": f"{INSTALL_HINT} (missing: {', '.join(missing)})",
            }
        if not self.model_present:
            return {
                "status": "unavailable",
                "message": (
                    f"Custom wake model not found at {self.model_path}. It is never "
                    "downloaded or replaced automatically — place your trained "
                    "fifi.onnx there (see models/wake_words/README.md)."
                ),
            }
        if self.loaded:
            return {"status": "ok"}

        try:
            from openwakeword.model import Model

            # Explicit model list + onnx framework: openWakeWord must load OUR
            # file only, never fetch or substitute a pretrained model.
            model_kwargs: dict[str, Any] = {
                "wakeword_models": [str(self.model_path)],
                "inference_framework": "onnx",
            }
            # Only when the optional verifier is enabled AND present do we add
            # the custom-verifier gate — otherwise the kwargs are exactly the
            # bare model load (a missing/disabled verifier changes nothing).
            if self.verifier_active:
                model_kwargs["custom_verifier_models"] = {
                    self.model_path.stem: str(self.verifier_path)
                }
                model_kwargs["custom_verifier_threshold"] = self.verifier_threshold
            self._oww = Model(**model_kwargs)
        except Exception as exc:
            self._oww = None
            return {
                "status": "unavailable",
                "message": (
                    f"Failed to load the wake model ({exc}). If openWakeWord's "
                    "feature-extraction models are missing, install them once, "
                    "deliberately, with: python -c \"import openwakeword.utils; "
                    "openwakeword.utils.download_models()\" — this service never "
                    "downloads anything on its own."
                ),
            }

        try:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad(onnx=True)
        except Exception as exc:
            self._oww = None
            self._vad = None
            return {"status": "unavailable", "message": f"Failed to load Silero VAD: {exc}"}
        return {"status": "ok"}

    # -- frame processing (16 kHz mono int16 PCM) ----------------------------------

    def speech_probability(self, frame) -> float:
        """Max Silero VAD speech probability across the frame's 512-sample windows."""
        import numpy as np
        import torch

        if self._vad is None:
            raise RuntimeError("WakeWordService.load() must succeed before use")
        samples = np.asarray(frame, dtype=np.int16).reshape(-1)
        if samples.size < VAD_WINDOW_SAMPLES:
            samples = np.pad(samples, (0, VAD_WINDOW_SAMPLES - samples.size))
        audio = samples.astype(np.float32) / 32768.0
        best = 0.0
        for start in range(0, samples.size - VAD_WINDOW_SAMPLES + 1, VAD_WINDOW_SAMPLES):
            window = torch.from_numpy(audio[start : start + VAD_WINDOW_SAMPLES])
            best = max(best, float(self._vad(window, SAMPLE_RATE).item()))
        return best

    def is_speech(self, frame) -> bool:
        return self.speech_probability(frame) >= self.vad_threshold

    def score(self, frame) -> float:
        """Wake-word score for one frame (openWakeWord keeps internal context)."""
        import numpy as np

        if self._oww is None:
            raise RuntimeError("WakeWordService.load() must succeed before use")
        samples = np.asarray(frame, dtype=np.int16).reshape(-1)
        predictions = self._oww.predict(samples)
        return max(predictions.values()) if predictions else 0.0

    def detect(self, frame) -> bool:
        """VAD-gated detection: non-speech frames still feed the wake model's
        rolling context (so it hears whole words) but can never trigger."""
        score = self.score(frame)
        self.last_score = score
        if score < self.threshold:
            return False
        return self.is_speech(frame)

    def reset(self) -> None:
        """Clear the wake model's rolling audio context (e.g. after a trigger)."""
        if self._oww is not None:
            try:
                self._oww.reset()
            except Exception:
                pass
