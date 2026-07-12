"""Speech-to-text via faster-whisper (optional dependency).

faster-whisper is NOT in the base requirements. When it is missing, the
service reports a clear error instead of crashing — install it with:
pip install -r requirements-voice.txt

Phase 3C.1 — Blackwell safety: NVIDIA RTX 50 (sm_120) GPUs raise
CUBLAS_STATUS_NOT_SUPPORTED on INT8 cuBLAS variants. resolve_stt_config()
detects the GPU and pins float16 (never INT8) on Blackwell, resolving the
`auto` settings to concrete, safe values. A CUDA/STT failure returns a
structured error and never crashes the API; CPU inference is opt-in only.
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError:  # optional dependency
    WhisperModel = None

INSTALL_HINT = (
    "faster-whisper is not installed. Install the optional voice dependencies: "
    "pip install -r requirements-voice.txt"
)

# Serialize model load + inference: one WhisperModel is shared, faster-whisper
# is not guaranteed thread-safe, and one GPU should not run concurrent decodes.
# The API already offloads this to a worker thread, so blocking here is safe.
_INFERENCE_LOCK = threading.Lock()


# --- GPU architecture detection --------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    name: str | None
    compute_capability: float | None  # e.g. 12.0 for RTX 50 (sm_120)

    @property
    def is_blackwell(self) -> bool:
        # Blackwell is compute capability 10.0+ (datacenter) / 12.0 (RTX 50).
        if self.compute_capability is not None and self.compute_capability >= 10.0:
            return True
        name = (self.name or "").lower()
        if "blackwell" in name:
            return True
        # RTX 5060/5070/5080/5090 and the "RTX 50x0" family.
        return bool(re.search(r"rtx\s*50[0-9]0", name)) or bool(re.search(r"\b50[6789]0\b", name))


@lru_cache(maxsize=1)
def detect_gpu_info() -> GpuInfo:
    """Best-effort GPU name + compute capability via nvidia-smi. Never raises."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return GpuInfo(None, None)
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:  # best-effort — a probe failure must never raise
        return GpuInfo(None, None)

    if result.returncode == 0 and result.stdout.strip():
        parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
        name = parts[0] if parts and parts[0] else None
        cap: float | None = None
        if len(parts) > 1:
            try:
                cap = float(parts[1])
            except ValueError:
                cap = None
        return GpuInfo(name, cap)

    # Older drivers may not support the compute_cap field — fall back to name.
    try:
        name_only = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:  # best-effort — a probe failure must never raise
        return GpuInfo(None, None)
    if name_only.returncode == 0 and name_only.stdout.strip():
        return GpuInfo(name_only.stdout.strip().splitlines()[0].strip(), None)
    return GpuInfo(None, None)


def ctranslate2_version() -> str | None:
    try:
        import ctranslate2

        return getattr(ctranslate2, "__version__", None)
    except Exception:
        return None


def cuda_device_count() -> int | None:
    """CUDA device count per CTranslate2, or None when CT2 is unavailable."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return None


def resolve_stt_config(
    device: str, compute_type: str, gpu: GpuInfo, allow_cpu_fallback: bool
) -> tuple[str, str]:
    """Resolve (device, compute_type) to concrete, Blackwell-safe values.

    - `auto` device becomes cuda when a GPU is present, else cpu.
    - `auto` compute becomes float16 on cuda, int8 on cpu.
    - Blackwell (RTX 50) NEVER uses an INT8 compute variant on cuda — it is
      forced to float16 (INT8 cuBLAS raises CUBLAS_STATUS_NOT_SUPPORTED there).
    """
    device = (device or "auto").strip().lower()
    compute_type = (compute_type or "auto").strip().lower()

    count = cuda_device_count()
    cuda_present = bool((count or 0) > 0) or (gpu.name is not None)

    if device == "cpu":
        resolved_device = "cpu"
    elif device == "auto":
        resolved_device = "cuda" if cuda_present else "cpu"
    else:
        resolved_device = device  # explicit (e.g. "cuda")

    if resolved_device == "cuda":
        resolved_compute = "float16" if compute_type == "auto" else compute_type
        if gpu.is_blackwell and "int8" in resolved_compute:
            log.warning(
                "STT: %s is Blackwell — INT8 compute (%s) is unsupported; using float16",
                gpu.name,
                resolved_compute,
            )
            resolved_compute = "float16"
    else:  # cpu
        # float16 is a GPU format; on CPU, int8 is the sane, supported default.
        if compute_type in ("auto", "float16"):
            resolved_compute = "int8"
        else:
            resolved_compute = compute_type

    return resolved_device, resolved_compute


def _write_silence_wav(seconds: float = 0.3, rate: int = 16000) -> str:
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return name


class SpeechToTextService:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.stt_model
        self._requested_device = device or settings.stt_device
        self._requested_compute = compute_type or settings.stt_compute_type
        self.allow_cpu_fallback = settings.stt_allow_cpu_fallback
        self.gpu = detect_gpu_info()
        self.ct2_version = ctranslate2_version()
        self.device, self.compute_type = resolve_stt_config(
            self._requested_device, self._requested_compute, self.gpu, self.allow_cpu_fallback
        )
        self.cpu_fallback_active = False
        self._model = None  # loaded lazily on first transcription

    @staticmethod
    def available() -> bool:
        return WhisperModel is not None

    @property
    def loaded(self) -> bool:
        """Whether the whisper model is already in memory (warm)."""
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "model": self.model_name,
            "gpu": self.gpu.name,
            "gpu_compute_capability": self.gpu.compute_capability,
            "blackwell": self.gpu.is_blackwell,
            "ctranslate2_version": self.ct2_version,
            "cpu_fallback_active": self.cpu_fallback_active,
        }

    def _load(self):
        if self._model is not None:
            return self._model
        log.info(
            "loading faster-whisper %r device=%s compute_type=%s ct2=%s gpu=%s blackwell=%s",
            self.model_name,
            self.device,
            self.compute_type,
            self.ct2_version,
            self.gpu.name,
            self.gpu.is_blackwell,
        )
        try:
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:
            if self.allow_cpu_fallback and self.device != "cpu":
                log.warning(
                    "STT load failed on %s/%s (%s) — CPU fallback is enabled; "
                    "retrying on cpu/int8 (slow)",
                    self.device,
                    self.compute_type,
                    exc,
                )
                self.device, self.compute_type = "cpu", "int8"
                self.cpu_fallback_active = True
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            else:
                raise
        return self._model

    def preflight(self) -> dict[str, Any]:
        """Load the model and run a harmless silence probe (exercises the cuBLAS
        encoder path). Executes NO assistant tool. Returns a structured report."""
        if not self.available():
            return {**self.describe(), "status": "unavailable", "passed": False, "message": INSTALL_HINT}
        probe = _write_silence_wav()
        try:
            with _INFERENCE_LOCK:
                model = self._load()
                segments, _info = model.transcribe(probe, language=None)
                for _ in segments:  # force the generator so the encoder actually runs
                    pass
        except Exception as exc:
            log.exception("STT preflight failed")
            return {
                **self.describe(),
                "status": "error",
                "passed": False,
                "message": f"STT preflight failed: {exc}",
            }
        finally:
            try:
                os.unlink(probe)
            except OSError:
                pass
        return {**self.describe(), "status": "ok", "passed": True, "message": "STT preflight passed."}

    def transcribe_file(self, path: str | Path) -> dict[str, Any]:
        """Transcribe a .wav file. Returns {"text", "language", ...} or {"error"}."""
        if not self.available():
            return {"error": INSTALL_HINT}

        wav = Path(path)
        if not wav.exists():
            return {"error": f"Audio file not found: {wav}"}
        if wav.suffix.lower() != ".wav":
            return {"error": "Only .wav files are supported."}

        configured = get_settings().voice_language
        language = None if configured == "auto" else configured

        try:
            with _INFERENCE_LOCK:
                segments, info = self._load().transcribe(str(wav), language=language)
                text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:  # model/CUDA/audio failures must not crash the API
            log.exception("transcription failed")
            return {"error": f"Transcription failed: {exc}", "cpu_fallback_active": self.cpu_fallback_active}

        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration_seconds": round(float(getattr(info, "duration", 0.0)), 2),
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_fallback_active": self.cpu_fallback_active,
        }


@lru_cache
def get_stt_service() -> SpeechToTextService:
    """Shared instance so the whisper model is only loaded once."""
    return SpeechToTextService()
