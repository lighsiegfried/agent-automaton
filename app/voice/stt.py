"""Speech-to-text via faster-whisper (optional dependency).

faster-whisper is NOT in the base requirements. When it is missing, the
service reports a clear error instead of crashing — install it with:
pip install -r requirements-voice.txt

Model size (STT_MODEL), device (STT_DEVICE) and compute type
(STT_COMPUTE_TYPE) are configurable; the RTX 5060 Ti 16 GB comfortably runs
'medium' or 'large-v3'. Language detection is automatic (VOICE_LANGUAGE=auto)
and handles both Spanish and English.
"""

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


class SpeechToTextService:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.stt_model
        self.device = device or settings.stt_device
        self.compute_type = compute_type or settings.stt_compute_type
        self._model = None  # loaded lazily on first transcription

    @staticmethod
    def available() -> bool:
        return WhisperModel is not None

    def _load(self):
        if self._model is None:
            log.info(
                "loading faster-whisper model %r (device=%s, compute_type=%s)",
                self.model_name,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

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
            segments, info = self._load().transcribe(str(wav), language=language)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:  # model/audio failures must not crash the API
            log.exception("transcription failed")
            return {"error": f"Transcription failed: {exc}"}

        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration_seconds": round(float(getattr(info, "duration", 0.0)), 2),
        }


@lru_cache
def get_stt_service() -> SpeechToTextService:
    """Shared instance so the whisper model is only loaded once."""
    return SpeechToTextService()
