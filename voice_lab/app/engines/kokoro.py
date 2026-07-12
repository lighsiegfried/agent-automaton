"""Kokoro-82M engine — small, fast local neural TTS (24 kHz), Spanish + English.

Heavy imports (kokoro, torch) happen only in _load(). The model downloads on
first load into voice_lab/models (HF cache pinned by config.pin_cache_env) —
never into the user profile or the main project.

Language is selected per profile ("es", "en", ...); one KPipeline is cached per
language code so switching profiles never reloads the model weights. Speakers
are validated against the profile language before synthesis: a mismatched or
unknown speaker returns a clear error (which lets the worker fall back) instead
of producing garbled audio.
"""

from pathlib import Path
from typing import Any

from app.engines.base import EngineUnavailable, TtsEngine
from app.profiles.schema import VoiceProfile

INSTALL_HINT = (
    "kokoro/torch are not installed in the voice_lab environment: "
    "run python voice_lab/scripts/setup.py"
)

KOKORO_SAMPLE_RATE = 24000

# Kokoro selects the language via a one-letter pipeline code.
_LANG_CODES = {"en": "a", "en-gb": "b", "es": "e", "fr": "f", "it": "i", "pt": "p"}

# Voice ids are prefixed by language + gender (e.g. "ef_" = español, female).
# A speaker whose prefix disagrees with the profile language is rejected with a
# clear error — Kokoro would otherwise synthesize garbled audio, not fail.
_VOICE_PREFIXES = {"a": ("af_", "am_"), "b": ("bf_", "bm_"), "e": ("ef_", "em_"),
                   "f": ("ff_",), "i": ("if_", "im_"), "p": ("pf_", "pm_")}
_DEFAULT_VOICES = {"a": "af_heart", "b": "bf_emma", "e": "ef_dora",
                   "f": "ff_siwis", "i": "if_sara", "p": "pf_dora"}


def resolve_lang_code(language: str) -> str:
    return _LANG_CODES.get((language or "en").lower(), "a")


def validate_voice(language: str, speaker: str) -> tuple[str, str]:
    """(voice_id, error). Empty speaker gets the language default; a speaker
    that cannot belong to the language returns a clear error message."""
    lang_code = resolve_lang_code(language)
    if not speaker:
        return _DEFAULT_VOICES.get(lang_code, "af_heart"), ""
    prefixes = _VOICE_PREFIXES.get(lang_code, ())
    if prefixes and not speaker.startswith(prefixes):
        return "", (
            f"speaker {speaker!r} does not match language {language!r} "
            f"(expected a voice starting with {' or '.join(prefixes)}, "
            f"e.g. {_DEFAULT_VOICES.get(lang_code)!r})"
        )
    return speaker, ""


class KokoroEngine(TtsEngine):
    name = "kokoro"

    def __init__(self) -> None:
        super().__init__()
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline
        self._device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False, INSTALL_HINT
        return True, ""

    def _load(self, profile: VoiceProfile) -> None:
        from app.config import pin_cache_env

        pin_cache_env()  # downloads/caches must land inside voice_lab/
        try:
            import torch
        except Exception as exc:
            raise EngineUnavailable(INSTALL_HINT) from exc
        device = profile.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"  # degraded but working beats broken
        self._device = device
        self._get_pipeline(resolve_lang_code(profile.language))

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from app import gpu, progress

            try:
                from kokoro import KPipeline
            except Exception as exc:
                raise EngineUnavailable(INSTALL_HINT) from exc
            cached = gpu.model_cache_status("hexgrad/Kokoro-82M") == "cached"
            progress.report(
                "loading", engine=self.name, model="hexgrad/Kokoro-82M", cache_hit=cached,
                message=(
                    "Modelo Kokoro encontrado en caché local; no se requiere descarga."
                    if cached else "Descargando Kokoro-82M (solo la primera vez)…"
                ),
            )
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, device=self._device
            )
        return self._pipelines[lang_code]

    def _synthesize(self, text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        voice, error = validate_voice(profile.language, profile.speaker)
        if error:
            raise RuntimeError(error)  # structured by base.synthesize -> fallback

        lang_code = resolve_lang_code(profile.language)
        pipeline = self._get_pipeline(lang_code)
        chunks = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=profile.speed):
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("kokoro produced no audio")
        audio = np.concatenate([np.asarray(c) for c in chunks])
        sf.write(str(out_path), audio, KOKORO_SAMPLE_RATE)
        return {
            "sample_rate": KOKORO_SAMPLE_RATE,
            "device": self._device,
            "voice": voice,
            "language": profile.language,
            "duration_seconds": round(len(audio) / KOKORO_SAMPLE_RATE, 2),
        }

    def _unload(self) -> None:
        self._pipelines = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
