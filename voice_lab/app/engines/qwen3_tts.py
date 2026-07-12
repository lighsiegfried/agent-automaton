"""Qwen3-TTS engine (VoiceDesign / CustomVoice) — expressive neural voices.

Uses the official `qwen-tts` runtime (Qwen3TTSModel). Two open 1.7B variants
are supported, selected by the profile's `model` field:

- Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign — the voice is DESIGNED from the
  profile's natural-language `style_instruction` (timbre, age, pitch, mood).
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice — 9 named premium timbres (profile
  `speaker`, e.g. "Serena"), optionally styled by `style_instruction`.

Heavy imports (qwen_tts, torch) happen only in _load(); weights (~3.4 GB per
variant, bf16) download on first load into voice_lab/models (HF cache pinned
by config.pin_cache_env). One model is cached per model id.

Like every engine here it synthesizes audio and nothing else: no commands, no
safety-relevant behavior. A CUDA OOM releases VRAM and returns a structured
failure so the worker falls back (kokoro -> windows_sapi) — the main API and
Kokoro are never affected.
"""

from pathlib import Path
from typing import Any

from app.engines.base import EngineUnavailable, TtsEngine
from app.profiles.schema import VoiceProfile

INSTALL_HINT = (
    "Qwen3-TTS is optional and not installed. Opt in explicitly with: "
    "python voice_lab/scripts/setup.py --with-qwen"
)

OOM_HINT = (
    "GPU out of memory while running Qwen3-TTS. VRAM was released; the worker "
    "falls back (kokoro -> windows_sapi). Free VRAM (e.g. "
    "python scripts/local_runtime.py unload) or use a smaller model."
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
# 0.6B keeps clone inference light enough to coexist with the VoiceDesign
# model, Kokoro, Ollama, and whisper on the 16 GB card.
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# qwen-tts expects full language names.
_LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


def map_language(language: str) -> str:
    """Profile language code -> qwen-tts language name (default English)."""
    return _LANGUAGE_NAMES.get((language or "en").strip().lower(), "English")


def is_voice_design(model_id: str) -> bool:
    return "voicedesign" in (model_id or DEFAULT_MODEL_ID).lower()


def is_clone_base(model_id: str) -> bool:
    """Base checkpoints do reference-audio voice cloning (frozen identities)."""
    return (model_id or DEFAULT_MODEL_ID).lower().endswith("-base")


def _release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Qwen3TtsEngine(TtsEngine):
    name = "qwen3_tts"

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, Any] = {}  # model id -> Qwen3TTSModel
        self._device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import qwen_tts  # noqa: F401
            import soundfile  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False, INSTALL_HINT
        return True, ""

    def _load(self, profile: VoiceProfile) -> None:
        self._get_model(profile)  # cached per model id

    def _get_model(self, profile: VoiceProfile):
        model_id = profile.model or DEFAULT_MODEL_ID
        if model_id in self._models:
            return self._models[model_id]

        from app import gpu, progress
        from app.config import pin_cache_env

        pin_cache_env()  # downloads/caches must land inside voice_lab/
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise EngineUnavailable(INSTALL_HINT) from exc

        short_name = model_id.rsplit("/", 1)[-1]
        progress.report(
            "checking_cache", model=model_id, engine=self.name,
            message="Comprobando caché local de modelos…",
        )
        cached = gpu.model_cache_status(model_id) == "cached"
        watcher_stop = None
        if cached:
            progress.report(
                "cached", cache_hit=True,
                message="Modelo encontrado en caché local; no se requiere descarga.",
            )
        else:
            # A download message appears ONLY when a real download happens.
            watcher_stop = self._watch_download(model_id)
            progress.report(
                "downloading", cache_hit=False,
                message=f"Descargando {short_name} (solo la primera vez)…",
            )

        device = profile.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        # bf16 is the recommended precision; float16 profiles map to bf16 too
        # (the released checkpoints are bf16-native).
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        gpu_name = (gpu.gpu_snapshot() or {}).get("name") or "GPU"
        # Two REAL stages (Phase 3D.0.3a): load from disk into RAM, then move
        # to the GPU. Loading directly with device_map="cuda:0" segfaults on
        # this machine (torch 2.11.0+cu128, RTX 5060 Ti) and killed the worker
        # with no traceback — the CPU-then-transfer path is verified stable.
        progress.report(
            "loading_from_disk", message=f"Cargando {short_name} desde disco a RAM…"
        )
        try:
            model = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", dtype=dtype)
        except Exception as exc:
            # The model id (not a path) is safe to show; the hint tells the fix.
            raise EngineUnavailable(
                f"Could not load {model_id!r}: {type(exc).__name__}. {INSTALL_HINT}"
            ) from exc
        finally:
            if watcher_stop is not None:
                watcher_stop.set()
        if device == "cuda":
            progress.report(
                "moving_to_gpu", message=f"Transfiriendo {short_name} a {gpu_name}…"
            )
            try:
                model.model = model.model.to("cuda:0")
            except torch.cuda.OutOfMemoryError as exc:
                del model
                _release_cuda_memory()
                raise EngineUnavailable(OOM_HINT) from exc
            # The wrapper snapshots .device at __init__ — refresh it so
            # generation inputs land on the GPU with the weights.
            model.device = next(model.model.parameters()).device
        progress.report(
            "warming_cuda",
            message="Calentando kernels CUDA (la primera síntesis tarda más)…",
        )
        self._device = device
        self._models[model_id] = model
        return model

    @staticmethod
    def _watch_download(model_id: str, poll_seconds: float = 2.0):
        """Report real cache growth while a download runs. Returns a stop event."""
        import threading

        from app import gpu, progress

        stop = threading.Event()
        baseline = gpu.cache_dir_size_bytes()
        parent_reporter = progress.current_reporter()

        def _poll() -> None:
            if parent_reporter is not None:
                progress.set_reporter(parent_reporter)  # inherit the job's reporter
            while not stop.wait(poll_seconds):
                grown = gpu.cache_dir_size_bytes() - baseline
                if grown > 0:
                    progress.report(
                        "downloading", bytes_downloaded=grown,
                        message=f"Descargando… {grown / 2**20:.0f} MB recibidos",
                    )

        threading.Thread(target=_poll, daemon=True, name="hf-download-watch").start()
        return stop

    def unload_model(self, model_id: str) -> bool:
        """Drop ONE cached model (e.g. VoiceDesign before loading the clone).

        Framework cleanup (del + gc + empty_cache) runs only once the model is
        out of the registry — cached files on disk are never touched.
        """
        with self._lock:
            model = self._models.pop(model_id, None)
            if model is None:
                return False
            del model
        import gc

        gc.collect()
        _release_cuda_memory()
        return True

    def _synthesize(self, text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        import soundfile as sf
        import torch

        try:
            wav, sample_rate, mode = self._generate(text, profile)
        except torch.cuda.OutOfMemoryError as exc:
            # OOM must never poison the process: release VRAM and report a
            # structured failure the worker can fall back on.
            _release_cuda_memory()
            raise EngineUnavailable(OOM_HINT) from exc

        sf.write(str(out_path), wav, sample_rate)
        return {
            "sample_rate": sample_rate,
            "device": self._device,
            "mode": mode,
            "style_instruction": profile.style_instruction,
            "duration_seconds": round(len(wav) / sample_rate, 2),
        }

    def _generate(self, text: str, profile: VoiceProfile):
        model = self._get_model(profile)
        language = map_language(profile.language)
        model_id = profile.model or DEFAULT_MODEL_ID
        if is_clone_base(model_id):
            wavs, sample_rate = self._generate_clone(model, text, language, profile)
            return wavs[0], int(sample_rate), "voice_clone"
        if is_voice_design(model_id):
            if not profile.style_instruction:
                raise RuntimeError(
                    f"profile {profile.name!r} uses the VoiceDesign model but has "
                    "no style_instruction — the instruction IS the voice design"
                )
            wavs, sample_rate = model.generate_voice_design(
                text=text, language=language, instruct=profile.style_instruction
            )
            mode = "voice_design"
        else:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=profile.speaker or "Serena",
                instruct=profile.style_instruction or None,
            )
            mode = "custom_voice"
        return wavs[0], int(sample_rate), mode

    def _generate_clone(self, model, text: str, language: str, profile: VoiceProfile):
        """Frozen identity: clone from the design's reference audio+transcript.

        Cloning is authorization-gated (see app.profiles.designs) — synthetic
        VoiceDesign references are self-authorized; external references need an
        explicit authorization record or synthesis is refused.
        """
        from app.profiles.designs import clone_authorized, load_design, reference_audio_path

        design = load_design(profile.name)
        if design is None:
            raise RuntimeError(
                f"profile {profile.name!r} uses a clone model but has no design "
                "metadata (config/voices/designs/) — freeze the identity first"
            )
        authorized, reason = clone_authorized(design)
        if not authorized:
            raise RuntimeError(reason)
        ref_path = reference_audio_path(design)
        ref_text = (design.get("reference") or {}).get("text") or ""
        if ref_path is None or not ref_path.is_file() or not ref_text:
            raise RuntimeError(
                f"reference audio/transcript for {profile.name!r} is missing — "
                "regenerate and freeze the identity in the Voice Designer"
            )
        return model.generate_voice_clone(
            text=text, language=language, ref_audio=str(ref_path), ref_text=ref_text
        )

    def _unload(self) -> None:
        self._models = {}
        _release_cuda_memory()
