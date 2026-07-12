"""Voice Designer — create, compare, and freeze genuinely distinct identities.

Workflow (default mode: "voice_design_then_clone", the official Qwen3-TTS
consistency recipe):

1. GENERATE: a structured description (age, gender presentation, timbre,
   pitch, energy, speed, accent, emotion, personality + free-form text) is
   composed into one VoiceDesign instruction; 3-4 candidate variants are
   synthesized into voice_lab/storage/voice-identities/<profile>/ (temporary,
   gitignored, never committed).
2. AUDITION: play, A/B-compare, regenerate, rename, or delete variants.
3. FREEZE: the chosen variant's WAV becomes the identity's reference audio and
   its EXACT preview transcript is preserved; a reusable clone profile
   (Qwen3-TTS Base model) is written to config/voices/profiles/<name>.json and
   the design metadata to config/voices/designs/<name>.json. Every stored path
   is relative to voice_lab/.
4. ACTIVATE: /profile/select flips active.json atomically; Fifi hot-reloads.

Modes:
- builtin_speaker         — no design step; profiles reference a fixed speaker.
- voice_design            — freeze keeps the VoiceDesign model + instruction
                            (voice re-designed per synthesis; less consistent).
- voice_design_then_clone — DEFAULT: freeze clones the chosen variant.
- reference_clone         — clone an EXTERNAL reference recording; requires an
                            explicit authorization statement (safety-gated).

Synthesis is injectable so the whole designer is testable without any model,
GPU, or audio device.
"""

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.config import PROFILES_DIR, STORAGE_DIR, VOICE_LAB_ROOT, get_settings
from app.engines.base import get_engine
from app.engines.qwen3_tts import CLONE_MODEL_ID, DEFAULT_MODEL_ID
from app.profiles.designs import (
    DESIGNS_DIR,
    save_design,
    valid_identity_name,
)
from app.profiles.manager import atomic_write_json
from app.profiles.schema import VoiceProfile

IDENTITIES_DIR = STORAGE_DIR / "voice-identities"

MODES = ("builtin_speaker", "voice_design", "voice_design_then_clone", "reference_clone")
DEFAULT_MODE = "voice_design_then_clone"

# Designer form vocabulary — shown by GET /designer and used to compose the
# VoiceDesign instruction. Free-form text can always extend these.
FORM_OPTIONS = {
    "language": ["es", "en", "pt", "fr", "it", "de", "ja", "ko", "zh", "ru"],
    "age": ["niña", "joven", "adulta joven", "adulta", "madura"],
    "gender": ["femenina", "masculina", "andrógina"],
    "timbre": ["cálido", "brillante", "aterciopelado", "grave", "cristalino", "ronco", "nasal"],
    "pitch": ["muy agudo", "agudo", "medio-alto", "medio", "medio-grave", "grave"],
    "energy": ["muy tranquila", "tranquila", "moderada", "enérgica", "explosiva"],
    "speed": ["muy pausada", "pausada", "natural", "ágil", "rápida"],
    "accent": ["neutro", "castellano", "mexicano", "rioplatense", "caribeño", "andino"],
    "emotion": ["neutral", "alegre", "serena", "entusiasta", "cercana", "seria"],
    "personality": ["amable", "profesional", "juguetona", "directa", "soñadora", "sarcástica"],
}


class DesignerError(Exception):
    """User-facing designer failure; message is safe to return (no paths)."""


def compose_instruction(fields: dict[str, Any]) -> str:
    """Structured fields + free-form text -> one VoiceDesign instruction."""
    parts: list[str] = []
    gender = fields.get("gender") or "femenina"
    age = fields.get("age") or "adulta joven"
    parts.append(f"Voz {gender} de {age}")
    if fields.get("timbre"):
        parts.append(f"timbre {fields['timbre']}")
    if fields.get("pitch"):
        parts.append(f"tono {fields['pitch']}")
    if fields.get("energy"):
        parts.append(f"energía {fields['energy']}")
    if fields.get("speed"):
        parts.append(f"ritmo {fields['speed']}")
    if fields.get("accent"):
        parts.append(f"acento {fields['accent']}")
    if fields.get("emotion"):
        parts.append(f"emoción {fields['emotion']}")
    if fields.get("personality"):
        parts.append(f"personalidad {fields['personality']}")
    instruction = ", ".join(parts) + "."
    freeform = (fields.get("instruction") or "").strip()
    if freeform:
        instruction = f"{instruction} {freeform}"
    return instruction


class VoiceDesigner:
    def __init__(
        self,
        identities_dir: Path | None = None,
        profiles_dir: Path | None = None,
        designs_dir: Path | None = None,
        synth: Callable[..., dict[str, Any]] | None = None,
        root: Path | None = None,
    ) -> None:
        # Every stored path is RELATIVE to this root (voice_lab/ in production)
        # — manifests, designs, and API responses stay portable and path-free.
        self.root = Path(root or VOICE_LAB_ROOT)
        self.identities_dir = Path(identities_dir or IDENTITIES_DIR)
        self.profiles_dir = Path(profiles_dir or PROFILES_DIR)
        self.designs_dir = Path(designs_dir or DESIGNS_DIR)
        # synth(text, profile, out_path) -> engine result dict; injectable.
        self.synth = synth or self._engine_synth

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _abs(self, relative: str) -> Path | None:
        if not relative or ".." in relative or ":" in relative or relative.startswith(("/", "\\")):
            return None
        return self.root / relative

    @staticmethod
    def _engine_synth(text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        return get_engine(profile.engine).synthesize(text, profile, out_path)

    # -- manifest -------------------------------------------------------------------

    def _identity_dir(self, name: str) -> Path:
        if not valid_identity_name(name):
            raise DesignerError(
                f"invalid identity name {name!r} — use lowercase letters, digits, _ or -"
            )
        return self.identities_dir / name

    def _manifest_path(self, name: str) -> Path:
        return self._identity_dir(name) / "manifest.json"

    def manifest(self, name: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._manifest_path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_manifest(self, name: str, manifest: dict[str, Any]) -> None:
        atomic_write_json(self._manifest_path(name), manifest)

    def working_identities(self) -> list[dict[str, Any]]:
        if not self.identities_dir.is_dir():
            return []
        out = []
        for child in sorted(self.identities_dir.iterdir()):
            if child.is_dir():
                manifest = self.manifest(child.name)
                if manifest:
                    out.append(manifest)
        return out

    # -- variant generation ---------------------------------------------------------

    def _design_profile(self, name: str, language: str, instruction: str) -> VoiceProfile:
        return VoiceProfile(
            name=name,
            engine="qwen3_tts",
            model=DEFAULT_MODEL_ID,
            language=language,
            speaker="",
            style_instruction=instruction,
            dtype="bfloat16",
            fallback_engine="",  # designing must FAIL visibly, never fall back
        )

    def _synthesize_variant(
        self, name: str, variant_id: str, profile: VoiceProfile, text: str
    ) -> dict[str, Any]:
        out_path = self._identity_dir(name) / f"variant_{variant_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.synth(text, profile, out_path)
        if result.get("status") != "ok":
            raise DesignerError(
                f"variant generation failed ({result.get('engine')}): {result.get('message')}"
            )
        # A variant is only real once its audio VALIDATES (Phase 3D.0.4):
        # nonzero size, proper WAV header, positive duration.
        from app.audio.validation import validate_wav

        check = validate_wav(out_path)
        if not check["valid"]:
            try:
                out_path.unlink(missing_ok=True)  # never keep broken partials
            except OSError:
                pass
            raise DesignerError(f"generated audio failed validation: {check['reason']}")
        return {
            "id": variant_id,
            "label": f"variante {variant_id[:4]}",
            "file": self._rel(out_path),
            "text": text,  # the EXACT transcript of this WAV (frozen later)
            "seconds": result.get("seconds"),
            "duration_seconds": check["duration_seconds"],
            "audio_valid": True,
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create 3-4 variants of one described voice. Replaces prior variants."""
        name = (request.get("profile_name") or "").strip()
        directory = self._identity_dir(name)  # validates the name
        if (self.profiles_dir / f"{name}.json").exists():
            raise DesignerError(f"profile {name!r} already exists — pick a new name")
        mode = request.get("mode") or DEFAULT_MODE
        if mode not in MODES:
            raise DesignerError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
        if mode == "reference_clone":
            raise DesignerError(
                "reference_clone identities are frozen directly from an uploaded, "
                "authorized reference — variant generation applies to design modes"
            )
        count = int(request.get("count") or 3)
        if count not in (3, 4):
            raise DesignerError("count must be 3 or 4")

        language = (request.get("language") or "es").strip().lower()
        instruction = compose_instruction(request)
        text = (request.get("preview_text") or "").strip() or get_settings().preview_text
        profile = self._design_profile(name, language, instruction)

        if directory.exists():
            shutil.rmtree(directory)  # regenerating a working identity starts clean
        started = time.monotonic()
        variants: list[dict[str, Any]] = []
        failed_variants: list[dict[str, Any]] = []
        try:
            from app import progress

            for index in range(count):
                progress.check_cancelled()
                progress.report(
                    "generating",
                    current_item=index + 1,
                    total_items=count,
                    percentage=int(100 * index / count),
                    message=f"Generando variante {index + 1} de {count}…",
                )
                try:
                    variants.append(
                        self._synthesize_variant(name, uuid.uuid4().hex[:8], profile, text)
                    )
                except DesignerError as exc:
                    # Partial failure is TRACKED, never silent: the job layer
                    # refuses to report completed unless every variant is valid.
                    failed_variants.append({"index": index, "error": str(exc)})
            progress.report(
                "validating_audio", percentage=100,
                message="Validando el audio generado…",
            )
            if not variants:
                raise DesignerError(
                    "no variant produced valid audio — "
                    + "; ".join(f["error"] for f in failed_variants)
                )
            progress.report("saving", message="Guardando previews…")
        except BaseException:
            # Cancellation / total failure: never leave half an identity behind.
            shutil.rmtree(directory, ignore_errors=True)
            raise
        manifest = {
            "profile": name,
            "mode": mode,
            "language": language,
            "instruction": instruction,
            "fields": {k: request.get(k) or "" for k in FORM_OPTIONS},
            "freeform": (request.get("instruction") or "").strip(),
            "preview_text": text,
            "variants": variants,
            "failed_variants": failed_variants,
            "total_seconds": round(time.monotonic() - started, 2),
        }
        self._write_manifest(name, manifest)
        return manifest

    def regenerate(self, name: str, variant_id: str) -> dict[str, Any]:
        """Re-roll ONE variant with the same instruction/text."""
        manifest = self.manifest(name)
        if manifest is None:
            raise DesignerError(f"no working identity {name!r} — generate first")
        index = self._variant_index(manifest, variant_id)
        profile = self._design_profile(name, manifest["language"], manifest["instruction"])
        old = manifest["variants"][index]
        self._delete_variant_file(old)
        fresh = self._synthesize_variant(
            name, uuid.uuid4().hex[:8], profile, manifest["preview_text"]
        )
        fresh["label"] = old.get("label") or fresh["label"]
        manifest["variants"][index] = fresh
        self._write_manifest(name, manifest)
        return manifest

    def rename_variant(self, name: str, variant_id: str, label: str) -> dict[str, Any]:
        manifest = self.manifest(name)
        if manifest is None:
            raise DesignerError(f"no working identity {name!r}")
        index = self._variant_index(manifest, variant_id)
        manifest["variants"][index]["label"] = (label or "").strip()[:60] or "variante"
        self._write_manifest(name, manifest)
        return manifest

    def delete_variant(self, name: str, variant_id: str) -> dict[str, Any]:
        manifest = self.manifest(name)
        if manifest is None:
            raise DesignerError(f"no working identity {name!r}")
        index = self._variant_index(manifest, variant_id)
        self._delete_variant_file(manifest["variants"].pop(index))
        if manifest["variants"]:
            self._write_manifest(name, manifest)
        else:
            shutil.rmtree(self._identity_dir(name), ignore_errors=True)
        return manifest

    def variant_audio_path(self, name: str, variant_id: str) -> Path:
        manifest = self.manifest(name)
        if manifest is None:
            raise DesignerError(f"no working identity {name!r}")
        index = self._variant_index(manifest, variant_id)
        path = self._abs(manifest["variants"][index]["file"])
        if path is None:
            raise DesignerError("variant audio path is invalid")
        return path

    def _variant_index(self, manifest: dict[str, Any], variant_id: str) -> int:
        for index, variant in enumerate(manifest.get("variants", [])):
            if variant.get("id") == variant_id:
                return index
        raise DesignerError(f"unknown variant {variant_id!r}")

    def _delete_variant_file(self, variant: dict[str, Any]) -> None:
        path = self._abs(variant.get("file") or "")
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # -- freeze ----------------------------------------------------------------------

    def freeze(self, name: str, variant_id: str) -> dict[str, Any]:
        """Turn the chosen variant into a permanent, reusable voice identity.

        voice_design_then_clone (default): the variant WAV becomes the clone
        reference with its EXACT transcript; the profile uses the Base (clone)
        model. voice_design: the profile keeps the VoiceDesign model +
        instruction. The reference is synthetic → self-authorized.
        """
        manifest = self.manifest(name)
        if manifest is None:
            raise DesignerError(f"no working identity {name!r} — generate first")
        index = self._variant_index(manifest, variant_id)
        chosen = manifest["variants"][index]
        mode = manifest.get("mode") or DEFAULT_MODE

        identity_dir = self._identity_dir(name)
        reference_path = identity_dir / "reference.wav"
        source = self._abs(chosen["file"])
        if source is None or not source.is_file():
            raise DesignerError("chosen variant audio is missing — regenerate it")
        shutil.copyfile(source, reference_path)

        if mode == "voice_design":
            model_id = DEFAULT_MODEL_ID  # re-designed per synthesis
        else:
            model_id = CLONE_MODEL_ID  # cloned from the frozen reference

        profile = VoiceProfile(
            name=name,
            engine="qwen3_tts",
            model=model_id,
            language=manifest["language"],
            speaker="",
            style_instruction=manifest["instruction"],
            dtype="bfloat16",
            fallback_engine="kokoro",
            provenance={
                "source": "voice_designer",
                "mode": mode,
                "design": name,
                "phase": "3D.0.2",
            },
        )
        design = {
            "name": name,
            "mode": mode,
            "language": manifest["language"],
            "fields": manifest.get("fields") or {},
            "freeform": manifest.get("freeform") or "",
            "instruction": manifest["instruction"],
            "chosen_variant": {"id": chosen["id"], "label": chosen.get("label")},
            "reference": {
                "audio": self._rel(reference_path),
                "text": chosen["text"],  # exact transcript of the reference WAV
                "authorization": {
                    "type": "synthetic",
                    "note": "reference generated by Qwen3-TTS VoiceDesign from a text description; no real person's voice",
                },
            },
        }
        atomic_write_json(self.profiles_dir / f"{name}.json", profile.public_dict())
        save_design(name, design, self.designs_dir)

        # Tidy: audition variants are temporary; only the reference survives.
        for variant in manifest["variants"]:
            self._delete_variant_file(variant)
        manifest["variants"] = []
        manifest["frozen"] = {"profile": name, "reference": self._rel(reference_path)}
        self._write_manifest(name, manifest)
        return {"profile": profile.public_dict(), "design": design}

    def freeze_reference_clone(
        self,
        name: str,
        language: str,
        wav_bytes: bytes,
        transcript: str,
        authorization_statement: str,
    ) -> dict[str, Any]:
        """External-reference clone — requires explicit authorization (safety)."""
        directory = self._identity_dir(name)
        if (self.profiles_dir / f"{name}.json").exists():
            raise DesignerError(f"profile {name!r} already exists — pick a new name")
        statement = (authorization_statement or "").strip()
        if len(statement) < 10:
            raise DesignerError(
                "reference cloning requires an explicit authorization statement "
                "(who authorized the use of this voice and when)"
            )
        transcript = (transcript or "").strip()
        if not transcript:
            raise DesignerError("the reference transcript is required for cloning")
        if not wav_bytes:
            raise DesignerError("reference audio is empty")

        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "reference.wav"
        reference_path.write_bytes(wav_bytes)
        reference_rel = self._rel(reference_path)

        profile = VoiceProfile(
            name=name,
            engine="qwen3_tts",
            model=CLONE_MODEL_ID,
            language=(language or "es").lower(),
            speaker="",
            style_instruction="",
            dtype="bfloat16",
            fallback_engine="kokoro",
            provenance={"source": "voice_designer", "mode": "reference_clone", "design": name},
        )
        design = {
            "name": name,
            "mode": "reference_clone",
            "language": profile.language,
            "instruction": "",
            "reference": {
                "audio": reference_rel,
                "text": transcript,
                "authorization": {"type": "external", "authorized": True, "statement": statement},
            },
        }
        atomic_write_json(self.profiles_dir / f"{name}.json", profile.public_dict())
        save_design(name, design, self.designs_dir)
        self._write_manifest(
            name,
            {
                "profile": name, "mode": "reference_clone", "language": profile.language,
                "instruction": "", "preview_text": transcript, "variants": [],
                "frozen": {"profile": name, "reference": reference_rel},
            },
        )
        return {"profile": profile.public_dict(), "design": design}
