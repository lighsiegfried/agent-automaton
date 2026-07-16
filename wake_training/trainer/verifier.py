"""Optional speaker verifier (openWakeWord custom verifier) — disabled by default.

A verifier is a small logistic model trained on the openWakeWord embeddings of
the AUTHORIZED user's own recordings; at runtime it only makes wake detection
stricter (a second gate that rejects other voices). It is entirely optional:
wake-word detection works fully without it, and nothing here trains one unless
``verifier.enabled: true`` is set in the candidate config AND the user has
provided enough of their own samples (record_samples.py).

The heavy openWakeWord training call is lazy + injectable, so the gating logic is
testable without the training stack.
"""

from __future__ import annotations

from pathlib import Path

from . import datasets, paths
from .config import TrainingConfig

# Minimum authorized recordings before a verifier can be trained meaningfully.
MIN_REFERENCE_CLIPS = 3


def verifier_output_path(candidate: str) -> Path:
    return paths.candidate_output_dir(candidate) / f"{candidate}_verifier.pkl"


def reference_clips(config: TrainingConfig, root: Path | None = None) -> list[Path]:
    """The authorized user's recordings that a verifier would be trained on."""
    directory = datasets.resolve_dataset_path(config.verifier.reference_dir, root)
    return datasets.audio_files(directory)


def verifier_status(config: TrainingConfig, root: Path | None = None) -> dict:
    clips = reference_clips(config, root)
    output = verifier_output_path(config.model_name)
    return {
        "enabled": bool(config.verifier.enabled),
        "threshold": config.verifier.threshold,
        "reference_dir": config.verifier.reference_dir,
        "reference_count": len(clips),
        "min_required": MIN_REFERENCE_CLIPS,
        "trainable": bool(config.verifier.enabled and len(clips) >= MIN_REFERENCE_CLIPS),
        "output_path": str(output),
        "present": output.is_file(),
    }


def _default_trainer():
    """Lazily import openWakeWord's custom-verifier trainer (isolated venv)."""
    from openwakeword.train import train_custom_verifier

    return train_custom_verifier


def train_verifier(
    config: TrainingConfig,
    candidate: str,
    *,
    negative_clips: list[str | Path] | None = None,
    trainer_fn=None,
    root: Path | None = None,
) -> dict:
    """Train the optional verifier. Refuses cleanly when disabled or under-provisioned.

    Returns ``{"status": "disabled"|"error"|"ok", ...}``. ``trainer_fn`` is
    injectable for tests; it receives
    ``(positive_reference_clips, negative_reference_clips, output_path, model_name)``.
    """
    if not config.verifier.enabled:
        return {
            "status": "disabled",
            "message": "verifier.enabled is false — wake detection works without it",
        }

    positive = [str(p) for p in reference_clips(config, root)]
    if len(positive) < MIN_REFERENCE_CLIPS:
        return {
            "status": "error",
            "message": (
                f"need >= {MIN_REFERENCE_CLIPS} authorized recordings in "
                f"{config.verifier.reference_dir} (found {len(positive)}); "
                "record more with wake_training/scripts/record_samples.py"
            ),
        }

    if negative_clips is None:
        # Fall back to the general negative-speech corpus (other voices).
        neg_dir = datasets.resolve_dataset_path("datasets/negative/speech", root)
        negative_clips = [str(p) for p in datasets.audio_files(neg_dir)]
    negative = [str(p) for p in negative_clips]

    output = verifier_output_path(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer_fn = trainer_fn or _default_trainer()
    trainer_fn(positive, negative, str(output), candidate)
    return {
        "status": "ok",
        "output_path": str(output),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "threshold": config.verifier.threshold,
        "present": output.is_file(),
    }
