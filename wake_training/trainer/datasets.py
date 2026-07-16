"""Dataset discovery + manifests: positives, negatives, validation, hard negatives.

Pure filesystem accounting (counts, durations, presence) plus the hard-negative
phrase list the trainer feeds to openWakeWord as ``custom_negative_phrases``.
Duration reading is best-effort via soundfile (available in the main venv, so
these functions are testable there); an unreadable file is skipped, never fatal.
"""

from __future__ import annotations

from pathlib import Path

from . import paths
from .config import TrainingConfig

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def audio_files(directory: str | Path) -> list[Path]:
    """Every audio file under ``directory`` (recursive), sorted. [] if missing."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    return sorted(found)


def count_audio(directory: str | Path) -> int:
    return len(audio_files(directory))


def total_duration_seconds(directory: str | Path) -> float:
    """Sum of clip durations (best-effort). Unreadable files are skipped."""
    try:
        import soundfile as sf
    except Exception:
        return 0.0
    total = 0.0
    for path in audio_files(directory):
        try:
            info = sf.info(str(path))
            if info.samplerate:
                total += info.frames / float(info.samplerate)
        except Exception:
            continue
    return round(total, 2)


def directory_summary(directory: str | Path, *, with_duration: bool = True) -> dict:
    directory = Path(directory)
    count = count_audio(directory)
    summary = {
        "path": str(directory),
        "exists": directory.is_dir(),
        "count": count,
    }
    if with_duration and count:
        summary["duration_seconds"] = total_duration_seconds(directory)
        summary["duration_hours"] = round(summary["duration_seconds"] / 3600.0, 3)
    return summary


def resolve_dataset_path(rel: str | Path, root: Path | None = None) -> Path:
    """Resolve a workspace-relative dataset path (absolute paths pass through)."""
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    return (root or paths.WAKE_TRAINING_ROOT) / rel


def hard_negatives(config: TrainingConfig) -> list[str]:
    """The confusable phrases openWakeWord will synthesize as negatives."""
    return list(config.hard_negative_phrases)


def write_hard_negative_manifest(config: TrainingConfig, path: str | Path) -> Path:
    """Write the hard-negative phrases one-per-line (provenance / inspection)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(hard_negatives(config)) + "\n", encoding="utf-8")
    return path


def user_recording_summary(root: Path | None = None) -> dict:
    """Optional authorized recordings (record_samples.py). Training works without."""
    directory = (root or paths.WAKE_TRAINING_ROOT) / "datasets" / "positive" / "user"
    return directory_summary(directory)


def dataset_status(config: TrainingConfig, root: Path | None = None) -> dict:
    """Everything the CLI ``status`` needs about the on-disk datasets.

    Reports what exists and its size, plus whether the negatives/validation the
    config references are actually present — so a training run does not fail
    halfway for want of background audio.
    """
    root = root or paths.WAKE_TRAINING_ROOT
    backgrounds = [
        directory_summary(resolve_dataset_path(p, root), with_duration=True)
        for p in config.negatives.background_paths
    ]
    rirs = [
        directory_summary(resolve_dataset_path(p, root), with_duration=False)
        for p in config.negatives.rir_paths
    ]
    fp_data = resolve_dataset_path(
        config.negatives.false_positive_validation_data, root
    ) if config.negatives.false_positive_validation_data else None

    background_hours = round(
        sum(b.get("duration_hours", 0.0) for b in backgrounds), 3
    )
    return {
        "hard_negative_phrases": hard_negatives(config),
        "user_recordings": user_recording_summary(root),
        "backgrounds": backgrounds,
        "background_hours": background_hours,
        "rirs": rirs,
        "false_positive_validation_data": {
            "path": str(fp_data) if fp_data else "",
            "present": bool(fp_data and fp_data.is_file()),
        },
        "ready_to_train": bool(background_hours > 0),
    }
