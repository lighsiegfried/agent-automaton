"""Canonical filesystem locations for the wake-word training workspace.

Everything is derived from this file's own location, so the trainer resolves the
same paths no matter the current working directory (the CLI and tests both rely
on this). Install targets point back into the MAIN project so a trained model
lands exactly where the runtime reads it.
"""

from pathlib import Path

# wake_training/trainer/paths.py -> wake_training/ -> repo root
WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WAKE_TRAINING_ROOT.parent

# --- workspace (all gitignored except code/config) --------------------------------
CONFIG_DIR = WAKE_TRAINING_ROOT / "config"
DATASETS_DIR = WAKE_TRAINING_ROOT / "datasets"
POSITIVE_DIR = DATASETS_DIR / "positive"
NEGATIVE_DIR = DATASETS_DIR / "negative"
VALIDATION_DIR = DATASETS_DIR / "validation"
OUTPUTS_DIR = WAKE_TRAINING_ROOT / "outputs"
REPORTS_DIR = WAKE_TRAINING_ROOT / "reports"
SCRIPTS_DIR = WAKE_TRAINING_ROOT / "scripts"
VENV_DIR = WAKE_TRAINING_ROOT / ".venv"
# Third-party synthetic-speech generator, cloned here by setup.py.
PIPER_DIR = WAKE_TRAINING_ROOT / "piper-sample-generator"
# Authorized user's own recordings (optional; local + gitignored).
USER_RECORDINGS_DIR = POSITIVE_DIR / "user"

# --- install targets in the MAIN project (where the runtime reads the model) ------
WAKE_MODELS_DIR = PROJECT_ROOT / "models" / "wake_words"
INSTALL_TARGET = WAKE_MODELS_DIR / "fifi.onnx"
METADATA_TARGET = WAKE_MODELS_DIR / "fifi.metadata.json"
BACKUPS_DIR = WAKE_MODELS_DIR / "backups"


def candidate_output_dir(name: str) -> Path:
    """Where a candidate's training artifacts live: outputs/<name>/."""
    return OUTPUTS_DIR / name


def candidate_model_path(name: str) -> Path:
    """The exported ONNX for a candidate: outputs/<name>/<name>.onnx.

    openWakeWord's auto-training writes ``<output_dir>/<model_name>.onnx``; with
    output_dir == outputs/<name> and model_name == <name> that is this path.
    """
    return candidate_output_dir(name) / f"{name}.onnx"


def candidate_report_path(name: str, kind: str) -> Path:
    """reports/<name>-<kind>.json (kind: 'train' | 'eval' | 'compare' ...)."""
    return REPORTS_DIR / f"{name}-{kind}.json"


def ensure_workspace_dirs() -> None:
    """Create the local (gitignored) working directories if missing."""
    for path in (
        POSITIVE_DIR, NEGATIVE_DIR, VALIDATION_DIR, OUTPUTS_DIR, REPORTS_DIR,
        USER_RECORDINGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
