"""Fifi custom wake-word training toolkit (Phase 3D.2) — isolated package.

Importable, dependency-light core for training, evaluating, calibrating, and
installing the custom "Fifi" openWakeWord model. Heavy dependencies (torch,
openWakeWord, audiomentations, librosa) are imported LAZILY inside the functions
that need them, so the pure logic — config validation, threshold selection,
metric math, metadata + hashing, atomic install/rollback — is testable in the
main project's .venv without the isolated training stack installed.

Nothing here runs a microphone, executes a command, or touches the main app's
safety layer. Training output is only ever *installed* through install.py, which
validates ONNX compatibility, backs up the previous model, and refuses to
replace the active model unless validation passes.
"""

__all__ = [
    "config",
    "paths",
    "datasets",
    "augment",
    "pipeline",
    "evaluate",
    "threshold",
    "install",
    "metadata",
    "onnx_validate",
    "verifier",
]
