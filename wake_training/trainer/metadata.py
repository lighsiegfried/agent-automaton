"""Provenance metadata for the installed wake model + content hashing.

``models/wake_words/fifi.metadata.json`` records exactly how the active model was
produced: the training version, the wake phrase, the calibrated threshold, the
evaluation metrics, the model's sha256 hash and size, and provenance (candidate,
config, dataset counts, seed, tool, timestamp). The runtime reads this back in
``wake-doctor`` / ``wake-status`` to prove the model on disk is the one described.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Bumped when the training method or metadata schema changes.
TRAINING_VERSION = "3D.2"
METADATA_SCHEMA_VERSION = 1

REQUIRED_KEYS = (
    "schema_version", "training_version", "phrase", "threshold", "model",
    "provenance",
)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file's bytes (hex)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_metadata(
    *,
    model_path: str | Path,
    candidate: str,
    phrase: str,
    threshold: float,
    metrics: dict | None = None,
    provenance: dict | None = None,
    verifier: dict | None = None,
    model_hash: str | None = None,
    created_utc: str | None = None,
) -> dict:
    """Assemble the metadata document for an installed model."""
    model_path = Path(model_path)
    size = model_path.stat().st_size if model_path.is_file() else 0
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "training_version": TRAINING_VERSION,
        "candidate": candidate,
        "phrase": phrase,
        "threshold": round(float(threshold), 4),
        "metrics": metrics or {},
        "model": {
            "filename": model_path.name,
            "sha256": model_hash or (sha256_file(model_path) if model_path.is_file() else ""),
            "size_bytes": size,
        },
        "verifier": verifier or {"enabled": False},
        "provenance": {
            "tool": "wake_training",
            "candidate": candidate,
            "created_utc": created_utc or _utc(),
            **(provenance or {}),
        },
    }


def validate_metadata(data: dict) -> list[str]:
    """Return problems ([] means structurally valid)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["metadata is not an object"]
    for key in REQUIRED_KEYS:
        if key not in data:
            problems.append(f"missing required key: {key}")
    model = data.get("model") or {}
    digest = model.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64:
        problems.append("model.sha256 must be a 64-char hex digest")
    if not isinstance(data.get("phrase", ""), str) or not data.get("phrase"):
        problems.append("phrase must be a non-empty string")
    import math

    try:
        value = float(data.get("threshold"))
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError):
        problems.append("threshold must be a finite number")
    return problems


def write_metadata(path: str | Path, data: dict) -> Path:
    """Atomically write the metadata JSON (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fifi-meta-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_metadata(path: str | Path) -> dict | None:
    """Read metadata JSON, or None if missing/corrupt (never raises)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_installed(model_path: str | Path, metadata_path: str | Path) -> dict:
    """Cross-check the on-disk model against its metadata hash.

    Returns ``{ok, problems, metadata}``. A hash mismatch means the installed
    model is NOT the one the metadata describes (tampered/stale) — the runtime
    surfaces this in wake-doctor.
    """
    model_path = Path(model_path)
    problems: list[str] = []
    metadata = read_metadata(metadata_path)
    if metadata is None:
        return {"ok": False, "problems": ["metadata missing or unreadable"], "metadata": None}
    problems.extend(validate_metadata(metadata))
    if not model_path.is_file():
        problems.append(f"model file missing: {model_path}")
    else:
        actual = sha256_file(model_path)
        expected = (metadata.get("model") or {}).get("sha256", "")
        if expected and actual != expected:
            problems.append(
                f"model hash mismatch: on disk {actual[:12]}… != metadata {expected[:12]}…"
            )
    return {"ok": not problems, "problems": problems, "metadata": metadata}
