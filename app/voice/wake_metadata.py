"""Read + verify the installed wake model's metadata at runtime (Phase 3D.2).

The trainer (wake_training/) WRITES ``models/wake_words/fifi.metadata.json`` when
it installs a model; the runtime only ever READS it here to answer wake-doctor /
wake-status: what phrase, which training version, which threshold, and does the
model on disk still match the metadata's hash. Pure stdlib + optional onnxruntime
(for an I/O sanity check) — the runtime never imports the isolated trainer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# openWakeWord feeds a 16-frame x 96-feature window to the classifier.
EXPECTED_FEATURE_DIMS = (16, 96)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_installed(model_path: str | Path, metadata_path: str | Path) -> dict:
    """Cross-check the on-disk model against its metadata hash.

    Returns ``{ok, problems, metadata}``. Missing metadata is a problem (the
    provenance is unknown), and a hash mismatch means the model on disk is not
    the one the metadata describes.
    """
    model_path = Path(model_path)
    problems: list[str] = []
    metadata = read_metadata(metadata_path)
    if metadata is None:
        return {"ok": False, "problems": ["metadata missing or unreadable"], "metadata": None}
    if not model_path.is_file():
        problems.append(f"model file missing: {model_path.name}")
    else:
        expected = (metadata.get("model") or {}).get("sha256", "")
        if expected:
            actual = sha256_file(model_path)
            if actual != expected:
                problems.append(
                    f"model hash mismatch: disk {actual[:12]}… != metadata {expected[:12]}…"
                )
        else:
            problems.append("metadata has no model.sha256 to verify against")
    return {"ok": not problems, "problems": problems, "metadata": metadata}


def validate_onnx_io(model_path: str | Path) -> dict:
    """Best-effort runtime I/O sanity check via onnxruntime (lazy import).

    Returns ``{ok, problems, checked}``. If onnxruntime is unavailable the check
    is skipped (``checked=False``) rather than failing — the model still loads
    through openWakeWord at listen time.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        return {"ok": False, "problems": [f"model file missing: {model_path}"], "checked": True}
    try:
        import onnxruntime
    except Exception:
        return {"ok": True, "problems": [], "checked": False}
    try:
        session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        return {"ok": False, "problems": [f"onnxruntime could not load the model: {exc}"],
                "checked": True}
    problems: list[str] = []
    inputs = session.get_inputs()
    if len(inputs) != 1:
        problems.append(f"expected 1 input, found {len(inputs)}")
    if not session.get_outputs():
        problems.append("model has no outputs")
    if len(inputs) == 1:
        shape = list(inputs[0].shape or [])
        trailing = shape[-2:]
        for got, expected in zip(trailing, EXPECTED_FEATURE_DIMS):
            if isinstance(got, int) and got > 0 and got != expected:
                problems.append(
                    f"input trailing dims {trailing} != expected {list(EXPECTED_FEATURE_DIMS)}"
                )
                break
    return {"ok": not problems, "problems": problems, "checked": True}


def describe_installed(model_path: str | Path, metadata_path: str | Path) -> dict:
    """Compact runtime summary for wake-status (phrase/version/threshold/hash)."""
    model_path = Path(model_path)
    present = model_path.is_file()
    info = {
        "model_present": present,
        "model_path": str(model_path),
        "size_bytes": model_path.stat().st_size if present else 0,
        "hash": sha256_file(model_path)[:16] if present else "",
        "phrase": None,
        "training_version": None,
        "threshold": None,
        "metadata_present": False,
        "metadata_ok": False,
    }
    metadata = read_metadata(metadata_path)
    if metadata:
        info["metadata_present"] = True
        info["phrase"] = metadata.get("phrase")
        info["training_version"] = metadata.get("training_version")
        info["threshold"] = metadata.get("threshold")
        verification = verify_installed(model_path, metadata_path)
        info["metadata_ok"] = verification["ok"]
        info["metadata_problems"] = verification["problems"]
    return info
