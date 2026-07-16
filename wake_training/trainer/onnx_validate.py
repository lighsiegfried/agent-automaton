"""Validate that an exported ONNX model is runtime-compatible before install.

openWakeWord's runtime feeds a 16-frame x 96-feature window to the classifier
and reads back a single wake probability, so a compatible model has exactly one
input whose trailing dims are (16, 96) and at least one output. We load the graph
with onnxruntime (available in the main venv) purely to introspect the I/O — the
``session_factory`` is injectable so this is testable without a real .onnx file.

Concrete dimension mismatches are hard failures (the runtime would break);
dynamic/symbolic dims are accepted.
"""

from __future__ import annotations

from pathlib import Path

# openWakeWord feature-window shape the runtime feeds the classifier.
EXPECTED_FEATURE_DIMS = (16, 96)


def _default_session_factory(path: str):
    import onnxruntime  # available in the main venv

    return onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def io_summary(session) -> dict:
    """Serialise a session's inputs/outputs to plain dicts."""
    def describe(items):
        return [
            {"name": getattr(i, "name", "?"),
             "shape": list(getattr(i, "shape", []) or []),
             "type": getattr(i, "type", "")}
            for i in items
        ]

    return {
        "inputs": describe(session.get_inputs()),
        "outputs": describe(session.get_outputs()),
    }


def _dim_mismatch(shape: list) -> str | None:
    """Return a problem string if concrete trailing dims != (16, 96)."""
    if len(shape) < 2:
        return f"input rank {len(shape)} is too small (expected 3: [batch, 16, 96])"
    trailing = shape[-2:]
    for got, expected in zip(trailing, EXPECTED_FEATURE_DIMS):
        if isinstance(got, int) and got > 0 and got != expected:
            return (
                f"input trailing dims {trailing} != expected {list(EXPECTED_FEATURE_DIMS)}"
            )
    return None


def validate_onnx(path: str | Path, session_factory=None) -> dict:
    """Return ``{ok, problems, summary}`` for an exported model.

    Hard failures: file missing/empty, graph won't load, not exactly one input,
    no outputs, or concrete input dims incompatible with the runtime window.
    """
    path = Path(path)
    problems: list[str] = []
    if not path.is_file():
        return {"ok": False, "problems": [f"model file not found: {path}"], "summary": {}}
    if path.stat().st_size == 0:
        return {"ok": False, "problems": [f"model file is empty: {path}"], "summary": {}}

    factory = session_factory or _default_session_factory
    try:
        session = factory(str(path))
    except Exception as exc:  # onnxruntime raises on malformed graphs
        return {
            "ok": False,
            "problems": [f"onnxruntime could not load the model: {exc}"],
            "summary": {},
        }

    summary = io_summary(session)
    inputs, outputs = summary["inputs"], summary["outputs"]
    if len(inputs) != 1:
        problems.append(f"expected exactly 1 input, found {len(inputs)}")
    if len(outputs) < 1:
        problems.append("model has no outputs")
    if len(inputs) == 1:
        mismatch = _dim_mismatch(inputs[0]["shape"])
        if mismatch:
            problems.append(mismatch)

    return {"ok": not problems, "problems": problems, "summary": summary}
