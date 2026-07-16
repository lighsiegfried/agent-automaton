"""Runtime-side wake metadata read/verify/describe + ONNX I/O sanity."""

import json

import pytest

from app.voice import wake_metadata


def _model(tmp_path, data=b"onnx-bytes"):
    path = tmp_path / "fifi.onnx"
    path.write_bytes(data)
    return path


def _write_meta(tmp_path, model, phrase="Fifi", threshold=0.55):
    doc = {
        "schema_version": 1,
        "training_version": "3D.2",
        "phrase": phrase,
        "threshold": threshold,
        "model": {"filename": "fifi.onnx", "sha256": wake_metadata.sha256_file(model),
                  "size_bytes": model.stat().st_size},
        "provenance": {"tool": "wake_training"},
    }
    path = tmp_path / "fifi.metadata.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_verify_installed_ok(tmp_path):
    model = _model(tmp_path)
    meta = _write_meta(tmp_path, model)
    result = wake_metadata.verify_installed(model, meta)
    assert result["ok"] is True
    assert result["metadata"]["phrase"] == "Fifi"


def test_verify_installed_detects_tamper(tmp_path):
    model = _model(tmp_path)
    meta = _write_meta(tmp_path, model)
    model.write_bytes(b"changed")
    result = wake_metadata.verify_installed(model, meta)
    assert result["ok"] is False
    assert any("hash mismatch" in p for p in result["problems"])


def test_verify_installed_missing_metadata(tmp_path):
    result = wake_metadata.verify_installed(_model(tmp_path), tmp_path / "none.json")
    assert result["ok"] is False
    assert "metadata missing" in result["problems"][0]


def test_describe_installed_with_metadata(tmp_path):
    model = _model(tmp_path)
    meta = _write_meta(tmp_path, model, phrase="Oye Fifí", threshold=0.6)
    info = wake_metadata.describe_installed(model, meta)
    assert info["model_present"] is True
    assert info["phrase"] == "Oye Fifí"
    assert info["threshold"] == 0.6
    assert info["training_version"] == "3D.2"
    assert info["metadata_ok"] is True
    assert len(info["hash"]) == 16


def test_describe_installed_without_metadata(tmp_path):
    model = _model(tmp_path)
    info = wake_metadata.describe_installed(model, tmp_path / "absent.json")
    assert info["metadata_present"] is False
    assert info["hash"]  # still hashes the file


def test_validate_onnx_io_missing_file(tmp_path):
    result = wake_metadata.validate_onnx_io(tmp_path / "nope.onnx")
    assert result["ok"] is False
    assert result["checked"] is True


def test_validate_onnx_io_bad_bytes(tmp_path):
    # onnxruntime is an OPTIONAL, host-only wake dependency (requirements-wakeword.txt).
    # Skip when it isn't installed rather than reporting a false failure; when it IS
    # present this runs for real, so a genuine ONNX validation regression still fails.
    pytest.importorskip("onnxruntime")
    result = wake_metadata.validate_onnx_io(_model(tmp_path, b"not a real onnx graph"))
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("could not load" in p for p in result["problems"])
