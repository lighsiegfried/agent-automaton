"""Metadata build/validate/hash + on-disk verification."""

import hashlib

from trainer import metadata as meta


def _model(tmp_path, data=b"model-bytes"):
    path = tmp_path / "fifi.onnx"
    path.write_bytes(data)
    return path


def test_sha256_file_matches_hashlib(tmp_path):
    path = _model(tmp_path, b"abc123")
    assert meta.sha256_file(path) == hashlib.sha256(b"abc123").hexdigest()


def test_build_metadata_has_required_fields(tmp_path):
    path = _model(tmp_path)
    doc = meta.build_metadata(
        model_path=path, candidate="fifi", phrase="Fifi", threshold=0.5123,
        metrics={"recall": 0.9}, provenance={"seed": 1234},
    )
    assert doc["training_version"] == meta.TRAINING_VERSION
    assert doc["phrase"] == "Fifi"
    assert doc["threshold"] == 0.5123
    assert doc["model"]["sha256"] == meta.sha256_file(path)
    assert doc["model"]["size_bytes"] == len(b"model-bytes")
    assert doc["verifier"] == {"enabled": False}
    assert doc["provenance"]["seed"] == 1234
    assert doc["metrics"]["recall"] == 0.9
    assert meta.validate_metadata(doc) == []


def test_validate_metadata_catches_problems():
    problems = meta.validate_metadata(
        {"schema_version": 1, "training_version": "x", "threshold": "nan",
         "phrase": "", "model": {"sha256": "short"}, "provenance": {}}
    )
    assert any("sha256" in p for p in problems)
    assert any("phrase" in p for p in problems)
    assert any("threshold" in p for p in problems)


def test_write_read_roundtrip_preserves_accents(tmp_path):
    doc = meta.build_metadata(
        model_path=_model(tmp_path), candidate="oye_fifi",
        phrase="Oye Fifí", threshold=0.6,
    )
    out = meta.write_metadata(tmp_path / "fifi.metadata.json", doc)
    loaded = meta.read_metadata(out)
    assert loaded["phrase"] == "Oye Fifí"  # Unicode preserved


def test_read_metadata_missing_returns_none(tmp_path):
    assert meta.read_metadata(tmp_path / "none.json") is None


def test_verify_installed_matching_hash(tmp_path):
    model = _model(tmp_path)
    doc = meta.build_metadata(model_path=model, candidate="fifi", phrase="Fifi", threshold=0.5)
    meta_path = meta.write_metadata(tmp_path / "fifi.metadata.json", doc)
    result = meta.verify_installed(model, meta_path)
    assert result["ok"] is True
    assert result["problems"] == []


def test_verify_installed_detects_tamper(tmp_path):
    model = _model(tmp_path)
    doc = meta.build_metadata(model_path=model, candidate="fifi", phrase="Fifi", threshold=0.5)
    meta_path = meta.write_metadata(tmp_path / "fifi.metadata.json", doc)
    model.write_bytes(b"different-bytes-now")  # model changed after metadata written
    result = meta.verify_installed(model, meta_path)
    assert result["ok"] is False
    assert any("hash mismatch" in p for p in result["problems"])


def test_verify_installed_missing_metadata(tmp_path):
    result = meta.verify_installed(_model(tmp_path), tmp_path / "absent.json")
    assert result["ok"] is False
    assert any("metadata missing" in p for p in result["problems"])
