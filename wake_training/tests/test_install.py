"""Atomic install / refusal / backup / rollback of the active wake model."""

from trainer import install
from trainer import metadata as meta

OK_VALIDATION = {"ok": True, "problems": [], "summary": {"inputs": [], "outputs": []}}
BAD_VALIDATION = {"ok": False, "problems": ["bad graph"], "summary": {}}


def _ok(_source):
    return OK_VALIDATION


def _bad(_source):
    return BAD_VALIDATION


def _targets(tmp_path):
    return {
        "target": tmp_path / "models" / "fifi.onnx",
        "metadata_target": tmp_path / "models" / "fifi.metadata.json",
        "backups_dir": tmp_path / "models" / "backups",
    }


def _source(tmp_path, data=b"candidate-v1"):
    path = tmp_path / "outputs" / "fifi.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_fresh_install_creates_model_and_metadata(tmp_path):
    t = _targets(tmp_path)
    result = install.install_model(
        _source(tmp_path), candidate="fifi", phrase="Fifi", threshold=0.55,
        metrics={"recall": 0.9}, validate_fn=_ok, **t,
    )
    assert result["status"] == "ok"
    assert result["replaced"] is False
    assert result["backup"] == {}
    target = t["target"]
    assert target.read_bytes() == b"candidate-v1"
    assert result["hash"] == meta.sha256_file(target)
    doc = meta.read_metadata(t["metadata_target"])
    assert doc["phrase"] == "Fifi"
    assert doc["model"]["sha256"] == result["hash"]


def test_refuses_to_replace_installed_model_without_force(tmp_path):
    t = _targets(tmp_path)
    install.install_model(_source(tmp_path, b"v1"), candidate="fifi", phrase="Fifi",
                          threshold=0.5, validate_fn=_ok, **t)
    before = t["target"].read_bytes()
    result = install.install_model(
        _source(tmp_path, b"v2"), candidate="fifi", phrase="Fifi", threshold=0.5,
        validate_fn=_ok, **t,
    )
    assert result["status"] == "refused"
    assert result["reason"] == "already_installed"
    assert t["target"].read_bytes() == before  # active model untouched


def test_force_replace_backs_up_then_installs(tmp_path):
    t = _targets(tmp_path)
    install.install_model(_source(tmp_path, b"v1"), candidate="fifi", phrase="Fifi",
                          threshold=0.5, validate_fn=_ok, stamp="20260101T000000Z", **t)
    result = install.install_model(
        _source(tmp_path, b"v2-new"), candidate="fifi", phrase="Fifi", threshold=0.6,
        validate_fn=_ok, force=True, stamp="20260102T000000Z", **t,
    )
    assert result["status"] == "ok"
    assert result["replaced"] is True
    assert t["target"].read_bytes() == b"v2-new"
    # A backup of v1 (model + metadata) was written.
    backups = install.list_backups(t["backups_dir"])
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"v1"
    assert (backups[0].parent / (backups[0].stem + ".metadata.json")).is_file()


def test_validation_failure_refuses_and_preserves_active(tmp_path):
    t = _targets(tmp_path)
    install.install_model(_source(tmp_path, b"good"), candidate="fifi", phrase="Fifi",
                          threshold=0.5, validate_fn=_ok, **t)
    result = install.install_model(
        _source(tmp_path, b"bad"), candidate="fifi", phrase="Fifi", threshold=0.5,
        validate_fn=_bad, force=True, **t,
    )
    assert result["status"] == "refused"
    assert result["reason"] == "validation_failed"
    assert t["target"].read_bytes() == b"good"  # never replaced by an invalid model


def test_missing_source_errors(tmp_path):
    t = _targets(tmp_path)
    result = install.install_model(
        tmp_path / "nope.onnx", candidate="fifi", phrase="Fifi", threshold=0.5,
        validate_fn=_ok, **t,
    )
    assert result["status"] == "error"
    assert "source model not found" in result["message"]


def test_rollback_restores_previous_model(tmp_path):
    t = _targets(tmp_path)
    install.install_model(_source(tmp_path, b"v1"), candidate="fifi", phrase="Fifi",
                          threshold=0.5, validate_fn=_ok, stamp="20260101T000000Z", **t)
    install.install_model(_source(tmp_path, b"v2"), candidate="fifi", phrase="Fifi",
                          threshold=0.6, validate_fn=_ok, force=True,
                          stamp="20260102T000000Z", **t)
    assert t["target"].read_bytes() == b"v2"
    result = install.rollback(target=t["target"], metadata_target=t["metadata_target"],
                              backups_dir=t["backups_dir"])
    assert result["status"] == "ok"
    assert t["target"].read_bytes() == b"v1"  # restored
    assert result["metadata_restored"] is True


def test_rollback_without_backups_errors(tmp_path):
    t = _targets(tmp_path)
    result = install.rollback(**t)
    assert result["status"] == "error"
    assert "no backups" in result["message"]
