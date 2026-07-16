"""Optional speaker-verifier gating (disabled by default; injected trainer)."""

import numpy as np
import pytest
import soundfile as sf

from trainer import paths, verifier
from trainer.config import load_config


@pytest.fixture
def outputs_under_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    return tmp_path


def _wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(8000, dtype="float32"), 16000)
    return path


def _enabled_config(tmp_path, n_user=4):
    config = load_config("fifi")
    config.verifier.enabled = True
    config.verifier.reference_dir = "datasets/positive/user"
    for i in range(n_user):
        _wav(tmp_path / "datasets" / "positive" / "user" / f"me_{i}.wav")
    # a couple of "other voice" negatives
    for i in range(2):
        _wav(tmp_path / "datasets" / "negative" / "speech" / f"other_{i}.wav")
    return config


def test_status_disabled_by_default():
    status = verifier.verifier_status(load_config("fifi"))
    assert status["enabled"] is False
    assert status["present"] is False
    assert status["trainable"] is False


def test_train_verifier_refuses_when_disabled(outputs_under_tmp):
    result = verifier.train_verifier(load_config("fifi"), "fifi")
    assert result["status"] == "disabled"


def test_train_verifier_errors_with_too_few_samples(outputs_under_tmp):
    config = _enabled_config(outputs_under_tmp, n_user=1)  # below MIN_REFERENCE_CLIPS
    result = verifier.train_verifier(config, "fifi", root=outputs_under_tmp)
    assert result["status"] == "error"
    assert "authorized recordings" in result["message"]


def test_train_verifier_ok_with_injected_trainer(outputs_under_tmp):
    config = _enabled_config(outputs_under_tmp, n_user=4)
    calls = []

    def fake_trainer(positive, negative, output_path, model_name):
        calls.append((positive, negative, output_path, model_name))
        from pathlib import Path

        Path(output_path).write_bytes(b"verifier-pickle")

    result = verifier.train_verifier(
        config, "fifi", trainer_fn=fake_trainer, root=outputs_under_tmp
    )
    assert result["status"] == "ok"
    assert result["positive_count"] == 4
    assert result["negative_count"] == 2
    assert result["present"] is True
    # The trainer received positives, negatives, an output path, and the name.
    positive, negative, output_path, model_name = calls[0]
    assert len(positive) == 4 and len(negative) == 2
    assert model_name == "fifi"
    assert output_path.endswith("fifi_verifier.pkl")


def test_status_trainable_when_enabled_with_samples(outputs_under_tmp):
    config = _enabled_config(outputs_under_tmp, n_user=5)
    status = verifier.verifier_status(config, root=outputs_under_tmp)
    assert status["enabled"] is True
    assert status["reference_count"] == 5
    assert status["trainable"] is True
