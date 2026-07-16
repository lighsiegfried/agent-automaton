"""Optional speaker-verifier load path in WakeWordService (Phase 3D.2).

The verifier is DISABLED by default: the bare model load must pass exactly
``wakeword_models`` + ``inference_framework`` (the existing contract). Only when
it is explicitly enabled AND the verifier file exists are the custom-verifier
kwargs added. No real openWakeWord / model download here.
"""

import sys
import types
from pathlib import Path

import pytest

from app.voice.wake_word import WakeWordService


@pytest.fixture
def fake_wake_modules(monkeypatch):
    created = {"models": []}

    class FakeModel:
        def __init__(
            self,
            wakeword_models=None,
            inference_framework=None,
            custom_verifier_models=None,
            custom_verifier_threshold=None,
        ):
            created["models"].append({
                "wakeword_models": wakeword_models,
                "inference_framework": inference_framework,
                "custom_verifier_models": custom_verifier_models,
                "custom_verifier_threshold": custom_verifier_threshold,
            })

        def predict(self, frame):
            return {"fifi": 0.0}

        def reset(self):
            pass

    oww = types.ModuleType("openwakeword")
    oww_model = types.ModuleType("openwakeword.model")
    oww_model.Model = FakeModel
    oww.model = oww_model
    silero = types.ModuleType("silero_vad")
    silero.load_silero_vad = lambda onnx=False: (lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "openwakeword", oww)
    monkeypatch.setitem(sys.modules, "openwakeword.model", oww_model)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    return created


def _model_file(tmp_path: Path, name="fifi.onnx") -> Path:
    path = tmp_path / name
    path.write_bytes(b"onnx-bytes")
    return path


def test_verifier_disabled_by_default_bare_kwargs(fake_wake_modules, tmp_path):
    """Contract preserved: with the verifier off, kwargs are exactly the two."""
    service = WakeWordService(model_path=_model_file(tmp_path))
    assert service.verifier_active is False
    assert service.load() == {"status": "ok"}
    call = fake_wake_modules["models"][0]
    assert call["wakeword_models"] == [str(tmp_path / "fifi.onnx")]
    assert call["inference_framework"] == "onnx"
    assert call["custom_verifier_models"] is None
    assert call["custom_verifier_threshold"] is None


def test_enabled_but_missing_file_stays_inactive(fake_wake_modules, tmp_path):
    service = WakeWordService(
        model_path=_model_file(tmp_path),
        verifier_enabled=True,
        verifier_path=tmp_path / "missing_verifier.pkl",  # does not exist
    )
    assert service.verifier_active is False
    service.load()
    assert fake_wake_modules["models"][0]["custom_verifier_models"] is None


def test_enabled_with_file_adds_verifier_gate(fake_wake_modules, tmp_path):
    verifier_pkl = tmp_path / "fifi_verifier.pkl"
    verifier_pkl.write_bytes(b"pickle")
    service = WakeWordService(
        model_path=_model_file(tmp_path),
        verifier_enabled=True,
        verifier_path=verifier_pkl,
        verifier_threshold=0.42,
    )
    assert service.verifier_active is True
    service.load()
    call = fake_wake_modules["models"][0]
    assert call["custom_verifier_models"] == {"fifi": str(verifier_pkl)}
    assert call["custom_verifier_threshold"] == 0.42


def test_status_reports_verifier(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=_model_file(tmp_path))
    status = service.status()
    assert status["verifier_enabled"] is False
    assert status["verifier_active"] is False
    assert "verifier_path" in status
