"""Unit tests for app/voice/wake_word.py — the VAD-gated wake-word service.

All wake dependencies are mocked (fake openwakeword / silero_vad modules): no
model download, microphone, or GPU is needed. These tests pin the contract:
- missing dependencies or a missing custom model => "unavailable";
- the custom fifi.onnx is NEVER downloaded or replaced automatically;
- only the explicit custom model is loaded (onnx framework, exact path);
- detection is VAD-gated: a high wake score without speech never triggers.
"""

import sys
import types
from pathlib import Path

import pytest

from app.voice import wake_word
from app.voice.wake_word import WakeWordService


@pytest.fixture
def fake_wake_modules(monkeypatch):
    """Install fake openwakeword / onnxruntime / silero_vad modules. Any attempt
    to download models raises immediately."""
    created = {"models": [], "downloads": 0}

    class FakeModel:
        def __init__(self, wakeword_models=None, inference_framework=None):
            created["models"].append(
                {
                    "wakeword_models": wakeword_models,
                    "inference_framework": inference_framework,
                }
            )
            self._score = 0.0

        def predict(self, frame):
            return {"fifi": self._score}

        def reset(self):
            pass

    def _forbidden_download(*args, **kwargs):
        created["downloads"] += 1
        raise AssertionError("wake service must NEVER download models")

    openwakeword = types.ModuleType("openwakeword")
    openwakeword_model = types.ModuleType("openwakeword.model")
    openwakeword_model.Model = FakeModel
    openwakeword_utils = types.ModuleType("openwakeword.utils")
    openwakeword_utils.download_models = _forbidden_download
    openwakeword.model = openwakeword_model
    openwakeword.utils = openwakeword_utils

    silero = types.ModuleType("silero_vad")
    silero.load_silero_vad = lambda onnx=False: (lambda *a, **k: None)

    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", openwakeword_model)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", openwakeword_utils)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    return created


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fifi.onnx"
    path.write_bytes(b"onnx-model-bytes")
    return path


# --- availability -------------------------------------------------------------------


def test_unavailable_when_dependencies_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_word, "check_wake_deps", lambda: (False, ["openwakeword"]))
    service = WakeWordService(model_path=_model_file(tmp_path))
    result = service.load()
    assert result["status"] == "unavailable"
    assert "requirements-wakeword.txt" in result["message"]
    assert not service.loaded


def test_unavailable_when_custom_model_missing(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=tmp_path / "fifi.onnx")  # does not exist
    result = service.load()
    assert result["status"] == "unavailable"
    assert "never" in result["message"].lower()  # never downloaded/replaced
    assert str(tmp_path / "fifi.onnx") in result["message"]
    assert fake_wake_modules["models"] == []  # nothing was loaded
    assert fake_wake_modules["downloads"] == 0  # and nothing was fetched


def test_load_uses_only_the_explicit_custom_model(fake_wake_modules, tmp_path):
    model_path = _model_file(tmp_path)
    service = WakeWordService(model_path=model_path)
    assert service.load() == {"status": "ok"}
    assert service.loaded
    assert fake_wake_modules["models"] == [
        {"wakeword_models": [str(model_path)], "inference_framework": "onnx"}
    ]
    assert fake_wake_modules["downloads"] == 0


def test_status_reports_everything(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=_model_file(tmp_path), threshold=0.6)
    status = service.status()
    assert status["dependencies_ok"] is True
    assert status["model_present"] is True
    assert status["available"] is True
    assert status["loaded"] is False
    assert status["threshold"] == 0.6


def test_status_when_model_absent(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=tmp_path / "missing.onnx")
    status = service.status()
    assert status["model_present"] is False
    assert status["available"] is False


# --- VAD gating ----------------------------------------------------------------------


def test_high_score_without_speech_never_triggers(fake_wake_modules, tmp_path):
    """False-positive reduction: the VAD gate vetoes non-speech triggers."""
    service = WakeWordService(model_path=_model_file(tmp_path), threshold=0.55)
    assert service.load() == {"status": "ok"}
    service._oww._score = 0.99  # loud TV noise scores high...
    service.is_speech = lambda frame: False  # ...but VAD says it isn't speech
    assert service.detect([0] * 1280) is False


def test_high_score_with_speech_triggers(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=_model_file(tmp_path), threshold=0.55)
    assert service.load() == {"status": "ok"}
    service._oww._score = 0.7
    service.is_speech = lambda frame: True
    assert service.detect([0] * 1280) is True


def test_below_threshold_never_triggers(fake_wake_modules, tmp_path):
    service = WakeWordService(model_path=_model_file(tmp_path), threshold=0.55)
    assert service.load() == {"status": "ok"}
    service._oww._score = 0.54
    service.is_speech = lambda frame: True
    assert service.detect([0] * 1280) is False


def test_default_configuration_matches_settings():
    service = WakeWordService()
    assert service.model_path.name == "fifi.onnx"
    assert service.threshold == 0.55
    assert service.vad_threshold == 0.5
