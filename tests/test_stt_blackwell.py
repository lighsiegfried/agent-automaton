"""Phase 3C.1 — Blackwell-safe STT resolution, CPU fallback, and preflight.

CUDA / faster-whisper are fully mocked: no GPU, driver, or model is required.
"""

import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.voice import stt as stt_module
from app.voice.stt import GpuInfo, SpeechToTextService, get_stt_service, resolve_stt_config


def _wav_bytes(seconds=0.1, rate=16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


BLACKWELL = GpuInfo(name="NVIDIA GeForce RTX 5060 Ti", compute_capability=12.0)
AMPERE = GpuInfo(name="NVIDIA GeForce RTX 3090", compute_capability=8.6)
NO_GPU = GpuInfo(name=None, compute_capability=None)


@pytest.fixture(autouse=True)
def _no_real_gpu_probe(monkeypatch):
    """Never shell out to nvidia-smi or CTranslate2 in unit tests."""
    monkeypatch.setattr(stt_module, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(stt_module, "ctranslate2_version", lambda: "4.8.1")
    monkeypatch.setattr(stt_module, "detect_gpu_info", lambda: BLACKWELL)
    get_stt_service.cache_clear()
    yield
    get_stt_service.cache_clear()


# --- GPU detection ---------------------------------------------------------------


def test_blackwell_detected_by_capability_and_name():
    assert BLACKWELL.is_blackwell is True
    assert GpuInfo("Some Blackwell card", None).is_blackwell is True
    assert GpuInfo("NVIDIA GeForce RTX 5090", None).is_blackwell is True
    assert AMPERE.is_blackwell is False
    assert NO_GPU.is_blackwell is False


# --- resolution ------------------------------------------------------------------


def test_blackwell_auto_resolves_to_float16():
    device, compute = resolve_stt_config("auto", "auto", BLACKWELL, allow_cpu_fallback=False)
    assert device == "cuda"
    assert compute == "float16"


def test_blackwell_never_uses_int8():
    for requested in ("int8", "int8_float16", "int8_float32"):
        device, compute = resolve_stt_config("cuda", requested, BLACKWELL, allow_cpu_fallback=False)
        assert device == "cuda"
        assert "int8" not in compute
        assert compute == "float16"


def test_non_blackwell_int8_is_left_alone():
    device, compute = resolve_stt_config("cuda", "int8_float16", AMPERE, allow_cpu_fallback=False)
    assert device == "cuda"
    assert compute == "int8_float16"  # only Blackwell forces float16


def test_cpu_device_uses_int8():
    device, compute = resolve_stt_config("cpu", "auto", NO_GPU, allow_cpu_fallback=True)
    assert device == "cpu"
    assert compute == "int8"
    # float16 is a GPU format; on CPU it is coerced to int8
    _, compute2 = resolve_stt_config("cpu", "float16", NO_GPU, allow_cpu_fallback=True)
    assert compute2 == "int8"


def test_auto_device_without_gpu_falls_to_cpu(monkeypatch):
    monkeypatch.setattr(stt_module, "cuda_device_count", lambda: 0)
    device, compute = resolve_stt_config("auto", "auto", NO_GPU, allow_cpu_fallback=False)
    assert device == "cpu"
    assert compute == "int8"


# --- service construction resolves safely ---------------------------------------


def test_service_resolves_blackwell_to_float16(settings):
    settings.stt_device = "auto"
    settings.stt_compute_type = "auto"
    service = SpeechToTextService()
    assert service.device == "cuda"
    assert service.compute_type == "float16"
    assert service.describe()["blackwell"] is True
    assert service.describe()["ctranslate2_version"] == "4.8.1"


# --- CPU fallback (explicit only) ------------------------------------------------


def test_cpu_fallback_only_when_enabled(settings, monkeypatch):
    settings.stt_device = "cuda"
    settings.stt_compute_type = "float16"
    settings.stt_allow_cpu_fallback = True

    calls = []

    class FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)
    service = SpeechToTextService()
    service._load()
    assert service.cpu_fallback_active is True
    assert service.device == "cpu"
    assert service.compute_type == "int8"
    assert ("cuda", "float16") in calls and ("cpu", "int8") in calls


def test_no_cpu_fallback_when_disabled(settings, monkeypatch):
    settings.stt_device = "cuda"
    settings.stt_compute_type = "float16"
    settings.stt_allow_cpu_fallback = False

    class FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None):
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)
    service = SpeechToTextService()
    with pytest.raises(RuntimeError):
        service._load()
    assert service.cpu_fallback_active is False


# --- preflight -------------------------------------------------------------------


def test_preflight_passes_without_running_any_tool(settings, monkeypatch):
    settings.stt_device = "cuda"
    settings.stt_compute_type = "float16"
    transcribed = []

    class FakeWhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, language=None, **kwargs):
            transcribed.append(path)
            return iter([]), object()

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)
    report = SpeechToTextService().preflight()
    assert report["passed"] is True
    assert report["device"] == "cuda"
    assert report["compute_type"] == "float16"
    assert report["gpu"] == "NVIDIA GeForce RTX 5060 Ti"
    assert transcribed, "preflight must actually exercise the model (silence probe)"


def test_preflight_reports_failure_structurally(settings, monkeypatch):
    class FakeWhisperModel:
        def __init__(self, *a, **k):
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)
    report = SpeechToTextService().preflight()
    assert report["passed"] is False
    assert report["status"] == "error"
    assert "CUBLAS_STATUS_NOT_SUPPORTED" in report["message"]


# --- API survives an STT failure -------------------------------------------------


def test_stt_exception_leaves_health_available(settings, monkeypatch):
    settings.enable_voice = True
    get_stt_service.cache_clear()

    class ExplodingModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, language=None, **kwargs):
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(stt_module, "WhisperModel", ExplodingModel)

    with TestClient(app) as client:
        voice = client.post(
            "/voice/command", files={"file": ("clip.wav", _wav_bytes(), "audio/wav")}
        ).json()
        # A CUDA failure returns a structured error, NOT a fake successful command.
        assert voice.get("status") in ("error", "unavailable")
        assert "command" not in voice

        # /health and /identity stay reachable after the STT failure.
        assert client.get("/health").status_code == 200
        assert client.get("/identity").json()["project_name"] == "agent-automaton"

    get_stt_service.cache_clear()
