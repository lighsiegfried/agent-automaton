"""Voice pipeline tests. faster-whisper and pyttsx3 are always mocked —
no microphone, GPU, or audio model required."""

import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.core.router import handle_command
from app.main import app
from app.schemas.commands import CommandRequest, ExecutionStatus
from app.voice import stt as stt_module
from app.voice import tts as tts_module
from app.voice.stt import SpeechToTextService, get_stt_service


def wav_bytes(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def upload(name: str = "clip.wav") -> dict:
    return {"file": (name, wav_bytes(), "audio/wav")}


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeInfo:
    def __init__(self, language: str, duration: float = 1.2) -> None:
        self.language = language
        self.duration = duration


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def voice_on(settings):
    settings.enable_voice = True
    get_stt_service.cache_clear()
    yield settings
    get_stt_service.cache_clear()


@pytest.fixture
def fake_whisper(monkeypatch):
    """Install a fake WhisperModel; returns a setter for the transcript."""

    def set_result(text: str, language: str = "en"):
        detected = language

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def transcribe(self, path, language=None, **kwargs):
                return [FakeSegment(text)], FakeInfo(detected)

        monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)

    return set_result


# --- disabled by default -------------------------------------------------------


def test_voice_endpoints_disabled_by_default(client, settings):
    assert settings.enable_voice is False
    for call in (
        lambda: client.post("/voice/transcribe", files=upload()),
        lambda: client.post("/voice/speak", json={"text": "hola"}),
        lambda: client.post("/voice/command", files=upload()),
    ):
        data = call().json()
        assert data["status"] == "disabled"
        assert "ENABLE_VOICE" in data["message"]


def test_health_reports_voice_flag(client):
    assert client.get("/health").json()["voice"] is False


# --- /voice/status (Phase 3D.1: STT warm reporting) ------------------------------


def test_voice_status_disabled_by_default(client, settings):
    assert settings.enable_voice is False
    assert client.get("/voice/status").json()["status"] == "disabled"


def test_voice_status_reports_warm_state_without_loading(client, voice_on, fake_whisper):
    fake_whisper("anything")
    response = client.get("/voice/status").json()
    assert response["status"] == "ok"
    assert response["loaded"] is False  # reporting must NOT load the model

    service = get_stt_service()
    service._model = object()  # pretend prewarm loaded it
    warm = client.get("/voice/status").json()
    assert warm["loaded"] is True
    assert "device" in warm and "compute_type" in warm and "model" in warm


def test_voice_status_never_breaks_health(client, voice_on, monkeypatch):
    """A broken STT stack must not affect /health (or /voice/status itself)."""
    monkeypatch.setattr(stt_module, "WhisperModel", None)
    status = client.get("/voice/status").json()
    assert status["available"] is False
    health = client.get("/health").json()
    assert health["status"] == "ok"


# --- missing dependency behavior -----------------------------------------------


def test_transcribe_reports_missing_dependency(client, voice_on, monkeypatch):
    monkeypatch.setattr(stt_module, "WhisperModel", None)
    data = client.post("/voice/transcribe", files=upload()).json()
    assert data["status"] == "unavailable"
    assert "requirements-voice" in data["message"]


def test_voice_command_reports_missing_dependency(client, voice_on, monkeypatch):
    monkeypatch.setattr(stt_module, "WhisperModel", None)
    data = client.post("/voice/command", files=upload()).json()
    assert data["status"] == "unavailable"


def test_speak_simulates_when_pyttsx3_missing(client, voice_on, monkeypatch):
    voice_on.tts_engine = "windows"
    monkeypatch.setattr(tts_module, "pyttsx3", None)
    data = client.post("/voice/speak", json={"text": "hello"}).json()
    assert data["status"] == "ok"
    assert data["simulated"] is True
    assert "pyttsx3" in data["note"]


# --- transcription flow ----------------------------------------------------------


def test_transcribe_wav(client, voice_on, fake_whisper):
    fake_whisper("abre la calculadora", language="es")
    data = client.post("/voice/transcribe", files=upload()).json()
    assert data["status"] == "ok"
    assert data["text"] == "abre la calculadora"
    assert data["language"] == "es"


def test_transcribe_rejects_non_wav(client, voice_on):
    response = client.post(
        "/voice/transcribe", files={"file": ("clip.mp3", b"12345", "audio/mpeg")}
    )
    assert response.status_code == 400
    assert ".wav" in response.json()["detail"]


def test_transcribe_rejects_empty_upload(client, voice_on):
    response = client.post("/voice/transcribe", files={"file": ("clip.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_stt_service_rejects_missing_file(voice_on, fake_whisper):
    fake_whisper("anything")
    result = SpeechToTextService().transcribe_file("C:/nope/missing.wav")
    assert "not found" in result["error"]


# --- /voice/command uses the same pipeline ---------------------------------------


def test_voice_command_goes_through_safety_layer(client, voice_on, fake_whisper):
    fake_whisper("open app notepad", language="en")

    first = client.post("/voice/command", files=upload()).json()
    assert first["status"] == "ok"
    assert first["transcription"] == "open app notepad"
    command = first["command"]
    assert command["planner"] == "rule_router"
    assert command["intent"] == "open_app"
    assert command["status"] == "needs_confirmation"  # same safety gate as typed

    second = client.post("/voice/command?confirm=true", files=upload()).json()
    assert second["command"]["status"] == "simulated"


def test_voice_command_spanish(client, voice_on, fake_whisper):
    fake_whisper("abre descargas", language="es")
    data = client.post("/voice/command", files=upload()).json()
    assert data["language"] == "es"
    assert data["command"]["intent"] == "open_folder"
    assert data["command"]["status"] == "simulated"


def test_voice_command_is_logged_like_typed_commands(client, voice_on, fake_whisper):
    fake_whisper("busca en internet gatos", language="es")
    client.post("/voice/command", files=upload())
    latest = client.get("/history?limit=1").json()[0]
    assert latest["input_text"] == "busca en internet gatos"
    assert latest["tool"] == "search_web"


def test_voice_command_never_launches_anything(client, voice_on, fake_whisper, launches):
    fake_whisper("open app notepad")
    client.post("/voice/command?confirm=true", files=upload())
    assert launches.popen_calls == []
    assert launches.startfile_calls == []


# --- TTS ---------------------------------------------------------------------------


def test_speak_real_with_mocked_pyttsx3(client, voice_on, monkeypatch):
    voice_on.tts_engine = "windows"
    spoken = []

    class FakeDriver:
        def say(self, text):
            spoken.append(text)

        def runAndWait(self):
            pass

    class FakePyttsx3:
        @staticmethod
        def init():
            return FakeDriver()

    monkeypatch.setattr(tts_module, "pyttsx3", FakePyttsx3)
    data = client.post("/voice/speak", json={"text": "hola mundo"}).json()
    assert data["status"] == "ok"
    assert data["simulated"] is False
    assert spoken == ["hola mundo"]


def test_speak_tool_simulated_by_default(settings):
    assert settings.enable_voice is False
    response = handle_command(CommandRequest(text="say hello world"))
    assert response.status is ExecutionStatus.SIMULATED
    assert response.result["simulated"] is True
