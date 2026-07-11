"""Response generator + spoken response tests. Ollama and TTS are mocked."""

import pytest
from fastapi.testclient import TestClient

from app.core.router import handle_command
from app.llm import response_generator
from app.llm.response_generator import generate_assistant_message, resolve_language
from app.main import app
from app.schemas.commands import (
    CommandRequest,
    CommandResponse,
    ExecutionStatus,
    Intent,
)
from tests.test_voice import upload  # shared WAV upload helper


def make_response(status, tool=None, message="msg", text="hola") -> CommandResponse:
    return CommandResponse(
        input_text=text, intent=Intent.UNKNOWN, tool=tool, status=status, message=message
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def voice_on(settings):
    from app.voice.stt import get_stt_service

    settings.enable_voice = True
    get_stt_service.cache_clear()
    yield settings
    get_stt_service.cache_clear()


# --- deterministic templates ---------------------------------------------------


@pytest.mark.parametrize(
    ("status", "language", "expected"),
    [
        (ExecutionStatus.SIMULATED, "en", "simulated"),
        (ExecutionStatus.SIMULATED, "es", "simulé"),
        (ExecutionStatus.EXECUTED, "en", "Done"),
        (ExecutionStatus.EXECUTED, "es", "Listo"),
        (ExecutionStatus.NEEDS_CONFIRMATION, "en", "confirmation"),
        (ExecutionStatus.NEEDS_CONFIRMATION, "es", "confirmación"),
        (ExecutionStatus.REJECTED, "en", "I can't do that"),
        (ExecutionStatus.REJECTED, "es", "No puedo hacer eso"),
        (ExecutionStatus.BLOCKED, "en", "destructive actions are blocked"),
        (ExecutionStatus.BLOCKED, "es", "acciones destructivas"),
        (ExecutionStatus.NOT_HANDLED, "en", "didn't understand"),
        (ExecutionStatus.NOT_HANDLED, "es", "No entendí"),
    ],
)
def test_deterministic_templates(settings, status, language, expected):
    assert settings.enable_response_generator is False
    response = make_response(status, tool="open_folder")
    assert expected in generate_assistant_message(response, language=language)


def test_language_detection_heuristic():
    assert resolve_language(make_response(ExecutionStatus.SIMULATED, text="open the folder")) == "en"
    assert resolve_language(make_response(ExecutionStatus.SIMULATED, text="abre la carpeta")) == "es"
    # unknown language falls back to Spanish
    assert resolve_language(make_response(ExecutionStatus.SIMULATED, text="xyzzy 123")) == "es"
    # explicit override wins
    assert resolve_language(make_response(ExecutionStatus.SIMULATED, text="abre"), "en") == "en"


# --- /command carries assistant_message -----------------------------------------


def test_command_endpoint_returns_assistant_message(client, settings):
    data = client.post("/command", json={"text": "open app notepad"}).json()
    assert "confirmation" in data["assistant_message"]  # English command -> English reply

    data = client.post("/command", json={"text": "abre notepad"}).json()
    assert "confirmación" in data["assistant_message"]  # Spanish command -> Spanish reply


def test_confirmed_and_unconfirmed_messages_differ(client, settings):
    first = client.post("/command", json={"text": "open app notepad"}).json()
    second = client.post("/command", json={"text": "open app notepad", "confirm": True}).json()
    assert first["assistant_message"] != second["assistant_message"]
    assert "confirmation" in first["assistant_message"]
    assert "simulated" in second["assistant_message"]


def test_language_hint_field_wins(client, settings):
    data = client.post(
        "/command", json={"text": "open app notepad", "language": "es"}
    ).json()
    assert "confirmación" in data["assistant_message"]


# --- LLM-generated responses (mocked) --------------------------------------------


def test_llm_response_used_when_enabled(settings, monkeypatch):
    settings.enable_response_generator = True
    monkeypatch.setattr(
        response_generator, "_llm_message", lambda response, language: "Custom reply."
    )
    response = handle_command(CommandRequest(text="search for cats"))
    assert response.assistant_message == "Custom reply."


def test_llm_response_failure_falls_back_to_template(settings, monkeypatch):
    settings.enable_response_generator = True
    monkeypatch.setattr(response_generator, "_llm_message", lambda response, language: None)
    response = handle_command(CommandRequest(text="search for cats"))
    assert "simulated" in response.assistant_message  # English template fallback


# --- /voice/command: assistant_message + spoken response ---------------------------


def test_voice_command_returns_assistant_message(client, voice_on, monkeypatch):
    from tests.test_voice import FakeInfo, FakeSegment
    from app.voice import stt as stt_module

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, language=None, **kwargs):
            return [FakeSegment("abre la calculadora")], FakeInfo("es")

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)

    data = client.post("/voice/command", files=upload()).json()
    assert data["assistant_message"] == data["command"]["assistant_message"]
    assert "confirmación" in data["assistant_message"]  # detected es -> Spanish
    assert "speech" not in data  # spoken response disabled by default


def _install_fake_whisper(monkeypatch, text, language="en"):
    from tests.test_voice import FakeInfo, FakeSegment
    from app.voice import stt as stt_module

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, language=None, **kwargs):
            return [FakeSegment(text)], FakeInfo(language)

    monkeypatch.setattr(stt_module, "WhisperModel", FakeWhisperModel)


def _install_fake_tts(monkeypatch):
    from app.voice import tts as tts_module

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
    return spoken


def test_spoken_response_with_mocked_tts(client, voice_on, monkeypatch):
    voice_on.voice_speak_command_response = True
    voice_on.tts_engine = "windows"
    _install_fake_whisper(monkeypatch, "abre descargas", "es")
    spoken = _install_fake_tts(monkeypatch)

    data = client.post("/voice/command", files=upload()).json()
    assert data["speech"]["status"] == "spoken"
    assert spoken == [data["assistant_message"]]
    assert "simulé" in data["assistant_message"]  # safe action, Spanish


def test_spoken_response_tts_unavailable(client, voice_on, monkeypatch):
    from app.voice import tts as tts_module

    voice_on.voice_speak_command_response = True
    voice_on.tts_engine = "windows"
    _install_fake_whisper(monkeypatch, "abre descargas", "es")
    monkeypatch.setattr(tts_module, "pyttsx3", None)

    data = client.post("/voice/command", files=upload()).json()
    assert data["speech"]["status"] == "simulated"
    assert "pyttsx3" in data["speech"]["note"]
    assert data["assistant_message"]  # message still returned


def test_needs_confirmation_spoken_response(client, voice_on, monkeypatch):
    voice_on.voice_speak_command_response = True
    voice_on.tts_engine = "windows"
    _install_fake_whisper(monkeypatch, "open app notepad", "en")
    spoken = _install_fake_tts(monkeypatch)

    data = client.post("/voice/command", files=upload()).json()
    assert data["command"]["status"] == "needs_confirmation"
    assert "confirmation" in spoken[0]

    confirmed = client.post("/voice/command?confirm=true", files=upload()).json()
    assert confirmed["command"]["status"] == "simulated"
    assert spoken[1] != spoken[0]  # confirmed reply differs from unconfirmed


def test_rejected_spoken_response(client, voice_on, monkeypatch):
    voice_on.voice_speak_command_response = True
    voice_on.tts_engine = "windows"
    _install_fake_whisper(monkeypatch, "abre photoshop", "es")
    spoken = _install_fake_tts(monkeypatch)

    data = client.post("/voice/command?confirm=true", files=upload()).json()
    assert data["command"]["status"] == "rejected"
    assert spoken[0].startswith("No puedo hacer eso")


def test_blocked_destructive_message_both_languages():
    response = make_response(ExecutionStatus.BLOCKED, tool="delete_file")
    assert "blocked" in generate_assistant_message(response, language="en")
    assert "bloqueadas" in generate_assistant_message(response, language="es")


def test_assistant_message_never_mentions_planner(client, settings):
    data = client.post("/command", json={"text": "open app notepad"}).json()
    assert "planner" not in data["assistant_message"].lower()
    assert "fallback" not in data["assistant_message"].lower()
