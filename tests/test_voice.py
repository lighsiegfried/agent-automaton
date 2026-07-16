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


# --- Phase 3D.1: daily wake-mode TTS latency guard (Qwen -> Kokoro) ------------------

import httpx  # noqa: E402


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class FakeVoiceLab:
    """A fake Voice Lab /synthesize: per-profile behavior 'timeout' | 'error' |
    ('ok', engine). Records which profiles were contacted, in order."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls: list[str] = []

    def __call__(self, url, json=None, timeout=None):
        profile = json["profile"]
        self.calls.append(profile)
        outcome = self.behavior.get(profile, ("ok", "kokoro"))
        if outcome == "timeout":
            raise httpx.ReadTimeout("synthesis slow")
        if outcome == "error":
            raise httpx.ConnectError("worker down")
        _, engine = outcome
        # The worker reports its OWN fallback (requested engine failed). Here the
        # requested engine serves the request, so False; the guard's Qwen->Kokoro
        # fallback is signalled separately by _voice_lab_payload's param.
        return _Resp({
            "status": "ok", "profile": profile, "engine": engine, "fallback_used": False,
        })


def _wake_voice(voice_on, monkeypatch, *, active, active_info, behavior, fallback="fifi_warm"):
    voice_on.enable_voice = True
    voice_on.tts_engine = "voice_lab"
    voice_on.voice_profile = fallback  # the Kokoro fallback voice
    voice_on.wake_max_tts_wait_seconds = 20.0
    monkeypatch.setattr(tts_module, "active_voice_profile", lambda: active)
    monkeypatch.setattr(
        tts_module, "profile_engine_info",
        lambda name: active_info if name == active else
        {"engine": "kokoro", "model": "hexgrad/Kokoro-82M", "is_voice_design": False},
    )
    fake = FakeVoiceLab(behavior)
    monkeypatch.setattr(tts_module.httpx, "post", fake)
    return fake


def test_wake_tts_qwen_timeout_falls_back_to_kokoro(voice_on, monkeypatch):
    """A slow/cold Qwen daily voice must never keep the user waiting: the reply
    falls back to KOKORO (not Windows), reporting requested vs actual engine."""
    fake = _wake_voice(
        voice_on, monkeypatch,
        active="fifi_luna",
        active_info={"engine": "qwen3_tts", "model": "Qwen/...0.6B-Base", "is_voice_design": False},
        behavior={"fifi_luna": "timeout", "fifi_warm": ("ok", "kokoro")},
    )
    result = tts_module.TextToSpeechService().speak("hola", wake=True)
    assert result["requested_engine"] == "qwen3_tts"
    assert result["actual_engine"] == "kokoro"
    assert result["fallback_used"] is True
    assert "Kokoro" in result["fallback_reason"]
    assert fake.calls == ["fifi_luna", "fifi_warm"]  # tried Qwen, then Kokoro


def test_wake_tts_never_loads_voicedesign(voice_on, monkeypatch):
    """A VoiceDesign active voice is NEVER contacted in wake mode — the reply is
    spoken with Kokoro directly."""
    fake = _wake_voice(
        voice_on, monkeypatch,
        active="fifi_nova",
        active_info={"engine": "qwen3_tts", "model": "Qwen/...VoiceDesign", "is_voice_design": True},
        behavior={"fifi_warm": ("ok", "kokoro")},
    )
    result = tts_module.TextToSpeechService().speak("hola", wake=True)
    assert result["requested_engine"] == "voice_design"
    assert result["actual_engine"] == "kokoro"
    assert result["fallback_used"] is True
    assert "fifi_nova" not in fake.calls  # the VoiceDesign profile is never loaded
    assert fake.calls == ["fifi_warm"]


def test_wake_tts_kokoro_active_has_no_fallback(voice_on, monkeypatch):
    fake = _wake_voice(
        voice_on, monkeypatch,
        active="fifi_warm",
        active_info={"engine": "kokoro", "model": "hexgrad/Kokoro-82M", "is_voice_design": False},
        behavior={"fifi_warm": ("ok", "kokoro")},
    )
    result = tts_module.TextToSpeechService().speak("hola", wake=True)
    assert result["requested_engine"] == "kokoro"
    assert result["actual_engine"] == "kokoro"
    assert result["fallback_used"] is False
    assert fake.calls == ["fifi_warm"]  # a single attempt, no fallback


def test_non_wake_qwen_timeout_uses_windows_not_kokoro(voice_on, monkeypatch):
    """Outside wake mode the behavior is unchanged: a worker timeout degrades to
    Windows TTS, and there is NO Kokoro re-request."""
    fake = _wake_voice(
        voice_on, monkeypatch,
        active="fifi_luna",
        active_info={"engine": "qwen3_tts", "model": "Qwen/...0.6B-Base", "is_voice_design": False},
        behavior={"fifi_luna": "timeout"},
    )

    class FakeDriver:
        def say(self, text):
            pass

        def runAndWait(self):
            pass

    monkeypatch.setattr(tts_module, "pyttsx3", type("P", (), {"init": staticmethod(FakeDriver)}))
    result = tts_module.TextToSpeechService().speak("hola", wake=False)
    assert result.get("fallback_from") == "voice_lab"  # Windows, not Kokoro
    assert result.get("actual_engine") is None
    assert fake.calls == ["fifi_luna"]  # no second (Kokoro) attempt in non-wake mode


def test_voice_command_wake_forces_spoken_reply_with_guard(client, voice_on, fake_whisper, monkeypatch):
    """?wake=true voices the reply even when VOICE_SPEAK_COMMAND_RESPONSE is off,
    applying the latency guard; a non-wake request stays silent."""
    voice_on.voice_speak_command_response = False  # PTT speaking OFF
    fake = _wake_voice(
        voice_on, monkeypatch,
        active="fifi_luna",
        active_info={"engine": "qwen3_tts", "model": "Qwen/...0.6B-Base", "is_voice_design": False},
        behavior={"fifi_luna": "timeout", "fifi_warm": ("ok", "kokoro")},
    )
    fake_whisper("abre descargas", language="es")
    data = client.post("/voice/command?wake=true", files=upload()).json()
    assert data["status"] == "ok"
    assert data["command"]["intent"] == "open_folder"  # command STILL executed
    speech = data["speech"]
    assert speech["status"] == "spoken"
    assert speech["requested_engine"] == "qwen3_tts"
    assert speech["actual_engine"] == "kokoro"
    assert speech["fallback_used"] is True

    # A non-wake request with speaking off does NOT speak (PTT unchanged).
    silent = client.post("/voice/command", files=upload()).json()
    assert "speech" not in silent
