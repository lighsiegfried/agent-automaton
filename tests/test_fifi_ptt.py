"""Unit tests for scripts/fifi_ptt.py — the push-to-talk client.

Microphone, global hotkey, HTTP, and TTS are all mocked: no real recording,
key hook, network call, app launch, or audio output happens. These tests pin
the recording lifecycle, the one-shot confirmation flow, and startup validation.
"""

import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_ptt  # noqa: E402


# --- fakes -----------------------------------------------------------------------


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self):
        self.gets = {}
        self.posts = []
        self.raise_on_post = False

    def get(self, path, timeout=None):
        return self.gets.get(path, FakeResp(200, {}))

    def post(self, path, params=None, files=None, json=None, timeout=None):
        self.posts.append({"path": path, "params": params, "json": json})
        if self.raise_on_post:
            raise httpx.ConnectError("refused")
        return FakeResp(200, {"status": "ok"})

    def close(self):
        pass


class FakeRecorder:
    def __init__(self, duration=1.0, fail_start=False, fail_stop=False):
        self.duration = duration
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = self.stopped = self.discarded = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("mic disconnected")
        self.started = True

    def stop(self):
        if self.fail_stop:
            raise RuntimeError("mic error")
        self.stopped = True
        return ("AUDIO", self.duration)

    def discard(self):
        self.discarded = True


def make_client(clock=None, audio_feedback=False, recorder=None, http=None, **cfg):
    config = fifi_ptt.PttConfig(
        server="http://test",
        hotkey=cfg.get("hotkey", "ctrl+alt+space"),
        max_seconds=cfg.get("max_seconds", 20.0),
        min_seconds=cfg.get("min_seconds", 0.4),
        confirm_window=cfg.get("confirm_window", 30.0),
        audio_feedback=audio_feedback,
        input_device="default",
        speak_responses=cfg.get("speak_responses", False),
    )
    out: list[str] = []
    rec = recorder or FakeRecorder()
    client = fifi_ptt.PttClient(
        config,
        http=http or FakeHttp(),
        recorder_factory=lambda: rec,
        clock=clock or Clock(),
        out=out.append,
    )
    return client, out


def _needs_confirmation(transcription="open app notepad", language="en"):
    return {
        "status": "ok",
        "transcription": transcription,
        "language": language,
        "assistant_message": "I need confirmation.",
        "command": {
            "status": "needs_confirmation",
            "intent": "open_app",
            "planner": "rule_router",
            "safety_level": "sensitive",
            "message": "Confirm?",
        },
    }


# --- recording lifecycle ---------------------------------------------------------


def test_press_record_release_submits(monkeypatch):
    rec = FakeRecorder(duration=1.0)
    client, out = make_client(recorder=rec)
    submitted = []
    monkeypatch.setattr(client, "_save_temp_wav", lambda audio: "/tmp/fake.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: submitted.append(p))

    assert client.begin_recording() is True
    assert rec.started
    client.finish_recording()
    assert rec.stopped
    assert submitted == ["/tmp/fake.wav"]
    # no lingering state / timer / recorder after release
    assert client._recording is False
    assert client._recorder is None
    assert client._max_timer is None


def test_concurrent_recording_is_prevented(monkeypatch):
    client, out = make_client()
    monkeypatch.setattr(client, "_save_temp_wav", lambda audio: "/tmp/fake.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: None)
    assert client.begin_recording() is True
    assert client.begin_recording() is False  # debounced / no concurrent capture
    client.finish_recording()


def test_too_short_recording_is_ignored(monkeypatch):
    rec = FakeRecorder(duration=0.1)  # below min_seconds 0.4
    client, out = make_client(recorder=rec)
    monkeypatch.setattr(
        client, "handle_utterance",
        lambda p: pytest.fail("must not submit a too-short clip"),
    )
    client.begin_recording()
    client.finish_recording()
    assert any("Too short" in m for m in out)


def test_max_duration_stops_and_submits(monkeypatch):
    rec = FakeRecorder(duration=20.0)
    client, out = make_client(recorder=rec)
    submitted = []
    monkeypatch.setattr(client, "_save_temp_wav", lambda audio: "/tmp/fake.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: submitted.append(p))
    client.begin_recording()
    client._on_max_duration()  # simulate the max-duration timer firing
    assert submitted == ["/tmp/fake.wav"]
    assert any("Max duration" in m for m in out)


def test_escape_cancels_without_submitting(monkeypatch):
    rec = FakeRecorder()
    client, out = make_client(recorder=rec)
    monkeypatch.setattr(
        client, "handle_utterance",
        lambda p: pytest.fail("cancel must not submit"),
    )
    client.begin_recording()
    client.cancel_recording()
    assert rec.discarded
    assert client._recording is False
    assert any("cancelled" in m.lower() for m in out)


def test_microphone_disconnect_is_handled(monkeypatch):
    rec = FakeRecorder(fail_start=True)
    client, out = make_client(recorder=rec)
    assert client.begin_recording() is False
    assert client._recording is False  # state safely reset
    assert any("Microphone unavailable" in m for m in out)


def test_no_audio_output_when_feedback_disabled():
    client, out = make_client(audio_feedback=False)
    client._feedback("start")  # must not raise or beep
    assert out == []


# --- hotkey listener: per-key release detection ----------------------------------


def _wire_listener(client, hotkey="ctrl+alt+space"):
    listener = fifi_ptt.HotkeyListener(
        hotkey, on_activate=client.begin_recording, on_release_any=client.finish_recording
    )
    client.latch_hook = listener.force_latch
    return listener


def test_ctrl_alt_space_release_stops_immediately(monkeypatch):
    rec = FakeRecorder(duration=1.0)
    client, out = make_client(recorder=rec)
    submits = []
    monkeypatch.setattr(client, "_save_temp_wav", lambda a: "/tmp/x.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: submits.append(p))
    listener = _wire_listener(client)

    listener.on_event("ctrl", True)
    listener.on_event("alt", True)
    listener.on_event("space", True)
    assert client._recording is True  # started once all keys are down

    listener.on_event("space", False)  # release ONE key
    assert client._recording is False  # stops immediately, not at max duration
    assert submits == ["/tmp/x.wav"]


def test_single_key_f8_press_release(monkeypatch):
    rec = FakeRecorder(duration=1.0)
    client, out = make_client(recorder=rec)
    submits = []
    monkeypatch.setattr(client, "_save_temp_wav", lambda a: "/tmp/x.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: submits.append(p))
    listener = _wire_listener(client, hotkey="f8")

    listener.on_event("f8", True)
    assert client._recording is True
    listener.on_event("f8", False)
    assert client._recording is False
    assert submits == ["/tmp/x.wav"]


def test_key_repeat_is_ignored():
    starts, stops = [], []
    listener = fifi_ptt.HotkeyListener(
        "ctrl+alt+space", on_activate=lambda: starts.append(1), on_release_any=lambda: stops.append(1)
    )
    listener.on_event("ctrl", True)
    listener.on_event("alt", True)
    listener.on_event("space", True)
    listener.on_event("space", True)  # auto-repeat
    listener.on_event("alt", True)    # auto-repeat
    assert starts == [1]              # activated exactly once
    listener.on_event("space", False)
    assert stops == [1]


def test_no_repeated_recording_after_max_duration(monkeypatch):
    rec = FakeRecorder(duration=20.0)
    client, out = make_client(recorder=rec)
    submits = []
    monkeypatch.setattr(client, "_save_temp_wav", lambda a: "/tmp/x.wav")
    monkeypatch.setattr(client, "handle_utterance", lambda p: submits.append(p))
    listener = _wire_listener(client)

    listener.on_event("ctrl", True)
    listener.on_event("alt", True)
    listener.on_event("space", True)
    client._on_max_duration()          # stop at max + latch (keys still held)
    assert submits == ["/tmp/x.wav"]
    assert client._recording is False

    listener.on_event("space", True)   # auto-repeat while still holding the combo
    assert client._recording is False  # latched — no second recording
    assert submits == ["/tmp/x.wav"]

    # Release everything, then press again — now it re-arms.
    listener.on_event("space", False)
    listener.on_event("alt", False)
    listener.on_event("ctrl", False)
    listener.on_event("ctrl", True)
    listener.on_event("alt", True)
    listener.on_event("space", True)
    assert client._recording is True
    client.cancel_recording()


# --- HTTP failure: connection vs processing timeout ------------------------------


def test_api_unavailable_returns_none(tmp_path):
    http = FakeHttp()
    http.raise_on_post = True
    client, out = make_client(http=http)
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfake")
    assert client._post_voice_command(str(wav), confirm=False) is None
    assert any("not reachable" in m for m in out)


def test_processing_timeout_not_labeled_unreachable(tmp_path):
    class TimeoutHttp:
        def post(self, *a, **k):
            raise httpx.ReadTimeout("slow STT")

        def close(self):
            pass

    client, out = make_client(http=TimeoutHttp())
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfake")
    assert client._post_voice_command(str(wav), confirm=False) is None
    assert any("processing timed out" in m.lower() for m in out)
    assert not any("not reachable" in m.lower() for m in out)


def test_connect_error_is_labeled_unreachable(tmp_path):
    class ConnHttp:
        def post(self, *a, **k):
            raise httpx.ConnectError("refused")

        def close(self):
            pass

    client, out = make_client(http=ConnHttp())
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfake")
    assert client._post_voice_command(str(wav), confirm=False) is None
    assert any("not reachable" in m.lower() for m in out)


# --- confirmation flow -----------------------------------------------------------


def _wire_posts(monkeypatch, client, voice_queue):
    text_calls = []
    monkeypatch.setattr(client, "_post_voice_command", lambda path, confirm: voice_queue.pop(0))
    monkeypatch.setattr(
        client, "_post_text_command",
        lambda text, confirm, language: text_calls.append((text, confirm, language))
        or {
            "status": "simulated",
            "intent": "open_app",
            "planner": "rule_router",
            "safety_level": "sensitive",
            "message": "done",
            "assistant_message": "Done.",
        },
    )
    return text_calls


def test_confirmation_resends_exact_original(monkeypatch):
    client, out = make_client()
    queue = [
        _needs_confirmation("open app notepad", "en"),
        {"status": "ok", "transcription": "confirm", "language": "en",
         "command": {"status": "not_handled", "intent": "unknown", "planner": "rule_router"}},
    ]
    text_calls = _wire_posts(monkeypatch, client, queue)

    client.handle_utterance("a.wav")
    assert client.pending.valid()  # sensitive action is pending

    client.handle_utterance("b.wav")  # spoken "confirm"
    assert text_calls == [("open app notepad", True, "en")]  # exact original, confirm=true
    assert not client.pending.valid()


def test_cancellation_clears_pending(monkeypatch):
    client, out = make_client()
    queue = [
        _needs_confirmation(),
        {"status": "ok", "transcription": "cancel", "language": "en",
         "command": {"status": "not_handled", "intent": "unknown", "planner": "rule_router"}},
    ]
    text_calls = _wire_posts(monkeypatch, client, queue)

    client.handle_utterance("a.wav")
    assert client.pending.valid()
    client.handle_utterance("b.wav")  # spoken "cancel"
    assert not client.pending.valid()
    assert text_calls == []  # nothing resent


def test_confirmation_expires(monkeypatch):
    clock = Clock()
    client, out = make_client(clock=clock, confirm_window=30.0)
    queue = [
        _needs_confirmation(),
        {"status": "ok", "transcription": "confirm", "language": "en",
         "command": {"status": "not_handled", "intent": "unknown", "planner": "rule_router"}},
    ]
    text_calls = _wire_posts(monkeypatch, client, queue)

    client.handle_utterance("a.wav")
    assert client.pending.valid()
    clock.advance(31.0)  # past the confirmation window
    assert not client.pending.valid()
    client.handle_utterance("b.wav")  # "confirm" now too late — treated as a new command
    assert text_calls == []  # no resend


def test_different_command_clears_pending(monkeypatch):
    client, out = make_client()
    queue = [
        _needs_confirmation("open app notepad", "en"),
        {"status": "ok", "transcription": "busca gatos", "language": "es",
         "command": {"status": "simulated", "intent": "search_web",
                     "planner": "llm_planner", "safety_level": "safe", "message": "ok"}},
    ]
    text_calls = _wire_posts(monkeypatch, client, queue)

    client.handle_utterance("a.wav")
    assert client.pending.valid()
    client.handle_utterance("b.wav")  # a different command
    assert not client.pending.valid()  # previous pending dropped
    assert text_calls == []


def test_pending_does_not_survive_new_client():
    # A fresh client (i.e. a restarted process) starts with no pending command.
    client, _ = make_client()
    assert client.pending.valid() is False


# --- startup validation ----------------------------------------------------------


def _health(app="agent-automaton", voice=True, planner=True):
    return FakeResp(200, {"app": app, "voice": voice, "llm_planner": planner})


def _preflight_ok():
    return FakeResp(200, {
        "status": "ok", "passed": True, "device": "cuda", "compute_type": "float16",
        "model": "small", "gpu": "NVIDIA GeForce RTX 5060 Ti",
        "ctranslate2_version": "4.8.1", "cpu_fallback_active": False,
        "message": "STT preflight passed.",
    })


def test_startup_ok_warns_when_planner_and_speech_off():
    http = FakeHttp()
    http.gets = {
        "/health": _health(planner=False),
        "/identity": FakeResp(200, {"project_name": "agent-automaton"}),
        "/voice/preflight": _preflight_ok(),
    }
    config = fifi_ptt.PttConfig.load(server="http://test")
    config.speak_responses = False
    out = []
    assert fifi_ptt.validate_startup(config, http, out.append) is True
    assert any("LLM planner is disabled" in m for m in out)
    assert any("spoken responses are off" in m for m in out)
    assert any("STT preflight — passed" in m for m in out)


def test_startup_refuses_when_preflight_fails():
    http = FakeHttp()
    http.gets = {
        "/health": _health(),
        "/identity": FakeResp(200, {"project_name": "agent-automaton"}),
        "/voice/preflight": FakeResp(200, {
            "status": "error", "passed": False, "device": "cuda",
            "compute_type": "int8_float16", "model": "small",
            "gpu": "NVIDIA GeForce RTX 5060 Ti", "ctranslate2_version": "4.4.0",
            "message": "STT preflight failed: CUBLAS_STATUS_NOT_SUPPORTED",
        }),
    }
    config = fifi_ptt.PttConfig.load(server="http://test")
    out = []
    assert fifi_ptt.validate_startup(config, http, out.append) is False
    assert any("STT preflight — FAILED" in m for m in out)


def test_startup_reports_cpu_fallback_active():
    http = FakeHttp()
    http.gets = {
        "/health": _health(),
        "/identity": FakeResp(200, {"project_name": "agent-automaton"}),
        "/voice/preflight": FakeResp(200, {
            "status": "ok", "passed": True, "device": "cpu", "compute_type": "int8",
            "model": "small", "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "ctranslate2_version": "4.8.1", "cpu_fallback_active": True,
            "message": "STT preflight passed.",
        }),
    }
    config = fifi_ptt.PttConfig.load(server="http://test")
    out = []
    assert fifi_ptt.validate_startup(config, http, out.append) is True
    assert any("CPU fallback is ACTIVE" in m for m in out)


def test_startup_refuses_non_agent_automaton():
    http = FakeHttp()
    http.gets = {"/health": _health(app="something-else")}
    config = fifi_ptt.PttConfig.load(server="http://test")
    out = []
    assert fifi_ptt.validate_startup(config, http, out.append) is False
    assert any("not agent-automaton" in m for m in out)


def test_startup_refuses_when_voice_disabled():
    http = FakeHttp()
    http.gets = {
        "/health": _health(voice=False),
        "/identity": FakeResp(200, {"project_name": "agent-automaton"}),
    }
    config = fifi_ptt.PttConfig.load(server="http://test")
    out = []
    assert fifi_ptt.validate_startup(config, http, out.append) is False
    assert any("Voice is disabled" in m for m in out)


def test_startup_refuses_when_api_unreachable():
    class Dead:
        def get(self, path, timeout=None):
            raise httpx.ConnectError("refused")

    config = fifi_ptt.PttConfig.load(server="http://test")
    out = []
    assert fifi_ptt.validate_startup(config, Dead(), out.append) is False
    assert any("not reachable" in m for m in out)


# --- misc ------------------------------------------------------------------------


def test_confirmation_phrases_are_exact_not_fuzzy():
    assert fifi_ptt._normalize("Confirm.") == "confirm"
    assert fifi_ptt._normalize("Yes, confirm") == "yes confirm"
    assert "confirm" in fifi_ptt.CONFIRM_PHRASES
    assert "confirmar" in fifi_ptt.CONFIRM_PHRASES
    # a near-miss must NOT match (no fuzzy matching)
    assert "confirmed" not in fifi_ptt.CONFIRM_PHRASES
    assert "please confirm the thing" not in fifi_ptt.CONFIRM_PHRASES


def test_check_desktop_deps_returns_tuple():
    ok, missing = fifi_ptt.check_desktop_deps()
    assert isinstance(ok, bool)
    assert isinstance(missing, list)
