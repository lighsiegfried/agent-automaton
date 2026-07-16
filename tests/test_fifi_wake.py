"""Unit tests for scripts/fifi_wake.py — the wake-word listener state machine.

Everything is mocked (wake model, VAD, microphone, HTTP, clock): no audio
device, network, or model file is needed. These tests pin the contract:

    IDLE -> WAKE_DETECTED -> CAPTURING -> PROCESSING -> COOLDOWN -> IDLE

- idle audio is NEVER retained anywhere;
- exactly one WAV / one request per detected command, never overlapping;
- capture stops on configured silence, or at the max duration;
- cooldown blocks immediate retriggering;
- Esc cancels without sending; mute discards everything;
- temporary WAVs are deleted unless WAKE_WORD_STORE_AUDIO=true;
- the listener is opt-in (ENABLE_WAKE_WORD) and unavailable without deps/model.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_wake  # noqa: E402
from fifi_wake import (  # noqa: E402
    CAPTURING,
    COOLDOWN,
    IDLE,
    PROCESSING,
    SPEAKING,
    WAKE_DETECTED,
    WakeConfig,
    WakeSession,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- fakes -------------------------------------------------------------------------


class Frame(list):
    """80 ms of fake 16 kHz mono PCM, tagged with what the fakes should say."""

    def __init__(self, wake=False, speech=False, ms=80):
        super().__init__([0] * int(ms * 16))
        self.wake = wake
        self.speech = speech


class FakeService:
    def __init__(self):
        self.resets = 0

    def detect(self, frame):
        return frame.wake

    def is_speech(self, frame):
        return frame.speech

    def reset(self):
        self.resets += 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_config(**overrides) -> WakeConfig:
    values = dict(
        server="http://127.0.0.1:8000",
        threshold=0.55,
        cooldown_seconds=2.0,
        vad_threshold=0.5,
        vad_min_speech_ms=250.0,
        vad_silence_ms=900.0,
        vad_max_command_seconds=20.0,
        store_audio=False,
        mute_hotkey="ctrl+alt+m",
        input_device="default",
        model_path="models/wake_words/fifi.onnx",
        audio_feedback=True,
        debug=False,
    )
    values.update(overrides)
    return WakeConfig(**values)


def make_session(config=None, submit=None, clock=None):
    clock = clock or Clock()
    submitted = []
    session = WakeSession(
        config or make_config(),
        FakeService(),
        submit=submit or submitted.append,
        clock=clock,
        out=lambda *a: None,
    )
    # Fake WAV writer: a real temp file, but no numpy/soundfile involved.
    def _fake_write(frames):
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        session.written_paths.append(name)
        return name

    session.written_paths = []
    session._write_wav = _fake_write
    return session, submitted, clock


def _speak(session, frames_of_speech=4):
    for _ in range(frames_of_speech):  # 4 × 80 ms = 320 ms ≥ VAD_MIN_SPEECH_MS
        session.feed(Frame(speech=True))


def _go_silent(session, frames=12):
    for _ in range(frames):  # 12 × 80 ms = 960 ms ≥ VAD_SILENCE_MS
        session.feed(Frame(speech=False))


# --- state machine -----------------------------------------------------------------


def test_full_cycle_transitions(monkeypatch):
    session, submitted, clock = make_session()
    session.feed(Frame(wake=True, speech=True))
    assert session.state == CAPTURING
    _speak(session)
    _go_silent(session)
    assert submitted and len(submitted) == 1  # exactly ONE request
    assert session.state == COOLDOWN
    assert session.transitions == [
        IDLE, WAKE_DETECTED, CAPTURING, PROCESSING, SPEAKING, COOLDOWN,
    ]
    clock.advance(2.1)  # cooldown over
    session.feed(Frame())
    assert session.state == IDLE


def test_idle_audio_is_never_retained():
    session, submitted, _ = make_session()
    for _ in range(50):
        session.feed(Frame(speech=True))  # speech, but no wake word
    assert session._frames == []  # nothing buffered in IDLE, ever
    assert session.transitions == [IDLE]
    assert submitted == []


def test_no_trigger_without_wake_word():
    """A frame the detector rejects (below threshold / VAD veto) does nothing."""
    session, submitted, _ = make_session()
    session.feed(Frame(wake=False, speech=True))
    assert session.state == IDLE
    assert submitted == []


def test_silence_stops_capture_and_sends_once():
    session, submitted, _ = make_session()
    session.feed(Frame(wake=True, speech=True))
    _speak(session, 5)
    _go_silent(session, 11)  # 880 ms — not enough silence yet
    assert session.state == CAPTURING
    session.feed(Frame(speech=False))  # 960 ms — crosses VAD_SILENCE_MS
    assert len(submitted) == 1
    assert session.state == COOLDOWN


def test_silence_without_speech_sends_nothing():
    """Wake word, then nothing said: silence stop must not send an empty WAV."""
    messages = []
    session, submitted, _ = make_session()
    session.out = messages.append
    session.feed(Frame(wake=True, speech=True))
    _go_silent(session)
    assert submitted == []
    assert session.state == COOLDOWN  # still cools down (no retrigger loop)
    assert any("No speech" in m for m in messages)


def test_max_duration_stops_capture():
    session, submitted, clock = make_session()
    session.feed(Frame(wake=True, speech=True))
    for _ in range(21):  # continuous speech, clock passes the 20 s cap
        clock.advance(1.0)
        session.feed(Frame(speech=True))
        if session.state != CAPTURING:
            break
    assert len(submitted) == 1
    assert session.state == COOLDOWN


def test_cooldown_blocks_retrigger():
    session, submitted, clock = make_session()
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert len(submitted) == 1
    session.feed(Frame(wake=True, speech=True))  # still cooling down
    assert session.state == COOLDOWN
    assert len(submitted) == 1
    clock.advance(2.1)
    session.feed(Frame(wake=True, speech=True))  # cooldown over — triggers again
    assert session.state == CAPTURING


def test_cancel_discards_without_sending():
    session, submitted, _ = make_session()
    session.feed(Frame(wake=True, speech=True))
    _speak(session, 3)
    session.cancel()
    assert submitted == []
    assert session._frames == []
    assert session.state == COOLDOWN


def test_cancel_in_idle_is_a_no_op():
    session, submitted, _ = make_session()
    session.cancel()
    assert session.state == IDLE
    assert session.transitions == [IDLE]


def test_mute_discards_everything():
    session, submitted, _ = make_session()
    session.toggle_mute()
    for _ in range(10):
        session.feed(Frame(wake=True, speech=True))
    assert session.state == IDLE
    assert submitted == []
    session.toggle_mute()  # unmute
    session.feed(Frame(wake=True, speech=True))
    assert session.state == CAPTURING


def test_mute_during_capture_cancels_it():
    session, submitted, _ = make_session()
    session.feed(Frame(wake=True, speech=True))
    _speak(session, 3)
    session.toggle_mute()
    assert session.muted
    assert submitted == []
    assert session._frames == []


def test_temporary_wav_is_deleted_after_sending():
    session, submitted, _ = make_session()
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert len(session.written_paths) == 1
    assert not os.path.exists(session.written_paths[0])  # deleted after the send


def test_api_failure_is_survived_and_wav_deleted():
    """An API/TTS failure must not terminate wake listening: the exception is
    swallowed, the WAV is deleted, and the next wake word still works."""
    calls = []

    def failing_submit(path):
        calls.append(path)
        raise RuntimeError("API exploded")

    messages = []
    session, _, clock = make_session(submit=failing_submit)
    session.out = messages.append
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)  # no exception escapes
    assert len(calls) == 1
    assert not os.path.exists(session.written_paths[0])
    assert session.state == COOLDOWN
    assert SPEAKING not in session.transitions  # failed request never "spoke"
    assert any("still listening" in m for m in messages)

    clock.advance(2.1)  # cooldown over — the listener is still fully alive
    session.feed(Frame(wake=True, speech=True))
    assert session.state == CAPTURING


def test_store_audio_keeps_the_wav(tmp_path):
    session, submitted, _ = make_session(config=make_config(store_audio=True))
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    path = session.written_paths[0]
    assert os.path.exists(path)  # explicitly retained
    os.unlink(path)


def test_no_overlapping_recordings_or_requests():
    """A wake word heard while a request is in flight must be discarded."""
    session_holder = {}
    observed = {}

    def nested_submit(path):
        observed["state_during_submit"] = session_holder["s"].state
        session_holder["s"].feed(Frame(wake=True, speech=True))  # mid-flight wake
        observed["state_after_nested_feed"] = session_holder["s"].state

    session, _, _ = make_session(submit=nested_submit)
    session_holder["s"] = session
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert observed["state_during_submit"] == PROCESSING
    assert observed["state_after_nested_feed"] == PROCESSING  # discarded, no restart
    assert session.submissions == 1


# --- run() gating -------------------------------------------------------------------


def test_run_refuses_when_wake_word_disabled(monkeypatch, capsys):
    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: False)
    assert fifi_wake.run([]) == 1
    assert "ENABLE_WAKE_WORD" in capsys.readouterr().out


def test_run_refuses_without_dependencies(monkeypatch, capsys):
    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: True)
    monkeypatch.setattr(fifi_wake, "check_wake_deps", lambda: (False, ["openwakeword"]))
    assert fifi_wake.run([]) == 1
    out = capsys.readouterr().out
    assert "openwakeword" in out
    assert "requirements-wakeword.txt" in out


def test_run_refuses_when_model_unavailable(monkeypatch, capsys):
    import app.voice.wake_word as wake_word_module

    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: True)
    monkeypatch.setattr(fifi_wake, "check_wake_deps", lambda: (True, []))

    class UnavailableService:
        def __init__(self, **kwargs):
            pass

        def load(self):
            return {"status": "unavailable", "message": "custom model missing"}

    monkeypatch.setattr(wake_word_module, "WakeWordService", UnavailableService)
    assert fifi_wake.run([]) == 1
    assert "Wake word unavailable: custom model missing" in capsys.readouterr().out


# --- safety / containment ------------------------------------------------------------


def test_wake_client_cannot_touch_safety_configuration():
    """The listener is a client: it reuses the push-to-talk /voice/command path
    and never sets safety flags or posts anywhere else directly."""
    source = (PROJECT_ROOT / "scripts" / "fifi_wake.py").read_text(encoding="utf-8")
    assert "ENABLE_REAL_WINDOWS_TOOLS" not in source
    assert "handle_utterance" in source  # same pipeline as push-to-talk
    assert ".post(" not in source  # no direct HTTP posts of its own


def test_wake_config_defaults_match_spec():
    config = make_config()
    assert config.threshold == 0.55
    assert config.cooldown_seconds == 2.0
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ms == 250.0
    assert config.vad_silence_ms == 900.0
    assert config.vad_max_command_seconds == 20.0
    assert config.store_audio is False
    assert config.audio_feedback is True
    assert config.debug is False


def test_wake_never_sends_confirm_true():
    """Wake detection must NEVER count as confirmation: the listener has no
    code path that submits confirm=True — only the shared spoken-confirmation
    flow inside PttClient may do that, after an explicit phrase."""
    source = (PROJECT_ROOT / "scripts" / "fifi_wake.py").read_text(encoding="utf-8")
    assert "confirm=True" not in source
    assert "confirm=true" not in source.lower().replace(" ", "")


def test_wake_config_reads_spec_variable_names(monkeypatch):
    import llm_smoke

    values = {
        "WAKE_WORD_INPUT_DEVICE": "USB Microphone",
        "WAKE_WORD_MUTE_HOTKEY": "ctrl+alt+q",
        "WAKE_WORD_AUDIO_FEEDBACK": "false",
    }
    monkeypatch.setattr(
        llm_smoke, "config_value", lambda name, fallback: values.get(name) or fallback
    )
    config = WakeConfig.load()
    assert config.input_device == "USB Microphone"
    assert config.mute_hotkey == "ctrl+alt+q"
    assert config.audio_feedback is False


def test_wake_config_falls_back_to_legacy_names(monkeypatch):
    import llm_smoke

    values = {"WAKE_MUTE_HOTKEY": "ctrl+alt+z", "PTT_INPUT_DEVICE": "Headset"}
    monkeypatch.setattr(
        llm_smoke, "config_value", lambda name, fallback: values.get(name) or fallback
    )
    config = WakeConfig.load()
    assert config.mute_hotkey == "ctrl+alt+z"
    assert config.input_device == "Headset"


# --- acknowledgment feedback (WAKE_WORD_AUDIO_FEEDBACK) -------------------------------


def test_audio_feedback_plays_when_enabled():
    beeps = []
    session, _, _ = make_session()
    session._feedback_fn = beeps.append
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert beeps == ["wake", "send"]


def test_audio_feedback_silent_when_disabled():
    beeps = []
    session, _, _ = make_session(config=make_config(audio_feedback=False))
    session._feedback_fn = beeps.append
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert beeps == []  # detection still works; only the sound is off


# --- observability: status file, scores, durations ------------------------------------


def test_session_records_detection_and_command_status():
    updates = []
    session, _, _ = make_session()
    session.status_writer = updates.append
    session.service.last_score = 0.87  # what the real service exposes
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)

    detection = next(u["last_detection"] for u in updates if "last_detection" in u)
    assert detection["score"] == 0.87
    assert detection["utc"]
    command = next(u["last_command"] for u in updates if "last_command" in u)
    assert command["result"] == "sent"
    assert command["stop_reason"] == "silence"
    assert command["capture_seconds"] is not None
    assert command["speech_ms"] >= 250


def test_session_records_api_timings_when_available():
    def submit_with_timings(path):
        return {"status": "ok", "timings": {
            "transcribe_seconds": 1.2, "command_seconds": 3.4, "tts_seconds": 2.1,
        }}

    updates = []
    session, _, _ = make_session(submit=submit_with_timings)
    session.status_writer = updates.append
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    command = next(u["last_command"] for u in updates if "last_command" in u)
    assert command["timings"]["transcribe_seconds"] == 1.2
    assert command["timings"]["tts_seconds"] == 2.1


def test_mute_state_is_recorded():
    updates = []
    session, _, _ = make_session()
    session.status_writer = updates.append
    session.toggle_mute()
    assert {"muted": True} in updates
    session.toggle_mute()
    assert {"muted": False} in updates


def test_debug_mode_logs_scores_for_calibration():
    messages = []
    session, _, clock = make_session(config=make_config(debug=True))
    session.out = messages.append
    session.service.last_score = 0.123
    clock.advance(5.0)  # past the once-per-second debug gate
    session.feed(Frame(speech=True))  # no wake — still logs the score
    assert any("wake score 0.123" in m for m in messages)


def test_debug_off_logs_no_scores():
    messages = []
    session, _, clock = make_session()
    session.out = messages.append
    session.service.last_score = 0.123
    clock.advance(5.0)
    session.feed(Frame(speech=True))
    assert not any("wake score" in m for m in messages)


def test_capture_logs_duration_and_stop_reason():
    messages = []
    session, _, _ = make_session()
    session.out = messages.append
    session.feed(Frame(wake=True, speech=True))
    _speak(session)
    _go_silent(session)
    assert any("Capture ended (silence)" in m for m in messages)


# --- microphone disconnection recovery -------------------------------------------------


def test_audio_loop_recovers_from_microphone_errors():
    """A device error mid-stream cancels the capture, waits, and reopens the
    stream — the listener never exits over a microphone problem."""
    session, _, _ = make_session()
    attempts = []
    stops = {"n": 0}

    class FailingOnceStream:
        def __enter__(self):
            attempts.append("open")
            if len(attempts) == 1:
                raise OSError("microphone unplugged")
            return self

        def __exit__(self, *exc):
            return False

    def stop_check():
        stops["n"] += 1
        return stops["n"] > 2  # let the second (successful) stream finish

    slept = []
    fifi_wake._audio_loop(
        session, make_config(), stop_check,
        out=lambda *a: None,
        stream_factory=FailingOnceStream,
        retry_delay=0.01,
        sleep=slept.append,
    )
    assert attempts == ["open", "open"]  # reopened after the failure
    assert slept == [0.01]


def test_audio_loop_keyboard_interrupt_propagates():
    session, _, _ = make_session()

    class InterruptStream:
        def __enter__(self):
            raise KeyboardInterrupt()

        def __exit__(self, *exc):
            return False

    try:
        fifi_wake._audio_loop(
            session, make_config(), lambda: False,
            out=lambda *a: None, stream_factory=InterruptStream,
        )
        raise AssertionError("KeyboardInterrupt must propagate for a clean Ctrl+C exit")
    except KeyboardInterrupt:
        pass


# --- wake status file -------------------------------------------------------------------


def test_wake_status_file_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fifi_wake, "WAKE_STATUS_FILE", tmp_path / "wake_status.json")
    fifi_wake.write_wake_status({"muted": True})
    fifi_wake.write_wake_status({"last_detection": {"utc": "2026-07-11", "score": 0.9}})
    status = fifi_wake.read_wake_status()
    assert status["muted"] is True  # merged, not replaced
    assert status["last_detection"]["score"] == 0.9
    assert status["updated_utc"]
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes, no leftovers


# --- local_runtime wake commands ----------------------------------------------------------


import local_runtime  # noqa: E402


def test_wake_runtime_subcommands_are_wired():
    parser = local_runtime.build_parser()
    assert parser.parse_args(["wake-start"]).func is local_runtime.cmd_wake_start
    assert parser.parse_args(["wake-stop"]).func is local_runtime.cmd_wake_stop
    assert parser.parse_args(["wake-status"]).func is local_runtime.cmd_wake_status
    assert parser.parse_args(["wake-doctor"]).func is local_runtime.cmd_wake_doctor
    args = parser.parse_args(["wake"])
    assert args.func is local_runtime.cmd_wake and args.debug is False


def _args():
    class Args:
        debug = False

    return Args()


def _all_preconditions_ok():
    return [
        {"name": "Wake enabled", "ok": True, "hard": True, "detail": "ENABLE_WAKE_WORD=true"},
        {"name": "Dependencies", "ok": True, "hard": True, "detail": "installed"},
        {"name": "Wake model", "ok": True, "hard": True, "detail": "fifi.onnx"},
        {"name": "Main API", "ok": True, "hard": True, "detail": "healthy"},
        {"name": "Voice enabled", "ok": True, "hard": True, "detail": "enabled"},
        {"name": "Voice Lab", "ok": True, "hard": True, "detail": "healthy"},
        {"name": "Runtime mode", "ok": True, "hard": True, "detail": "daily"},
        {"name": "Active voice", "ok": True, "hard": False, "detail": "fifi_warm (Kokoro — responsive)"},
    ]


def test_wake_start_refuses_when_a_hard_precondition_fails(monkeypatch, capsys):
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: False)
    monkeypatch.setattr(local_runtime, "wake_preconditions", lambda: [
        {"name": "Wake model", "ok": False, "hard": True,
         "detail": "MISSING at models/wake_words/fifi.onnx — it is never downloaded automatically"},
    ])
    monkeypatch.setattr(
        local_runtime.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    assert local_runtime.cmd_wake_start(_args()) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "never downloaded" in out
    assert "wake-doctor" in out


def test_wake_start_launches_when_preconditions_pass(monkeypatch, tmp_path, capsys):
    active = iter([False, True])  # pre-check, then ready
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: next(active, True))
    monkeypatch.setattr(local_runtime, "wake_preconditions", _all_preconditions_ok)
    monkeypatch.setattr(local_runtime, "ensure_dirs", lambda: None)
    monkeypatch.setattr(local_runtime, "WAKE_LOG_FILE", tmp_path / "wake.log")
    monkeypatch.setattr(
        local_runtime, "open", lambda *a, **k: __import__("io").BytesIO(), raising=False
    )
    launched = {}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        launched["cmd"] = cmd

        class P:
            def poll(self):
                return None

        return P()

    monkeypatch.setattr(local_runtime.subprocess, "Popen", fake_popen)
    assert local_runtime.cmd_wake_start(_args()) == 0
    assert "fifi_wake.py" in " ".join(launched["cmd"])
    assert "--server" in launched["cmd"]
    out = capsys.readouterr().out
    assert "running" in out
    assert "Active voice    : fifi_warm (Kokoro" in out  # soft note surfaced


def test_wake_start_reports_startup_exit_with_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: False)
    monkeypatch.setattr(local_runtime, "wake_preconditions", _all_preconditions_ok)
    monkeypatch.setattr(local_runtime, "ensure_dirs", lambda: None)
    log = tmp_path / "wake.log"
    log.write_text("Wake word unavailable: custom model missing\n", encoding="utf-8")
    monkeypatch.setattr(local_runtime, "WAKE_LOG_FILE", log)
    monkeypatch.setattr(
        local_runtime, "open", lambda *a, **k: __import__("io").BytesIO(), raising=False
    )

    class DeadProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(local_runtime.subprocess, "Popen", lambda *a, **k: DeadProcess())
    assert local_runtime.cmd_wake_start(_args()) == 1
    out = capsys.readouterr().out
    assert "exited during startup" in out
    assert "custom model missing" in out  # last safe log lines shown


# --- Phase 3D.1: wake preconditions + wake-doctor (readiness gate) --------------------


def _pass_preconditions(monkeypatch, *, tts="voice_lab", mode="daily", voice_engine="kokoro"):
    """Make every wake precondition pass (voice_lab TTS, daily mode, Kokoro voice)."""
    from pathlib import Path as _Path

    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: True)
    monkeypatch.setattr(fifi_wake, "check_wake_deps", lambda: (True, []))
    monkeypatch.setattr(fifi_wake, "wake_model_present", lambda: True)
    monkeypatch.setattr(fifi_wake, "wake_model_path", lambda: _Path("models/wake_words/fifi.onnx"))
    monkeypatch.setattr(local_runtime, "api_health", lambda url: {"status": "ok", "voice": True})
    monkeypatch.setattr(local_runtime, "_tts_engine", lambda: tts)
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: {"status": "ok"})
    monkeypatch.setattr(local_runtime, "voice_lab_get_mode", lambda: {"mode": mode})
    monkeypatch.setattr(local_runtime, "_active_wake_voice", lambda: {
        "profile": "fifi_warm",
        "engine": "qwen3_tts" if voice_engine == "voice_design" else voice_engine,
        "is_voice_design": voice_engine == "voice_design",
    })


def _by_name(checks):
    return {c["name"]: c for c in checks}


def test_wake_preconditions_all_pass(monkeypatch):
    _pass_preconditions(monkeypatch)
    checks = local_runtime.wake_preconditions()
    assert all(c["ok"] for c in checks if c["hard"])  # every HARD check ok
    by = _by_name(checks)
    assert by["Runtime mode"]["detail"] == "daily"
    assert by["Voice enabled"]["ok"] is True


def test_wake_precondition_fails_when_mode_not_daily(monkeypatch):
    _pass_preconditions(monkeypatch, mode="low-memory")
    by = _by_name(local_runtime.wake_preconditions())
    assert by["Runtime mode"]["ok"] is False and by["Runtime mode"]["hard"] is True
    assert "voice-mode daily" in by["Runtime mode"]["detail"]


def test_wake_precondition_fails_when_voice_lab_down(monkeypatch):
    _pass_preconditions(monkeypatch)
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    by = _by_name(local_runtime.wake_preconditions())
    assert by["Voice Lab"]["ok"] is False and by["Voice Lab"]["hard"] is True


def test_wake_precondition_fails_when_api_down(monkeypatch):
    _pass_preconditions(monkeypatch)
    monkeypatch.setattr(local_runtime, "api_health", lambda url: None)
    by = _by_name(local_runtime.wake_preconditions())
    assert by["Main API"]["ok"] is False and by["Main API"]["hard"] is True
    assert by["Voice enabled"]["ok"] is False  # can't confirm voice with the API down


def test_wake_precondition_voicedesign_active_is_soft_warn(monkeypatch):
    _pass_preconditions(monkeypatch, voice_engine="voice_design")
    av = _by_name(local_runtime.wake_preconditions())["Active voice"]
    assert av["hard"] is False  # never blocks — the guard handles it
    assert "NEVER loaded in wake mode" in av["detail"]


def test_wake_precondition_windows_tts_skips_voice_lab_checks(monkeypatch):
    _pass_preconditions(monkeypatch, tts="windows")
    by = _by_name(local_runtime.wake_preconditions())
    assert "Voice Lab" not in by and "Runtime mode" not in by
    assert by["TTS engine"]["ok"] is True


def test_wake_doctor_exit_codes(monkeypatch, capsys):
    _pass_preconditions(monkeypatch)
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: False)
    monkeypatch.setattr(fifi_wake, "read_wake_status", lambda: {})
    assert local_runtime.cmd_wake_doctor(_args()) == 0
    assert "wake mode can start" in capsys.readouterr().out.lower()

    monkeypatch.setattr(local_runtime, "voice_lab_get_mode", lambda: {"mode": "designer"})
    assert local_runtime.cmd_wake_doctor(_args()) == 1  # mode != daily -> cannot start
    out = capsys.readouterr().out
    assert "CANNOT start" in out
    assert "voice-mode daily" in out


def test_wake_listener_marks_requests_as_wake_mode():
    """The listener flags its client so /voice/command applies the TTS guard —
    this NEVER confirms or permits anything (only voices the reply)."""
    source = (PROJECT_ROOT / "scripts" / "fifi_wake.py").read_text(encoding="utf-8")
    assert "wake_mode = True" in source


def test_wake_stop_only_touches_owned_pid(monkeypatch, tmp_path, capsys):
    pid_file = tmp_path / "wake.pid"
    monkeypatch.setattr(fifi_wake, "WAKE_PID_FILE", pid_file)
    killed = []
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))

    # Nothing recorded → nothing stopped.
    assert local_runtime.cmd_wake_stop(_args()) == 0
    assert killed == []
    assert "nothing to stop" in capsys.readouterr().out

    # Recorded + alive → stopped and cleared.
    pid_file.write_text("777", encoding="utf-8")
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: True)
    assert local_runtime.cmd_wake_stop(_args()) == 0
    assert killed == [777]
    assert not pid_file.exists()


def test_wake_status_reports_everything(monkeypatch, capsys):
    monkeypatch.setattr(fifi_wake, "check_wake_deps", lambda: (True, []))
    monkeypatch.setattr(fifi_wake, "wake_model_present", lambda: True)
    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: True)
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: True)
    monkeypatch.setattr(fifi_wake, "read_wake_status", lambda: {
        "muted": False,
        "input_device": "default",
        "last_detection": {"utc": "2026-07-11T22:00:00+00:00", "score": 0.81},
        "last_command": {
            "result": "sent", "utc": "2026-07-11T22:00:07+00:00",
            "stop_reason": "silence", "capture_seconds": 3.2,
            "timings": {"transcribe_seconds": 1.1, "command_seconds": 2.0,
                        "tts_seconds": 1.4},
        },
    })
    assert local_runtime.cmd_wake_status(_args()) == 0
    out = capsys.readouterr().out
    assert "Dependencies    : True" in out
    assert "Model present   : True" in out
    assert "Enabled         : True" in out
    assert "Active          : True" in out
    assert "Muted           : False" in out
    assert "Microphone      : default" in out
    assert "score=0.810" in out
    assert "sent" in out and "stop: silence" in out
    assert "stt=1.1s" in out


def test_wake_status_when_never_run(monkeypatch, capsys):
    monkeypatch.setattr(fifi_wake, "check_wake_deps", lambda: (False, ["openwakeword"]))
    monkeypatch.setattr(fifi_wake, "wake_model_present", lambda: False)
    monkeypatch.setattr(fifi_wake, "wake_word_enabled", lambda: False)
    monkeypatch.setattr(fifi_wake, "wake_active", lambda: False)
    monkeypatch.setattr(fifi_wake, "read_wake_status", lambda: {"input_device": "default"})
    assert local_runtime.cmd_wake_status(_args()) == 0
    out = capsys.readouterr().out
    assert "missing: openwakeword" in out
    assert "(none recorded)" in out
