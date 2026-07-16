"""Fifi wake-word listener (Phase 3D.1) — Windows host only.

Say "Fifi", then speak one command. The listener runs a strict state machine:

    IDLE -> WAKE_DETECTED -> CAPTURING -> PROCESSING -> SPEAKING -> COOLDOWN -> IDLE

- IDLE: frames are scored by the custom openWakeWord model, gated by Silero
  VAD (non-speech can never trigger). Idle audio only ever lives in the wake
  model's short in-memory rolling context — it is NEVER buffered or written.
- CAPTURING: frames are buffered until VAD_SILENCE_MS of silence (after at
  least VAD_MIN_SPEECH_MS of speech) or VAD_MAX_COMMAND_SECONDS.
- PROCESSING: exactly ONE WAV is sent to /voice/command through the same
  client logic as push-to-talk (spoken confirmation window, TTS reply). The
  temporary WAV is deleted afterwards unless WAKE_WORD_STORE_AUDIO=true.
- SPEAKING: the API's spoken reply (selected Voice Lab profile) — frames are
  still discarded; a new wake word cannot interrupt the response.
- COOLDOWN: for WAKE_WORD_COOLDOWN_SECONDS all audio is discarded, then IDLE.

No overlapping recordings or requests: the state machine is single-threaded
and a capture can only start from IDLE. Esc cancels a capture, the mute hotkey
(WAKE_WORD_MUTE_HOTKEY, default ctrl+alt+m) pauses listening, Ctrl+C exits.
An API or TTS failure is logged and the listener keeps running; a microphone
disconnect is retried until the device returns.

This is a *client*: it cannot enable real Windows tools, cannot bypass
confirmations, and only ever sends audio through /voice/command. The custom
wake model is never downloaded — missing model means "unavailable", period.

Run it via:  python scripts/local_runtime.py wake        (foreground)
         or  python scripts/local_runtime.py wake-start  (background)
Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import fifi_ptt  # noqa: E402  (reused /voice/command client + startup validation)
import llm_smoke  # noqa: E402  (reused .env/env config reader)

SAMPLE_RATE = 16000
# 80 ms frames (1280 samples) — openWakeWord's expected streaming chunk.
FRAME_SAMPLES = 1280

WAKE_PID_FILE = PROJECT_ROOT / "storage" / "runtime" / "wake.pid"
# Observability for `wake-status`: muted flag, last detection, last command
# result. Written atomically; contains no idle audio and no secrets.
WAKE_STATUS_FILE = PROJECT_ROOT / "storage" / "runtime" / "wake_status.json"

# States
IDLE = "IDLE"
WAKE_DETECTED = "WAKE_DETECTED"
CAPTURING = "CAPTURING"
PROCESSING = "PROCESSING"
SPEAKING = "SPEAKING"
COOLDOWN = "COOLDOWN"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- configuration (env / .env, same source as the other scripts) ----------------


class WakeConfig:
    def __init__(
        self,
        server: str,
        threshold: float,
        cooldown_seconds: float,
        vad_threshold: float,
        vad_min_speech_ms: float,
        vad_silence_ms: float,
        vad_max_command_seconds: float,
        store_audio: bool,
        mute_hotkey: str,
        input_device: str,
        model_path: str,
        audio_feedback: bool = True,
        debug: bool = False,
    ) -> None:
        self.audio_feedback = audio_feedback
        self.debug = debug
        self.server = server
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_silence_ms = vad_silence_ms
        self.vad_max_command_seconds = vad_max_command_seconds
        self.store_audio = store_audio
        self.mute_hotkey = mute_hotkey
        self.input_device = input_device
        self.model_path = model_path

    @staticmethod
    def _bool(name: str, fallback: bool) -> bool:
        raw = llm_smoke.config_value(name, "true" if fallback else "false")
        return raw.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _float(name: str, fallback: float) -> float:
        try:
            return float(llm_smoke.config_value(name, str(fallback)))
        except ValueError:
            return fallback

    @classmethod
    def load(cls, server: str | None = None) -> "WakeConfig":
        return cls(
            server=server or fifi_ptt.DEFAULT_SERVER,
            threshold=cls._float("WAKE_WORD_THRESHOLD", 0.55),
            cooldown_seconds=cls._float("WAKE_WORD_COOLDOWN_SECONDS", 2.0),
            vad_threshold=cls._float("VAD_THRESHOLD", 0.5),
            vad_min_speech_ms=cls._float("VAD_MIN_SPEECH_MS", 250.0),
            vad_silence_ms=cls._float("VAD_SILENCE_MS", 900.0),
            vad_max_command_seconds=cls._float("VAD_MAX_COMMAND_SECONDS", 20.0),
            store_audio=cls._bool("WAKE_WORD_STORE_AUDIO", False),
            # WAKE_WORD_MUTE_HOTKEY is the canonical name; WAKE_MUTE_HOTKEY is
            # accepted for backward compatibility with earlier .env files.
            mute_hotkey=llm_smoke.config_value(
                "WAKE_WORD_MUTE_HOTKEY",
                llm_smoke.config_value("WAKE_MUTE_HOTKEY", "ctrl+alt+m"),
            ),
            # Wake has its own device setting; falls back to the push-to-talk one.
            input_device=llm_smoke.config_value(
                "WAKE_WORD_INPUT_DEVICE",
                llm_smoke.config_value("PTT_INPUT_DEVICE", "default"),
            ),
            model_path=llm_smoke.config_value(
                "WAKE_WORD_MODEL_PATH", "models/wake_words/fifi.onnx"
            ),
            audio_feedback=cls._bool("WAKE_WORD_AUDIO_FEEDBACK", True),
        )


def wake_word_enabled() -> bool:
    return WakeConfig._bool("ENABLE_WAKE_WORD", False)


# --- dependency / liveness helpers ------------------------------------------------


def check_wake_deps() -> tuple[bool, list[str]]:
    """(ok, missing) for wake + desktop audio dependencies. Never raises."""
    from app.voice.wake_word import check_wake_deps as _service_deps

    ok_service, missing = _service_deps()
    ok_desktop, missing_desktop = fifi_ptt.check_desktop_deps()
    return (ok_service and ok_desktop, missing + missing_desktop)


def wake_model_path() -> Path:
    raw = Path(WakeConfig.load().model_path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def wake_model_present() -> bool:
    return wake_model_path().is_file()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return False
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wake_active() -> bool:
    """Whether a wake listener started by this script is currently running."""
    if not WAKE_PID_FILE.exists():
        return False
    try:
        pid = int(WAKE_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    return _pid_alive(pid)


def _write_wake_pid() -> None:
    WAKE_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    WAKE_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _clear_wake_pid() -> None:
    WAKE_PID_FILE.unlink(missing_ok=True)


def write_wake_status(update: dict) -> None:
    """Merge + atomically write wake_status.json for `wake-status`.

    Contains flags, timestamps, scores, and result summaries only — never idle
    audio, raw frames, or secrets."""
    current = read_wake_status()
    current.update(update)
    current["updated_utc"] = _utc()
    try:
        WAKE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".wake-", suffix=".json.tmp", dir=WAKE_STATUS_FILE.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2, ensure_ascii=False)
            os.replace(tmp, WAKE_STATUS_FILE)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass  # observability must never take the listener down


def read_wake_status() -> dict:
    try:
        return json.loads(WAKE_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _frame_ms(frame) -> float:
    samples = getattr(frame, "size", None) or len(frame)
    return samples / (SAMPLE_RATE / 1000.0)


# --- the state machine -------------------------------------------------------------


class WakeSession:
    """Single-threaded wake state machine. Feed() one 16 kHz mono int16 frame
    at a time; audio outside CAPTURING is never stored anywhere."""

    def __init__(
        self,
        config: WakeConfig,
        service,
        submit,
        clock=time.monotonic,
        out=print,
        feedback=None,
        status_writer=None,
    ) -> None:
        self.config = config
        self.service = service  # WakeWordService (detect/is_speech/reset)
        self.submit = submit  # callable(wav_path) -> response|None; ONE request
        self.clock = clock
        self.out = out
        self._feedback_fn = feedback or (lambda kind: None)
        self.status_writer = status_writer or (lambda update: None)
        self.state = IDLE
        self.muted = False
        self.transitions: list[str] = [IDLE]
        self.submissions = 0
        self._frames: list = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._capture_started = 0.0
        self._cooldown_until = 0.0
        self._debug_last_print = 0.0

    def _enter(self, state: str) -> None:
        self.state = state
        self.transitions.append(state)

    def feedback(self, kind: str) -> None:
        """Optional acknowledgment sound — gated by WAKE_WORD_AUDIO_FEEDBACK."""
        if self.config.audio_feedback:
            self._feedback_fn(kind)

    # -- feed one frame -------------------------------------------------------------

    def feed(self, frame) -> None:
        if self.muted:
            return  # discarded — nothing is scored or stored while muted

        if self.state == COOLDOWN:
            if self.clock() < self._cooldown_until:
                return  # discard: no retriggering during cooldown
            self.service.reset()
            self._enter(IDLE)

        if self.state == IDLE:
            # Score-and-forget: the frame is never appended to any buffer (the
            # wake model keeps only its own short rolling context in memory).
            detected = self.service.detect(frame)
            score = getattr(self.service, "last_score", None)
            if self.config.debug and score is not None:
                # Threshold calibration: print the live score once per second.
                if self.clock() - self._debug_last_print >= 1.0:
                    self.out(
                        f"[debug] wake score {score:.3f} "
                        f"(threshold {self.config.threshold:g})"
                    )
                    self._debug_last_print = self.clock()
            if detected:
                self._enter(WAKE_DETECTED)
                self.feedback("wake")
                detected_at = _utc()
                score_text = f" (score {score:.3f})" if score is not None else ""
                self.out(
                    f"Wake word detected at {detected_at}{score_text} — "
                    "listening for a command…"
                )
                self.status_writer({
                    "last_detection": {"utc": detected_at, "score": score},
                })
                self._frames = []
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._capture_started = self.clock()
                self._enter(CAPTURING)
            return

        if self.state == CAPTURING:
            self._capture(frame)
            return

        # PROCESSING / SPEAKING (or an unexpected state): discard — no overlap.

    def _capture(self, frame) -> None:
        self._frames.append(frame)
        if self.service.is_speech(frame):
            self._speech_ms += _frame_ms(frame)
            self._silence_ms = 0.0
        else:
            self._silence_ms += _frame_ms(frame)

        elapsed = self.clock() - self._capture_started
        if self._silence_ms >= self.config.vad_silence_ms:
            self._finish_capture(reason="silence")
        elif elapsed >= self.config.vad_max_command_seconds:
            self.out(
                f"Max command duration ({self.config.vad_max_command_seconds:g}s) reached."
            )
            self._finish_capture(reason="max_duration")

    def _finish_capture(self, reason: str) -> None:
        self._enter(PROCESSING)
        frames, self._frames = self._frames, []
        capture_seconds = round(self.clock() - self._capture_started, 2)
        self.out(
            f"Capture ended ({reason}) — {capture_seconds:g}s recorded, "
            f"{self._speech_ms:.0f}ms speech."
        )
        record: dict = {
            "utc": _utc(),
            "stop_reason": reason,
            "capture_seconds": capture_seconds,
            "speech_ms": round(self._speech_ms),
        }
        try:
            if self._speech_ms < self.config.vad_min_speech_ms:
                self.out(
                    f"No speech after the wake word (heard {self._speech_ms:.0f}ms, "
                    f"need ≥ {self.config.vad_min_speech_ms:g}ms) — nothing sent."
                )
                record["result"] = "no_speech"
            else:
                self.feedback("send")
                path = self._write_wav(frames)
                started = self.clock()
                try:
                    self.submissions += 1
                    # Exactly one request per capture. An API/TTS failure is
                    # logged and the listener KEEPS running — never terminate
                    # wake listening over a request problem.
                    try:
                        response = self.submit(path)
                    except Exception as exc:
                        self.out(f"Request failed ({exc}) — still listening for 'Fifi'.")
                        record["result"] = "error"
                        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    else:
                        record["result"] = "sent"
                        record["request_seconds"] = round(self.clock() - started, 2)
                        timings = (response or {}).get("timings") if isinstance(response, dict) else None
                        if timings:
                            record["timings"] = timings
                            self.out(
                                "Durations   : "
                                f"transcription {timings.get('transcribe_seconds', '?')}s · "
                                f"command {timings.get('command_seconds', '?')}s · "
                                f"TTS {timings.get('tts_seconds', '?')}s"
                            )
                        # The spoken reply (selected Voice Lab profile) plays as
                        # part of the request; represent it as its own state so
                        # a wake word can never interrupt the response.
                        self._enter(SPEAKING)
                finally:
                    if not self.config.store_audio:
                        _delete_quietly(path)
        finally:
            frames.clear()
            self.service.reset()
            self.status_writer({"last_command": record})
            self._start_cooldown()

    def cancel(self) -> None:
        """Esc: discard the current capture without sending anything."""
        if self.state not in (WAKE_DETECTED, CAPTURING):
            return
        self._frames = []
        self.service.reset()
        self.out("Capture cancelled — nothing sent.")
        self._start_cooldown()

    def toggle_mute(self) -> None:
        if not self.muted and self.state in (WAKE_DETECTED, CAPTURING):
            self.cancel()
        self.muted = not self.muted
        self.service.reset()
        self.status_writer({"muted": self.muted})
        self.out("Muted — say nothing, I'm not listening." if self.muted else "Unmuted — listening for 'Fifi'.")

    def _start_cooldown(self) -> None:
        self._cooldown_until = self.clock() + self.config.cooldown_seconds
        self._enter(COOLDOWN)

    def _write_wav(self, frames) -> str:
        import numpy as np
        import soundfile as sf

        audio = np.concatenate([np.asarray(f, dtype="int16").reshape(-1, 1) for f in frames])
        if self.config.store_audio:
            store_dir = PROJECT_ROOT / "storage" / "audio"
            store_dir.mkdir(parents=True, exist_ok=True)
            name = str(store_dir / f"wake_{time.strftime('%Y%m%d_%H%M%S')}.wav")
        else:
            fd, name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        sf.write(name, audio, SAMPLE_RATE)
        return name


def _delete_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# --- microphone loop + entry point --------------------------------------------------


def _stream_frames(session: WakeSession, config: WakeConfig, stop_check, stream_factory=None) -> None:
    """One microphone stream session: 80 ms frames into the state machine."""
    frames: queue.Queue = queue.Queue(maxsize=64)

    def _callback(indata, frame_count, time_info, status):  # noqa: ANN001
        try:
            frames.put_nowait(indata.copy())
        except queue.Full:
            pass  # processing backlog: drop mic frames rather than queue them

    if stream_factory is None:
        import sounddevice as sd

        device = None if config.input_device in ("", "default") else config.input_device

        def stream_factory():
            return sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                device=device,
                callback=_callback,
            )

    with stream_factory():
        while not stop_check():
            try:
                frame = frames.get(timeout=0.25)
            except queue.Empty:
                continue
            session.feed(frame)


def _audio_loop(
    session: WakeSession,
    config: WakeConfig,
    stop_check,
    out=print,
    stream_factory=None,
    retry_delay: float = 2.0,
    sleep=time.sleep,
) -> None:
    """Microphone loop with disconnect recovery: a device error cancels any
    in-flight capture, waits, and reopens the stream — it never exits the
    listener. Only Ctrl+C (KeyboardInterrupt) stops it."""
    while not stop_check():
        try:
            _stream_frames(session, config, stop_check, stream_factory)
            return  # stop_check asked us to finish
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            session.cancel()
            out(
                f"Microphone error: {exc} — retrying in {retry_delay:g}s "
                "(Ctrl+C to exit)."
            )
            sleep(retry_delay)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fifi wake-word listener (Windows host).")
    parser.add_argument(
        "--server",
        default=fifi_ptt.DEFAULT_SERVER,
        help=f"API base URL (default {fifi_ptt.DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print live wake scores once per second for threshold calibration",
    )
    args = parser.parse_args(argv)

    if not wake_word_enabled():
        print(
            "Wake word listening is disabled. Set ENABLE_WAKE_WORD=true in .env "
            "to opt in (push-to-talk remains available either way)."
        )
        return 1

    deps_ok, missing = check_wake_deps()
    if not deps_ok:
        print(
            "Missing wake-word dependencies: " + ", ".join(missing) + "\n"
            "Install them on the Windows host:\n"
            "  .venv\\Scripts\\python.exe -m pip install -r requirements-wakeword.txt"
        )
        return 1

    from app.voice.wake_word import WakeWordService

    config = WakeConfig.load(server=args.server)
    config.debug = bool(args.debug)
    service = WakeWordService(
        model_path=config.model_path,
        threshold=config.threshold,
        vad_threshold=config.vad_threshold,
    )
    loaded = service.load()  # never downloads; missing model => unavailable
    if loaded.get("status") != "ok":
        print(f"Wake word unavailable: {loaded.get('message')}")
        return 1

    # Same startup validation + /voice/command client as push-to-talk: identity
    # check, voice enabled, STT preflight, spoken confirmation window, TTS reply.
    ptt_config = fifi_ptt.PttConfig.load(server=config.server)
    http = httpx.Client(base_url=config.server)
    try:
        if not fifi_ptt.validate_startup(ptt_config, http):
            return 1
    except Exception as exc:  # defensive — never crash on a bad response
        print(f"Startup validation failed: {exc}")
        http.close()
        return 1

    client = fifi_ptt.PttClient(ptt_config, http=http)
    # Wake requests are hands-free: mark them so /voice/command voices the reply
    # with the daily latency guard (Qwen->Kokoro, never VoiceDesign). This only
    # changes how the reply is spoken — it never confirms or permits anything.
    client.wake_mode = True

    def _feedback(kind: str) -> None:
        try:
            import winsound

            winsound.Beep(988 if kind == "wake" else 660, 90)
        except Exception:
            print("\a", end="")

    session = WakeSession(
        config,
        service,
        submit=client.handle_utterance,
        feedback=_feedback,
        status_writer=write_wake_status,
    )

    import keyboard

    keyboard.add_hotkey("esc", session.cancel)
    keyboard.add_hotkey(config.mute_hotkey, session.toggle_mute)

    _write_wake_pid()
    write_wake_status({
        "muted": False,
        "input_device": config.input_device,
        "threshold": config.threshold,
        "debug": config.debug,
        "started_utc": _utc(),
        "pid": os.getpid(),
    })
    print(
        f"Fifi wake listener ready — say 'Fifi' to talk. "
        f"Esc cancels a capture, {config.mute_hotkey} mutes, Ctrl+C exits."
    )
    try:
        _audio_loop(session, config, stop_check=lambda: False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        client.close()
        _clear_wake_pid()
        print("\nFifi wake listener stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
