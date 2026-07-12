"""Fifi push-to-talk client (Phase 3C) — Windows host only.

Hold a global hotkey (default Ctrl+Alt+Space), speak, release. The clip is
recorded as 16 kHz mono WAV, sent to POST /voice/command, and the transcription,
intent, safety status, planner path and Fifi's reply are printed. The API speaks
the reply when VOICE_SPEAK_COMMAND_RESPONSE=true.

This is a *client*. It never touches the safety layer: it cannot enable real
Windows tools, cannot unblock destructive actions, and only ever sends spoken
text through the exact same /voice/command → handle_command pipeline as typed
commands. Sensitive actions require an explicit, short-lived spoken confirmation.

There is no always-on microphone and no wake word — the mic is live only while
the key is held. Run it via:  python scripts/local_runtime.py ptt

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import os
import tempfile
import threading
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(SCRIPTS_DIR))

import llm_smoke  # noqa: E402  (reused .env/env config reader)

DEFAULT_SERVER = "http://127.0.0.1:8000"
SAMPLE_RATE = 16000  # 16 kHz mono, matches faster-whisper input

# Exact confirmation / cancellation phrases (NOT fuzzy-matched). Transcriptions
# are lowercased with surrounding punctuation stripped before comparison.
CONFIRM_PHRASES = {
    "confirm",
    "confirmar",
    "yes confirm",
    "si confirmar",
    "sí confirmar",
}
CANCEL_PHRASES = {"cancel", "cancelar", "no"}


def _normalize(text: str) -> str:
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(cleaned.split())


# --- configuration (env / .env, same source as the other scripts) ----------------


class PttConfig:
    def __init__(
        self,
        server: str,
        hotkey: str,
        max_seconds: float,
        min_seconds: float,
        confirm_window: float,
        audio_feedback: bool,
        input_device: str,
        speak_responses: bool,
        connect_timeout: float = 5.0,
        process_timeout: float = 180.0,
    ) -> None:
        self.server = server
        self.hotkey = hotkey
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        self.confirm_window = confirm_window
        self.audio_feedback = audio_feedback
        self.input_device = input_device
        self.speak_responses = speak_responses
        # Connection setup is quick; voice processing (STT on the host) can be
        # slow. Keep them separate so a slow transcription is never misreported
        # as an unreachable API.
        self.connect_timeout = connect_timeout
        self.process_timeout = process_timeout

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
    def load(cls, server: str | None = None) -> "PttConfig":
        return cls(
            server=server or DEFAULT_SERVER,
            hotkey=llm_smoke.config_value("PTT_HOTKEY", "ctrl+alt+space"),
            max_seconds=cls._float("PTT_MAX_SECONDS", 20.0),
            min_seconds=cls._float("PTT_MIN_SECONDS", 0.4),
            confirm_window=cls._float("PTT_CONFIRM_WINDOW_SECONDS", 30.0),
            audio_feedback=cls._bool("PTT_AUDIO_FEEDBACK", True),
            input_device=llm_smoke.config_value("PTT_INPUT_DEVICE", "default"),
            speak_responses=cls._bool("VOICE_SPEAK_COMMAND_RESPONSE", False),
            connect_timeout=cls._float("PTT_CONNECT_TIMEOUT", 5.0),
            process_timeout=cls._float("PTT_PROCESS_TIMEOUT", 180.0),
        )


# --- dependency + startup checks -------------------------------------------------


def check_desktop_deps() -> tuple[bool, list[str]]:
    """(ok, missing) for the optional desktop dependencies. Never raises."""
    missing: list[str] = []
    for module in ("keyboard", "sounddevice", "soundfile", "numpy"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    return (not missing, missing)


def validate_startup(config: "PttConfig", http, out=print) -> bool:
    """Confirm the API is agent-automaton and voice is enabled. Warn-only for
    the LLM planner / spoken responses. Never changes any configuration."""
    try:
        health_resp = http.get("/health", timeout=5.0)
    except Exception as exc:
        out(
            f"API not reachable at {config.server}: {exc}\n"
            "Start it first:  python scripts/local_runtime.py start"
        )
        return False
    if health_resp.status_code != 200:
        out(f"/health returned HTTP {health_resp.status_code} — is the API healthy?")
        return False
    health = health_resp.json()
    if health.get("app") != "agent-automaton":
        out(f"Refusing to start: API is not agent-automaton (got {health.get('app')!r}).")
        return False

    try:
        identity = http.get("/identity", timeout=5.0).json()
    except Exception:
        identity = {}
    project = identity.get("project_name")
    if project and project != "agent-automaton":
        out(f"Refusing to start: /identity reports {project!r}, not agent-automaton.")
        return False

    if not health.get("voice"):
        out(
            "Voice is disabled on the API. Set ENABLE_VOICE=true in .env and restart\n"
            "the API (push-to-talk needs the /voice endpoints)."
        )
        return False

    if not health.get("llm_planner"):
        out("Note: LLM planner is disabled — the rule router will handle commands.")
    if not config.speak_responses:
        out("Note: spoken responses are off (VOICE_SPEAK_COMMAND_RESPONSE=false) — text only.")

    # STT preflight: load the model and run a harmless probe before we accept
    # any speech. A broken CUDA/STT setup fails here, not mid-command.
    return _run_preflight(config, http, out)


def _run_preflight(config: "PttConfig", http, out=print) -> bool:
    try:
        response = http.get("/voice/preflight", timeout=config.process_timeout)
    except Exception as exc:
        out(f"STT preflight request failed: {exc}")
        return False
    try:
        report = response.json()
    except Exception:
        out("STT preflight returned an unreadable response.")
        return False
    out(
        "STT preflight — device={device} compute={compute_type} model={model} "
        "gpu={gpu} ct2={ctranslate2_version}".format(
            device=report.get("device"),
            compute_type=report.get("compute_type"),
            model=report.get("model"),
            gpu=report.get("gpu"),
            ctranslate2_version=report.get("ctranslate2_version"),
        )
    )
    if report.get("cpu_fallback_active"):
        out("STT preflight — WARNING: CPU fallback is ACTIVE (slow inference).")
    if report.get("passed"):
        out("STT preflight — passed.")
        return True
    out(f"STT preflight — FAILED: {report.get('message')}")
    return False


# --- one-shot pending confirmation (in memory only; never persisted) -------------


class PendingConfirmation:
    """At most ONE pending sensitive command, short-lived, exact-phrase only.

    Not persisted anywhere — a process restart clears it. A new/different command
    clears it too (the client calls clear() before handling a normal command).
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._text: str | None = None
        self._language: str | None = None
        self._expiry: float = 0.0

    def set(self, text: str, language: str | None, window: float) -> None:
        self._text = text
        self._language = language
        self._expiry = self._clock() + window

    def valid(self) -> bool:
        return self._text is not None and self._clock() < self._expiry

    def clear(self) -> None:
        self._text = None
        self._language = None
        self._expiry = 0.0

    @property
    def text(self) -> str | None:
        return self._text

    @property
    def language(self) -> str | None:
        return self._language


# --- audio recorder (sounddevice) ------------------------------------------------


class SoundDeviceRecorder:
    """Streams mono int16 audio at 16 kHz while active. Constructed per-recording
    so a mic disconnection can never poison the next attempt."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: str = "default") -> None:
        self.sample_rate = sample_rate
        self.device = None if device in ("", "default") else device
        self._frames: list = []
        self._stream = None
        self._start_time = 0.0

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=_callback,
        )
        self._stream.start()
        self._start_time = time.monotonic()

    def stop(self):
        import numpy as np

        duration = time.monotonic() - self._start_time
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._frames:
            audio = np.concatenate(self._frames, axis=0)
        else:
            import numpy

            audio = numpy.zeros((0, 1), dtype="int16")
        return audio, duration

    def discard(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._frames = []


# --- push-to-talk client ---------------------------------------------------------


class PttClient:
    def __init__(
        self,
        config: PttConfig,
        http=None,
        recorder_factory=None,
        clock=time.monotonic,
        out=print,
    ) -> None:
        self.config = config
        self.http = http if http is not None else httpx.Client(base_url=config.server)
        self.recorder_factory = recorder_factory or (
            lambda: SoundDeviceRecorder(SAMPLE_RATE, config.input_device)
        )
        self.clock = clock
        self.out = out
        self.pending = PendingConfirmation(clock=clock)
        self._recorder = None
        self._recording = False
        self._lock = threading.Lock()
        self._max_timer: threading.Timer | None = None
        # Set by register_hotkeys() to the listener's latch, so a stop that is
        # NOT triggered by a key release (max-duration, cancel, error) still
        # forces "release all keys before recording again".
        self.latch_hook = None

    def _latch(self) -> None:
        if self.latch_hook is not None:
            self.latch_hook()

    # -- recording lifecycle ------------------------------------------------------

    def begin_recording(self) -> bool:
        """Start capturing. Debounced + concurrency-guarded: a repeat key event
        or a second press while already recording is ignored."""
        with self._lock:
            if self._recording:
                return False
            self._recording = True
        try:
            self._recorder = self.recorder_factory()
            self._recorder.start()
        except Exception as exc:
            with self._lock:
                self._recording = False
            self._recorder = None
            self._latch()  # require a full key release before retrying
            self.out(f"Microphone unavailable: {exc}")
            return False
        self._feedback("start")
        self._arm_max_timer()
        self.out("Listening… (release the key to send, Esc to cancel)")
        return True

    def finish_recording(self) -> None:
        """Stop capturing and submit, unless the clip is too short."""
        recorder = self._take_recorder()
        if recorder is None:
            return
        self._feedback("stop")
        try:
            audio, duration = recorder.stop()
        except Exception as exc:
            self.out(f"Microphone error while stopping: {exc}")
            return
        if duration < self.config.min_seconds:
            self.out(f"Too short ({duration:.2f}s) — need ≥ {self.config.min_seconds:g}s. Ignored.")
            return
        path = self._save_temp_wav(audio)
        try:
            self.handle_utterance(path)
        finally:
            _delete_quietly(path)

    def cancel_recording(self) -> None:
        """Escape: stop and discard the current recording without submitting."""
        recorder = self._take_recorder()
        if recorder is None:
            return
        try:
            recorder.discard()
        except Exception:
            pass
        self._latch()  # require a full key release before recording again
        self.out("Recording cancelled.")

    def _take_recorder(self):
        """Atomically end the recording state and hand back the recorder (or None
        if we weren't recording). Cancels the max-duration timer."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
        self._cancel_max_timer()
        recorder = self._recorder
        self._recorder = None
        return recorder

    def _on_max_duration(self) -> None:
        if self._recording:
            self.out(f"Max duration ({self.config.max_seconds:g}s) reached — sending.")
            # Latch until every hotkey key is physically released, so a still-held
            # combo does not immediately start a fresh recording.
            self._latch()
            self.finish_recording()

    def _arm_max_timer(self) -> None:
        timer = threading.Timer(self.config.max_seconds, self._on_max_duration)
        timer.daemon = True
        self._max_timer = timer
        timer.start()

    def _cancel_max_timer(self) -> None:
        if self._max_timer is not None:
            self._max_timer.cancel()
            self._max_timer = None

    # -- audio helpers ------------------------------------------------------------

    def _save_temp_wav(self, audio) -> str:
        import soundfile as sf

        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(name, audio, SAMPLE_RATE)
        return name

    def _feedback(self, kind: str) -> None:
        if not self.config.audio_feedback:
            return
        try:
            import winsound

            winsound.Beep(880 if kind == "start" else 440, 90)
        except Exception:
            # Non-Windows or no sound device: a terminal bell is enough.
            self.out("\a")

    # -- HTTP ---------------------------------------------------------------------

    def _timeout(self):
        # Short connect timeout, long read/processing timeout — a slow STT is
        # NOT an unreachable API.
        return httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.process_timeout,
            write=30.0,
            pool=30.0,
        )

    def _post_voice_command(self, wav_path: str, confirm: bool):
        try:
            with open(wav_path, "rb") as audio:
                response = self.http.post(
                    "/voice/command",
                    params={"confirm": confirm},
                    files={"file": (os.path.basename(wav_path), audio, "audio/wav")},
                    timeout=self._timeout(),
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self.out(f"API not reachable at {self.config.server}: {exc}")
            return None
        except httpx.ReadTimeout:
            self.out(
                "Voice processing timed out (the API is up, but transcription took "
                f"longer than {self.config.process_timeout:g}s)."
            )
            return None
        except Exception as exc:
            self.out(f"Request failed: {exc}")
            return None
        if response.status_code != 200:
            self.out(f"HTTP {response.status_code}: {response.text}")
            return None
        return response.json()

    def _post_text_command(self, text: str, confirm: bool, language: str | None):
        try:
            response = self.http.post(
                "/command",
                json={"text": text, "confirm": confirm, "language": language},
                timeout=self._timeout(),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self.out(f"API not reachable at {self.config.server}: {exc}")
            return None
        except httpx.ReadTimeout:
            self.out(f"Command timed out after {self.config.process_timeout:g}s.")
            return None
        except Exception as exc:
            self.out(f"Request failed: {exc}")
            return None
        if response.status_code != 200:
            self.out(f"HTTP {response.status_code}: {response.text}")
            return None
        return response.json()

    # -- core: turn one utterance into a (possibly confirmed) command -------------

    def handle_utterance(self, wav_path: str):
        """Send one WAV through /voice/command. Returns the API payload (or
        None) so callers — e.g. the wake listener — can log real durations."""
        response = self._post_voice_command(wav_path, confirm=False)
        if response is None:
            return None
        if response.get("status") != "ok":
            self.out(f"[{response.get('status')}] {response.get('message')}")
            return response

        transcription = (response.get("transcription") or "").strip()
        language = response.get("language")
        phrase = _normalize(transcription)

        # 1) Explicit confirmation of the one pending sensitive command.
        if phrase in CONFIRM_PHRASES and self.pending.valid():
            original, original_lang = self.pending.text, self.pending.language
            self.pending.clear()
            self.out(f"Confirming: {original!r}")
            confirmed = self._post_text_command(original, confirm=True, language=original_lang)
            if confirmed is not None:
                self._report(original, original_lang, confirmed, note="confirmed")
            return confirmed

        # 2) Explicit cancellation / decline.
        if phrase in CANCEL_PHRASES:
            had_pending = self.pending.valid()
            self.pending.clear()
            self.out("Cancelled the pending action." if had_pending else "Nothing to cancel.")
            return response

        # 3) A normal command. Any different command clears a prior pending one.
        self.pending.clear()
        command = response.get("command") or {}
        self._report(
            transcription,
            language,
            command,
            assistant_message=response.get("assistant_message"),
            speech=response.get("speech"),
        )
        if command.get("status") == "needs_confirmation":
            self.pending.set(transcription, language, self.config.confirm_window)
            self.out(
                f"Say 'confirm' within {self.config.confirm_window:g}s to approve, "
                "or 'cancel' to drop it."
            )
        return response

    def _report(self, transcription, language, command, assistant_message=None, speech=None, note=None):
        self.out("")
        if note:
            self.out(f"[{note}]")
        self.out(f"You said   : {transcription!r}  (language: {language})")
        message = assistant_message or command.get("assistant_message") or command.get("message")
        self.out(f"Fifi says  : {message}")
        self.out(f"Planner    : {command.get('planner')}")
        self.out(f"Intent     : {command.get('intent')}")
        self.out(f"Safety     : {command.get('safety_level')} -> {command.get('status')}")
        if command.get("result"):
            self.out(f"Result     : {command['result']}")
        if speech:
            self.out(f"Speech     : {speech}")

    def close(self) -> None:
        """Stop any active recording and release the HTTP client."""
        self.cancel_recording()
        self._cancel_max_timer()
        try:
            self.http.close()
        except Exception:
            pass


def _delete_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# --- hotkey wiring + entry point -------------------------------------------------


_MODIFIER_ALIASES = {
    "left ctrl": "ctrl", "right ctrl": "ctrl", "control": "ctrl",
    "left alt": "alt", "right alt": "alt", "alt gr": "alt",
    "left shift": "shift", "right shift": "shift",
    "left windows": "windows", "right windows": "windows", "cmd": "windows", "win": "windows",
}


def _canonical_key(name: str) -> str:
    name = (name or "").strip().lower()
    return _MODIFIER_ALIASES.get(name, name)


def parse_hotkey(hotkey: str) -> frozenset[str]:
    """"ctrl+alt+space" -> {ctrl, alt, space}; "f8" -> {f8}."""
    return frozenset(
        _canonical_key(part) for part in (hotkey or "").split("+") if part.strip()
    )


class HotkeyListener:
    """Push-to-talk hotkey modelled as individual keys, not a combined string.

    - Recording starts ONCE when every required key is down.
    - It stops as soon as ANY required key is released.
    - Key-repeat (auto-repeat "down" events) is ignored.
    - After a stop, a re-arm requires ALL required keys to be released first —
      so a still-held combo (e.g. after a max-duration stop) cannot immediately
      start another recording.
    """

    def __init__(self, hotkey: str, on_activate, on_release_any) -> None:
        self.required = parse_hotkey(hotkey)
        self._on_activate = on_activate
        self._on_release_any = on_release_any
        self._pressed: set[str] = set()
        self._active = False
        self._latched = False

    def on_event(self, name: str, is_down: bool) -> None:
        key = _canonical_key(name)
        if key not in self.required:
            return
        if is_down:
            self._pressed.add(key)  # set add is idempotent -> key-repeat ignored
            if self.required <= self._pressed and not self._active and not self._latched:
                self._active = True
                self._on_activate()
        else:
            was_active = self._active
            self._pressed.discard(key)
            if was_active:
                self._active = False
                self._latched = True  # re-arm only after a full release
                self._on_release_any()
            if not self._pressed:
                self._latched = False

    def force_latch(self) -> None:
        """A stop happened without a key release (max-duration/cancel/error):
        drop active state and require a full release before the next recording."""
        self._active = False
        self._latched = True


def register_hotkeys(client: PttClient, config: PttConfig):
    """Model the hotkey as individual keys (press→record, any-release→submit),
    plus Esc→cancel. Returns (keyboard_module, listener) so the caller can
    unhook on exit."""
    import keyboard

    listener = HotkeyListener(
        config.hotkey,
        on_activate=client.begin_recording,
        on_release_any=client.finish_recording,
    )
    client.latch_hook = listener.force_latch

    def _dispatch(event) -> None:
        listener.on_event(event.name, event.event_type == keyboard.KEY_DOWN)

    keyboard.hook(_dispatch)
    keyboard.add_hotkey("esc", client.cancel_recording)
    return keyboard, listener


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fifi push-to-talk client (Windows host).")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"API base URL (default {DEFAULT_SERVER})")
    args = parser.parse_args(argv)

    config = PttConfig.load(server=args.server)

    deps_ok, missing = check_desktop_deps()
    if not deps_ok:
        print(
            "Missing desktop dependencies: " + ", ".join(missing) + "\n"
            "Install them on the Windows host:\n"
            "  .venv\\Scripts\\python.exe -m pip install -r requirements-desktop.txt"
        )
        return 1

    http = httpx.Client(base_url=config.server)
    try:
        if not validate_startup(config, http):
            return 1
    except Exception as exc:  # defensive — never crash on a bad response
        print(f"Startup validation failed: {exc}")
        http.close()
        return 1

    client = PttClient(config, http=http)
    print(f"Fifi push-to-talk ready. Hold {config.hotkey} to talk. Ctrl+C to quit.")
    try:
        keyboard, _listener = register_hotkeys(client, config)
    except Exception as exc:
        print(f"Could not register the global hotkey: {exc}")
        client.close()
        return 1

    try:
        keyboard.wait("ctrl+c")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        client.close()
        print("\nFifi push-to-talk stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
