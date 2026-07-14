"""Main-project side of the Voice Lab (Phase 3D.0).

The Voice Lab worker itself is tested in voice_lab/tests (run separately:
python -m pytest voice_lab/tests). Here we pin the MAIN app's contract:
- TTS_ENGINE=voice_lab only ever talks HTTP to the loopback worker;
- the active profile comes from config/voices/active.json;
- a dead/broken/slow worker falls back to Windows TTS (when allowed) and
  never affects /health, command execution, or safety;
- no neural TTS dependency exists in the main environment;
- runtime voice-start/voice-status/voice-stop manage the isolated worker,
  and `start` never launches it automatically.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, get_settings
from app.main import app
from app.voice import tts as tts_module
from app.voice.tts import TextToSpeechService, active_voice_profile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_runtime  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_port_probe(monkeypatch, tmp_path):
    """Ownership probes must never depend on this machine's real port 8766
    state (a live worker would flip results), and the real PID file must
    never be touched. Ownership tests override these stubs explicitly."""
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: None)
    monkeypatch.setattr(local_runtime, "VOICE_LAB_PID_FILE", tmp_path / "voice_lab.pid")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def voice_lab_on(settings, monkeypatch):
    settings.enable_voice = True
    settings.tts_engine = "voice_lab"
    monkeypatch.setattr(settings, "voice_lab_allow_fallback", True)
    return settings


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSapi:
    """Stands in for pyttsx3 so 'fallback to Windows TTS' is observable."""

    def __init__(self):
        self.spoken: list[str] = []

    def init(self):
        recorder = self

        class Driver:
            def say(self, text):
                recorder.spoken.append(text)

            def runAndWait(self):
                pass

        return Driver()


# --- configuration -------------------------------------------------------------------


def test_voice_lab_settings_defaults():
    settings = get_settings()
    assert settings.voice_lab_url == "http://127.0.0.1:8766"
    assert settings.voice_profile == "fifi_warm"
    assert settings.voice_lab_timeout_seconds == 60.0
    assert settings.voice_lab_allow_fallback is True


def test_active_profile_read_from_shared_file():
    """The reader returns exactly what the shared file says — whichever
    profile the user has selected (selection changes it at any time)."""
    on_disk = json.loads(
        (PROJECT_ROOT / "config" / "voices" / "active.json").read_text(encoding="utf-8")
    )
    assert active_voice_profile() == on_disk["profile"]
    profile_file = (
        PROJECT_ROOT / "config" / "voices" / "profiles" / f"{on_disk['profile']}.json"
    )
    assert profile_file.is_file()  # active always points at a real profile


@pytest.fixture(autouse=True)
def _fresh_active_cache(monkeypatch):
    """The active-profile reader is mtime-cached; isolate tests from each other."""
    monkeypatch.setattr(
        tts_module, "_active_cache", {"mtime": None, "profile": None, "updated_utc": None}
    )


def test_active_profile_hot_reloads_without_restart(monkeypatch, tmp_path):
    """Phase 3D.0.1: `voice_lab.py select` takes effect on the next speak —
    the reader notices the file change (mtime + updated_utc metadata)."""
    import os

    active = tmp_path / "active.json"
    monkeypatch.setattr(tts_module, "ACTIVE_VOICE_PATH", active)

    active.write_text(
        json.dumps({"profile": "fifi_warm", "updated_utc": "2026-07-11T10:00:00+00:00"}),
        encoding="utf-8",
    )
    assert active_voice_profile() == "fifi_warm"

    active.write_text(
        json.dumps({"profile": "fifi_es_warm", "updated_utc": "2026-07-11T11:00:00+00:00"}),
        encoding="utf-8",
    )
    os.utime(active)  # guarantee a fresh mtime even on coarse filesystems
    info = tts_module.active_voice_info()
    assert info["profile"] == "fifi_es_warm"  # reloaded, no restart
    assert info["updated_utc"] == "2026-07-11T11:00:00+00:00"  # change metadata

    active.unlink()
    assert active_voice_profile() is None  # deletion also detected


def test_active_profile_is_cached_between_reads(monkeypatch, tmp_path):
    active = tmp_path / "active.json"
    monkeypatch.setattr(tts_module, "ACTIVE_VOICE_PATH", active)
    active.write_text(json.dumps({"profile": "fifi_warm"}), encoding="utf-8")
    reads = {"count": 0}
    real_read = Path.read_text

    def counting_read(self, *args, **kwargs):
        if self == active:
            reads["count"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    for _ in range(5):
        assert active_voice_profile() == "fifi_warm"
    assert reads["count"] == 1  # unchanged file is parsed once


def test_active_profile_handles_missing_and_corrupt(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_module, "ACTIVE_VOICE_PATH", tmp_path / "missing.json")
    assert active_voice_profile() is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{nope", encoding="utf-8")
    monkeypatch.setattr(tts_module, "ACTIVE_VOICE_PATH", corrupt)
    assert active_voice_profile() is None


# --- voice_lab provider ---------------------------------------------------------------


def test_speak_posts_to_worker_with_active_profile(voice_lab_on, monkeypatch, tmp_path):
    active = tmp_path / "active.json"
    active.write_text(json.dumps({"profile": "fifi_warm"}), encoding="utf-8")
    monkeypatch.setattr(tts_module, "ACTIVE_VOICE_PATH", active)
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json, timeout=timeout)
        return FakeResponse({"status": "ok", "profile": "fifi_warm", "engine": "kokoro"})

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    result = TextToSpeechService().speak("hola")
    assert result["engine"] == "voice_lab"
    assert result["synthesis_engine"] == "kokoro"
    assert sent["url"] == "http://127.0.0.1:8766/synthesize"
    assert sent["body"] == {"text": "hola", "profile": "fifi_warm", "play": True}
    assert sent["timeout"] == 60.0  # VOICE_LAB_TIMEOUT_SECONDS


def test_worker_down_falls_back_to_windows_tts(voice_lab_on, monkeypatch):
    def fake_post(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    sapi = FakeSapi()
    monkeypatch.setattr(tts_module, "pyttsx3", sapi)
    result = TextToSpeechService().speak("hola")
    assert sapi.spoken == ["hola"]  # Windows TTS actually spoke
    assert result["engine"] == "windows"
    assert result["fallback_from"] == "voice_lab"
    assert "unavailable" in result["fallback_reason"]


def test_worker_timeout_falls_back(voice_lab_on, monkeypatch):
    def fake_post(url, **kwargs):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    sapi = FakeSapi()
    monkeypatch.setattr(tts_module, "pyttsx3", sapi)
    result = TextToSpeechService().speak("hola")
    assert result["fallback_from"] == "voice_lab"
    assert "ReadTimeout" in result["fallback_reason"]
    assert sapi.spoken == ["hola"]


def test_worker_error_result_falls_back(voice_lab_on, monkeypatch):
    monkeypatch.setattr(
        tts_module.httpx, "post",
        lambda url, **kw: FakeResponse({"status": "error", "message": "all engines failed"}),
    )
    sapi = FakeSapi()
    monkeypatch.setattr(tts_module, "pyttsx3", sapi)
    result = TextToSpeechService().speak("hola")
    assert result["fallback_from"] == "voice_lab"
    assert "all engines failed" in result["fallback_reason"]


def test_fallback_can_be_disabled(voice_lab_on, monkeypatch):
    monkeypatch.setattr(voice_lab_on, "voice_lab_allow_fallback", False)

    def fake_post(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    sapi = FakeSapi()
    monkeypatch.setattr(tts_module, "pyttsx3", sapi)
    result = TextToSpeechService().speak("hola")
    assert "error" in result
    assert sapi.spoken == []  # no fallback speech


# --- TTS failure never affects health / commands / safety -----------------------------


def test_health_unaffected_by_dead_worker(client, voice_lab_on, monkeypatch):
    def fake_post(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    health = client.get("/health").json()
    assert health["status"] == "ok"


def test_commands_and_safety_unaffected_by_dead_worker(client, voice_lab_on, monkeypatch):
    def fake_post(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(tts_module.httpx, "post", fake_post)
    data = client.post("/command", json={"text": "open app notepad"}).json()
    assert data["status"] == "needs_confirmation"  # safety gate untouched


# --- isolation guards ------------------------------------------------------------------


NEURAL_PACKAGES = ("kokoro", "torch", "transformers", "accelerate", "onnxruntime-gpu")


def test_main_requirements_stay_free_of_neural_tts():
    for name in ("requirements.txt", "requirements-voice.txt", "requirements-desktop.txt"):
        content = (PROJECT_ROOT / name).read_text(encoding="utf-8").lower()
        for line in content.splitlines():
            line = line.split("#")[0].strip()
            for package in NEURAL_PACKAGES:
                assert not line.startswith(package), f"{package} leaked into {name}"


def test_main_tts_module_never_imports_neural_engines():
    source = (PROJECT_ROOT / "app" / "voice" / "tts.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import kokoro", "import transformers", "voice_lab.app"):
        assert forbidden not in source
    # The integration is HTTP + the shared active.json — nothing else.
    assert "voice_lab_url" in source
    assert "active.json" in source


def test_voice_lab_worker_url_is_loopback():
    assert get_settings().voice_lab_url.startswith("http://127.0.0.1")


# --- runtime commands -------------------------------------------------------------------


def test_voice_subcommands_exist():
    parser = local_runtime.build_parser()
    for command, func in (
        ("voice-start", local_runtime.cmd_voice_start),
        ("voice-status", local_runtime.cmd_voice_status),
        ("voice-stop", local_runtime.cmd_voice_stop),
        ("voice-warm", local_runtime.cmd_voice_warm),
    ):
        args = parser.parse_args([command])
        assert args.func is func


def _worker_up(monkeypatch):
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": local_runtime.voice_lab_expected_build()},
    )


def test_voice_warm_polls_async_job_to_ready(monkeypatch, capsys):
    """3D.0.4: voice-warm submits the async job, prints REAL stages, exits 0
    only on ready."""
    _worker_up(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "voice_lab_warm",
        lambda profile=None: {"job_id": "warm123", "status": "queued",
                              "model_cache": "cached"},
    )
    states = iter([
        {"status": "loading_from_disk", "message": "Cargando…", "elapsed_seconds": 1},
        {"status": "moving_to_gpu", "message": "Transfiriendo…", "elapsed_seconds": 5},
        {"status": "completed", "elapsed_seconds": 9,
         "result": {"ready": True, "profile": "fifi_es_warm", "engine": "kokoro",
                    "seconds": 1.2, "fallback_used": False}},
    ])
    monkeypatch.setattr(local_runtime, "voice_lab_job", lambda job_id: next(states))
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-warm"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "loading_from_disk" in out and "moving_to_gpu" in out  # real stages
    assert "Warm READY" in out and "engine=kokoro" in out


def test_voice_warm_immediate_ready_when_already_loaded(monkeypatch, capsys):
    _worker_up(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "voice_lab_warm",
        lambda profile=None: {"status": "ready", "already_loaded": True,
                              "profile": "fifi_luna", "engine": "qwen3_tts",
                              "last_warm_job": {"status": "completed"}},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-warm"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    # Timeout reconciliation: a load that finished in the background is
    # reported truthfully — and no duplicate load is started.
    assert "La solicitud agotó el tiempo de espera, pero el motor terminó de cargar." in out


def test_voice_warm_failed_job_exits_nonzero(monkeypatch, capsys):
    _worker_up(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "voice_lab_warm",
        lambda profile=None: {"job_id": "warmbad", "status": "queued"},
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_job",
        lambda job_id: {"status": "failed", "error_code": "warm_failed",
                        "error": "el calentamiento falló"},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-warm"])
    exit_code = args.func(args)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Warm FAILED" in out  # FAILED text ALWAYS pairs with nonzero exit


def test_voice_warm_timeout_reconciles_with_late_completion(monkeypatch, capsys):
    _worker_up(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "voice_lab_warm",
        lambda profile=None: {"job_id": "warmslow", "status": "queued"},
    )
    monkeypatch.setattr(local_runtime, "warm_timeout_seconds", lambda: 0.0)  # instant timeout
    monkeypatch.setattr(
        local_runtime, "voice_lab_job",
        lambda job_id: {"status": "completed", "result": {"ready": True}},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-warm"])
    assert args.func(args) == 0  # ready after all — not a failure
    assert "terminó de cargar" in capsys.readouterr().out


def test_voice_warm_no_worker_response_exits_nonzero(monkeypatch, capsys):
    _worker_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "voice_lab_warm", lambda profile=None: None)
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-warm"])
    assert args.func(args) == 1
    assert "Warm FAILED" in capsys.readouterr().out


def test_voice_preview_command_reports_playable_result(monkeypatch, capsys):
    _worker_up(monkeypatch)

    class Response:
        def json(self):
            return {"job_id": "prev1", "status": "queued"}

    monkeypatch.setattr("httpx.post", lambda *a, **k: Response())
    monkeypatch.setattr(
        local_runtime, "voice_lab_job",
        lambda job_id: {"status": "completed", "result": {
            "requested_profile": "fifi_es_warm", "requested_engine": "kokoro",
            "actual_engine": "kokoro", "fallback_used": False,
            "audio_url": "/audio/previews/fifi_es_warm-abc123?v=1",
            "audio_duration_seconds": 3.4, "cache_hit": False,
            "file": "previews/auditions/fifi_es_warm-abc123.wav",
        }},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-preview", "fifi_es_warm"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "requested=kokoro actual=kokoro fallback=False" in out
    assert "/audio/previews/fifi_es_warm-abc123" in out


def test_voice_preview_refusal_exits_nonzero(monkeypatch, capsys):
    _worker_up(monkeypatch)

    class Refusal:
        def json(self):
            return {"status": "error", "error_code": "unknown_profile",
                    "user_message": "El perfil 'nope' no existe en el almacén compartido."}

    monkeypatch.setattr("httpx.post", lambda *a, **k: Refusal())
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-preview", "nope"])
    assert args.func(args) == 1
    assert "unknown_profile" in capsys.readouterr().out


def test_voice_ui_starts_worker_and_opens_browser(monkeypatch, launches, capsys):
    """voice-ui = worker up + Designer UI in the default browser (local only)."""
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": local_runtime.voice_lab_expected_build()},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-ui"])
    assert args.func is local_runtime.cmd_voice_ui
    assert args.func(args) == 0
    assert launches.browser_urls == ["http://127.0.0.1:8766/ui"]
    assert "loopback" in capsys.readouterr().out


def test_voice_ui_fails_cleanly_without_venv(monkeypatch, launches):
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", Path("Z:/missing"))
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-ui"])
    assert args.func(args) == 1
    assert launches.browser_urls == []  # never opens a browser on failure


def test_busy_worker_is_never_restarted(monkeypatch, capsys):
    """Phase 3D.0.3: /health slow + owned pid alive = BUSY, not down. The
    runtime must not start a duplicate worker against the bound port."""
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)  # probe timeout
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 4242)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: True)  # ours, alive
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: (_ for _ in ()).throw(AssertionError("must NOT start a duplicate worker")),
    )
    assert local_runtime._voice_lab_up() is True
    out = capsys.readouterr().out
    assert "busy" in out and "NOT starting a duplicate" in out


def test_unowned_pid_is_never_trusted_for_busy_detection(monkeypatch, tmp_path):
    """Only a PID recorded by this script counts as ours: no PID file → the
    dead-worker path (fresh start), never touching processes we don't own."""
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", tmp_path)  # no .venv → clean stop
    assert local_runtime._voice_lab_up(out=lambda *a: None) is False


def test_tts_distinguishes_timeout_from_unreachable(voice_lab_on, monkeypatch):
    """Timeout categories: a slow synthesis is NOT 'worker unavailable'."""
    sapi = FakeSapi()
    monkeypatch.setattr(tts_module, "pyttsx3", sapi)

    monkeypatch.setattr(
        tts_module.httpx, "post",
        lambda url, **kw: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
    )
    result = TextToSpeechService().speak("hola")
    assert "timed out" in result["fallback_reason"]
    assert "unavailable" not in result["fallback_reason"]

    monkeypatch.setattr(
        tts_module.httpx, "post",
        lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("refused")),
    )
    result = TextToSpeechService().speak("hola")
    assert "unavailable" in result["fallback_reason"]


def test_voice_start_recovers_after_worker_crash(monkeypatch, tmp_path, capsys):
    """A died worker is simply restarted by voice-start (restart recovery)."""
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", tmp_path)
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)  # crashed
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 999)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)  # truly dead
    started = {}
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: started.update(done=True) or object(),
    )
    monkeypatch.setattr(
        local_runtime, "wait_for_voice_lab_or_exit",
        lambda process, **k: (
            {"engines_available": {"kokoro": True}, "active_profile": "fifi_warm"}, None,
        ),
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-start"])
    assert args.func(args) == 0
    assert started == {"done": True}
    assert "Voice Lab is up" in capsys.readouterr().out


def test_start_auto_starts_voice_lab_only_when_opted_in(monkeypatch):
    """RUNTIME_AUTO_START_VOICE_LAB=true brings the lab up and prewarms TTS."""
    calls = []
    _stub_healthy_start(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTO_START_VOICE_LAB", "true")
    monkeypatch.setenv("RUNTIME_PREWARM_TTS", "true")
    monkeypatch.setattr(local_runtime, "_voice_lab_up", lambda out=print: calls.append("up") or True)
    monkeypatch.setattr(
        local_runtime, "voice_lab_warm",
        lambda profile=None: calls.append("warm") or {"status": "ok", "engine": "kokoro", "profile": "fifi_warm"},
    )

    class _Args:
        no_ollama = False
        stop_ollama = False

    assert local_runtime.cmd_start(_Args()) == 0
    assert calls == ["up", "warm"]


def test_voice_lab_failure_never_fails_start(monkeypatch, capsys):
    """Even opted-in, a broken Voice Lab is a warning — never a start failure."""
    _stub_healthy_start(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTO_START_VOICE_LAB", "true")
    monkeypatch.setattr(local_runtime, "_voice_lab_up", lambda out=print: False)

    class _Args:
        no_ollama = False
        stop_ollama = False

    assert local_runtime.cmd_start(_Args()) == 0  # runtime still comes up
    out = capsys.readouterr().out
    assert "Windows TTS fallback" in out
    assert "Runtime is up" in out


def _stub_healthy_start(monkeypatch):
    monkeypatch.setattr(local_runtime, "docker_available", lambda: True)
    monkeypatch.setattr(local_runtime, "start_ollama", lambda: True)
    monkeypatch.setattr(local_runtime, "wait_for_ollama", lambda url: True)
    monkeypatch.setattr(local_runtime, "ollama_container_id", lambda: "abc123")
    monkeypatch.setattr(local_runtime, "detect_gpu", lambda cid: (True, "RTX 5060 Ti"))
    monkeypatch.setattr(local_runtime, "ensure_model", lambda url, model: (True, "ready"))
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt", lambda url: True)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: False)
    monkeypatch.setattr(local_runtime, "wait_for_api", lambda url: {"status": "ok"})
    monkeypatch.setattr(local_runtime, "api_health", lambda url: {"version": "0.3.2"})
    monkeypatch.setattr(local_runtime, "get_identity", lambda url: {"agent_name": "Fifi"})
    monkeypatch.setattr(local_runtime, "read_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())


def test_voice_start_reports_already_running(monkeypatch, capsys):
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": local_runtime.voice_lab_expected_build()},
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-start"])
    assert args.func(args) == 0
    assert "already running" in capsys.readouterr().out


def test_voice_start_requires_isolated_venv(monkeypatch, capsys):
    """Without voice_lab/.venv the worker must not start with the main env."""
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: (_ for _ in ()).throw(AssertionError("must not launch without the venv")),
    )
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", Path("Z:/definitely/missing"))
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-start"])
    assert args.func(args) == 1
    assert "voice_lab/scripts/setup.py" in capsys.readouterr().out


def test_voice_stop_only_touches_owned_pid(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-stop"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "nothing to stop" in out
    assert "not touched" in out


def test_voice_status_when_worker_down(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    parser = local_runtime.build_parser()
    args = parser.parse_args(["voice-status"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "not running" in out
    assert "fallback" in out


def test_start_never_launches_voice_lab(monkeypatch):
    """By default (RUNTIME_AUTO_START_VOICE_LAB=false) start never launches it."""
    monkeypatch.setenv("RUNTIME_AUTO_START_VOICE_LAB", "false")

    class _Args:
        no_ollama = False
        stop_ollama = False

    # Everything healthy (mirrors test_local_runtime._all_up).
    monkeypatch.setattr(local_runtime, "docker_available", lambda: True)
    monkeypatch.setattr(local_runtime, "start_ollama", lambda: True)
    monkeypatch.setattr(local_runtime, "wait_for_ollama", lambda url: True)
    monkeypatch.setattr(local_runtime, "ollama_container_id", lambda: "abc123")
    monkeypatch.setattr(local_runtime, "detect_gpu", lambda cid: (True, "RTX 5060 Ti"))
    monkeypatch.setattr(local_runtime, "ensure_model", lambda url, model: (True, "ready"))
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt", lambda url: True)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: False)
    monkeypatch.setattr(local_runtime, "wait_for_api", lambda url: {"status": "ok"})
    monkeypatch.setattr(local_runtime, "api_health", lambda url: {"version": "0.3.2"})
    monkeypatch.setattr(local_runtime, "get_identity", lambda url: {"agent_name": "Fifi"})
    monkeypatch.setattr(local_runtime, "read_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: (_ for _ in ()).throw(AssertionError("voice lab must not auto-start")),
    )
    monkeypatch.setattr(
        local_runtime, "wait_for_voice_lab",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("voice lab must not auto-start")),
    )
    assert local_runtime.cmd_start(_Args()) == 0


def test_voice_lab_docker_worker_is_contained():
    """The Voice Lab worker CAN now run in Docker (profile 'voice'), but the
    container is a contained support service: profile-gated and loopback-only,
    and it never controls the desktop — host audio playback and Windows SAPI are
    disabled inside it. Real desktop automation and the Windows TTS fallback
    stay on the host."""
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    block = compose.split("\n  voice-lab:", 1)[1].split("\n  voice-model-seed:", 1)[0]
    assert "- voice" in block                                # profile-gated (opt-in)
    assert '"127.0.0.1:8766:8766"' in block                  # loopback only
    assert 'VOICE_LAB_PLAYBACK_ENABLED: "false"' in block    # never touches host audio
    assert 'VOICE_LAB_DISABLE_SAPI: "true"' in block         # SAPI stays host-only


def test_start_voice_lab_process_is_isolated_and_loopback(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

        class P:
            pid = 4242

        return P()

    monkeypatch.setattr(local_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_runtime, "ensure_dirs", lambda: None)
    monkeypatch.setattr(local_runtime, "VOICE_LAB_PID_FILE", tmp_path / "voice_lab.pid")
    monkeypatch.setattr(local_runtime, "VOICE_LAB_LOG_FILE", tmp_path / "worker.log")
    venv_python = str(local_runtime.VOICE_LAB_DIR / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(local_runtime, "voice_lab_python", lambda: venv_python)

    local_runtime.start_voice_lab_process()
    cmd = captured["cmd"]
    # Phase 3D.0.3a: the CWD-independent runner replaces `-m uvicorn app.main:app`
    # (which resolved `app` from the CWD). Loopback binding now lives in
    # run_worker.py via VoiceLabSettings (host defaults to 127.0.0.1:8766).
    assert cmd == [venv_python, str(local_runtime.VOICE_LAB_RUNNER)]
    assert cmd[0].startswith(str(local_runtime.VOICE_LAB_DIR))  # isolated venv only
    assert captured["cwd"] == local_runtime.VOICE_LAB_DIR  # its own package/env/.env
    assert "docker" not in cmd  # never a container
    record = json.loads((tmp_path / "voice_lab.pid").read_text(encoding="utf-8"))
    assert record["pid"] == 4242 and record["build"]
