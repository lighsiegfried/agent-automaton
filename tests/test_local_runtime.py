"""Unit tests for scripts/local_runtime.py — the local persistent runtime.

Every Docker, subprocess, and HTTP call is mocked: no Docker, Ollama, GPU, or
real API process is needed. These tests pin the runtime's contract:
- start brings up GPU-first Ollama (Docker) + the API on the Windows host;
- status reports each moving part;
- stop kills the host API but preserves Ollama and the model volume;
- GPU is required unless CPU is explicitly opted in;
- the runtime never enables real Windows tools and never runs the API in Docker;
- compose.yml is the only compose file, and runtime state/logs are gitignored.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_runtime  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- helpers ---------------------------------------------------------------------


def _all_up(monkeypatch, *, gpu=True, gpu_name="NVIDIA GeForce RTX 5060 Ti"):
    """Pretend Docker/Ollama/model/API are all healthy. Individual tests override."""
    monkeypatch.setattr(local_runtime, "docker_available", lambda: True)
    monkeypatch.setattr(local_runtime, "start_ollama", lambda: True)
    monkeypatch.setattr(local_runtime, "wait_for_ollama", lambda url: True)
    monkeypatch.setattr(local_runtime, "ollama_container_id", lambda: "abc123")
    monkeypatch.setattr(local_runtime, "detect_gpu", lambda cid: (gpu, gpu_name))
    monkeypatch.setattr(local_runtime, "ensure_model", lambda url, model: (True, "model ready"))
    # API not yet running, then comes up healthy with a Fifi identity.
    monkeypatch.setattr(local_runtime, "api_running", lambda url: False)
    monkeypatch.setattr(local_runtime, "wait_for_api", lambda url: {"status": "ok"})
    monkeypatch.setattr(local_runtime, "api_health", lambda url: {"version": "0.3.2"})
    monkeypatch.setattr(local_runtime, "get_identity", lambda url: {"agent_name": "Fifi"})
    # No owned API process by default (don't touch the real PID file in tests).
    monkeypatch.setattr(local_runtime, "read_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    # Prewarm (Phase 3D.1) succeeds by default; individual tests override.
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt", lambda url: True)


def _status_stubs(monkeypatch, *, llm_info=None, stt=None, wake_deps=(True, [])):
    """Stub the Phase 3D.1 status probes so cmd_status never talks to anything."""
    monkeypatch.setattr(local_runtime, "llm_loaded_info", lambda: llm_info)
    monkeypatch.setattr(local_runtime, "voice_status", lambda url: stt)
    monkeypatch.setattr(local_runtime.fifi_wake, "check_wake_deps", lambda: wake_deps)
    monkeypatch.setattr(local_runtime.fifi_wake, "wake_model_present", lambda: False)
    monkeypatch.setattr(local_runtime.fifi_wake, "wake_active", lambda: False)


class _Args:
    def __init__(self, **kw):
        self.no_ollama = kw.get("no_ollama", False)
        self.stop_ollama = kw.get("stop_ollama", False)


# --- start flow ------------------------------------------------------------------


def test_start_flow_brings_everything_up(monkeypatch, capsys):
    started = {}
    _all_up(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "start_api_process",
        lambda host, port: started.update(host=host, port=port) or object(),
    )
    assert local_runtime.cmd_start(_Args()) == 0
    out = capsys.readouterr().out
    assert "GPU detected    : yes" in out
    assert "Identity        : Fifi ready" in out
    assert "Runtime is up" in out
    # API is started on the loopback host, not in Docker.
    assert started["host"] == local_runtime.api_host()


def test_start_no_ollama_skips_docker(monkeypatch, capsys):
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    # If Docker were touched, this would fail loudly.
    monkeypatch.setattr(
        local_runtime, "start_ollama",
        lambda: (_ for _ in ()).throw(AssertionError("must not start Ollama")),
    )
    assert local_runtime.cmd_start(_Args(no_ollama=True)) == 0
    assert "Skipping Ollama startup" in capsys.readouterr().out


def test_start_leaves_already_running_api(monkeypatch, capsys):
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: True)
    monkeypatch.setattr(
        local_runtime, "start_api_process",
        lambda host, port: (_ for _ in ()).throw(AssertionError("must not restart API")),
    )
    assert local_runtime.cmd_start(_Args()) == 0
    assert "API already running" in capsys.readouterr().out


# --- GPU gating ------------------------------------------------------------------


def test_gpu_required_failure(monkeypatch, capsys):
    """No GPU + GPU required + CPU not allowed => start fails, API never launches."""
    _all_up(monkeypatch, gpu=False, gpu_name="no GPU")
    monkeypatch.setattr(local_runtime, "require_gpu", lambda: True)
    monkeypatch.setattr(local_runtime.docker_llm, "cpu_allowed", lambda: False)
    monkeypatch.setattr(
        local_runtime, "start_api_process",
        lambda host, port: (_ for _ in ()).throw(AssertionError("API must not start on GPU failure")),
    )
    assert local_runtime.cmd_start(_Args()) == 1
    assert "GPU detected    : NO" in capsys.readouterr().out


def test_cpu_fallback_only_when_explicitly_allowed(monkeypatch, capsys):
    _all_up(monkeypatch, gpu=False, gpu_name="no GPU")
    monkeypatch.setattr(local_runtime, "require_gpu", lambda: True)
    monkeypatch.setattr(local_runtime.docker_llm, "cpu_allowed", lambda: True)  # explicit opt-in
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    assert local_runtime.cmd_start(_Args()) == 0
    assert "CPU-only inference" in capsys.readouterr().out


# --- status flow -----------------------------------------------------------------


def test_status_flow_reports_every_part(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "docker_available", lambda: True)
    monkeypatch.setattr(local_runtime, "ollama_container_status", lambda: "running")
    monkeypatch.setattr(local_runtime, "ollama_container_id", lambda: "abc123")
    monkeypatch.setattr(local_runtime, "detect_gpu", lambda cid: (True, "NVIDIA RTX 5060 Ti"))
    monkeypatch.setattr(local_runtime.llm_smoke, "model_present", lambda url, model: True)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: True)
    monkeypatch.setattr(local_runtime, "get_identity", lambda url: {"agent_name": "Fifi"})
    _status_stubs(
        monkeypatch,
        llm_info={"processor": "gpu", "vram_gb": 5.6, "fully_gpu": True},
    )

    assert local_runtime.cmd_status(_Args()) == 0
    out = capsys.readouterr().out
    assert "Docker available: True" in out
    assert "Ollama container: running" in out
    assert "GPU detected    : True" in out
    assert "NVIDIA RTX 5060 Ti" in out
    assert "Model available : True" in out
    assert "LLM warm        : True (gpu, 5.6 GB VRAM)" in out
    assert "API running     : True" in out
    assert '"agent_name": "Fifi"' in out
    assert "Wake word       : enabled=False deps=True" in out
    assert "Wake model      : present=False" in out
    assert "Wake active     : False" in out


def test_status_handles_docker_down(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "docker_available", lambda: False)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: False)
    _status_stubs(monkeypatch, wake_deps=(False, ["openwakeword"]))
    assert local_runtime.cmd_status(_Args()) == 0
    out = capsys.readouterr().out
    assert "Docker available: False" in out
    assert "Model available : False" in out
    assert "LLM warm        : False" in out
    assert "API running     : False" in out
    assert "missing: openwakeword" in out


# --- stop flow -------------------------------------------------------------------


def test_stop_kills_host_api_but_keeps_ollama(monkeypatch, capsys):
    killed = {}
    monkeypatch.setattr(local_runtime, "read_pid", lambda: 4321)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: True)
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.update(pid=pid))
    monkeypatch.setattr(local_runtime, "clear_pid", lambda: killed.update(cleared=True))
    monkeypatch.setattr(local_runtime, "keep_ollama_running", lambda: True)
    monkeypatch.setattr(
        local_runtime, "stop_ollama_service",
        lambda: (_ for _ in ()).throw(AssertionError("must not stop Ollama by default")),
    )
    assert local_runtime.cmd_stop(_Args()) == 0
    out = capsys.readouterr().out
    assert killed == {"pid": 4321, "cleared": True}
    assert "stopped (pid 4321)" in out
    assert "left running (persistent)" in out
    assert "Models and Docker volumes were not deleted." in out


def test_stop_ollama_flag_stops_service_without_removing_volumes(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "read_pid", lambda: None)
    stopped = {}
    monkeypatch.setattr(
        local_runtime, "stop_ollama_service", lambda: stopped.update(called=True) or True
    )
    assert local_runtime.cmd_stop(_Args(stop_ollama=True)) == 0
    out = capsys.readouterr().out
    assert stopped == {"called": True}
    assert "Models remain in the ollama-models volume" in out
    # The stop command uses `stop`, never `down`/`-v` — volumes are preserved.
    assert local_runtime.OLLAMA_STOP == [
        "docker", "compose", "--profile", "llm", "stop", "ollama",
    ]
    assert "down" not in local_runtime.OLLAMA_STOP
    assert "-v" not in local_runtime.OLLAMA_STOP


def test_stop_clears_stale_pid(monkeypatch, capsys):
    cleared = {}
    monkeypatch.setattr(local_runtime, "read_pid", lambda: 999)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    monkeypatch.setattr(local_runtime, "clear_pid", lambda: cleared.update(done=True))
    monkeypatch.setattr(local_runtime, "keep_ollama_running", lambda: True)
    assert local_runtime.cmd_stop(_Args()) == 0
    assert cleared == {"done": True}
    assert "stale PID" in capsys.readouterr().out


# --- safety / containment --------------------------------------------------------


def test_runtime_never_enables_real_windows_tools(monkeypatch):
    """start_api_process launches uvicorn on the host and never forces real tools."""
    captured = {}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd

        class P:
            pid = 1234

        return P()

    monkeypatch.setattr(local_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_runtime, "write_pid", lambda pid: None)
    monkeypatch.setattr(local_runtime, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        local_runtime, "open", lambda *a, **k: __import__("io").BytesIO(), raising=False
    )

    local_runtime.start_api_process("127.0.0.1", 8000)
    cmd = captured["cmd"]
    # It is the host uvicorn process, never `docker compose up`.
    assert "uvicorn" in cmd and "app.main:app" in cmd
    assert "docker" not in cmd and "compose" not in cmd
    # No env override forcing real tools on (inherits .env; default stays false).
    env = captured["env"]
    if env is not None:
        assert env.get("ENABLE_REAL_WINDOWS_TOOLS", "false") != "true"


def test_docker_only_starts_ollama_support_service():
    # The only compose "up" the runtime performs is the profile-gated ollama
    # service — never the API, and never real tools.
    docker_up = local_runtime.docker_llm.COMPOSE_UP_OLLAMA
    assert docker_up == ["docker", "compose", "--profile", "llm", "up", "-d", "ollama"]
    assert "api" not in docker_up


def test_compose_file_naming_convention():
    assert (PROJECT_ROOT / "compose.yml").exists()
    assert not (PROJECT_ROOT / "docker-compose.yml").exists()


def test_compose_ollama_is_persistent():
    ollama_block = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8").split("ollama:", 1)[1]
    assert "restart: unless-stopped" in ollama_block
    assert "ollama-models" in ollama_block  # model volume persists


def test_runtime_state_and_logs_are_gitignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    # storage/* ignores everything but the .gitkeep; runtime/logs are covered and
    # also listed explicitly for clarity. No re-include negations for them.
    assert "storage/*" in gitignore
    assert "storage/runtime/" in gitignore
    assert "storage/logs/" in gitignore
    assert "!storage/runtime" not in gitignore
    assert "!storage/logs" not in gitignore
    # The runtime writes only under storage/, so the ignore rules apply.
    assert local_runtime.RUNTIME_DIR.parent.name == "storage"
    assert local_runtime.LOG_DIR.parent.name == "storage"


def test_ptt_subcommand_launches_client(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        local_runtime.fifi_ptt, "run",
        lambda argv: launched.update(argv=argv) or 0,
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["ptt"])
    assert args.func is local_runtime.cmd_ptt
    assert args.func(args) == 0
    # It targets the configured host API and never starts push-to-talk in Docker.
    assert launched["argv"][0] == "--server"
    assert launched["argv"][1] == local_runtime.api_url()


def test_start_restarts_owned_dead_api(monkeypatch, capsys):
    _all_up(monkeypatch)  # ollama healthy; api_running() is False
    monkeypatch.setattr(local_runtime, "read_pid", lambda: 4321)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)  # owned proc died
    cleared = {}
    started = {}
    monkeypatch.setattr(local_runtime, "clear_pid", lambda: cleared.update(done=True))
    monkeypatch.setattr(
        local_runtime, "start_api_process",
        lambda host, port: started.update(started=True) or object(),
    )
    assert local_runtime.cmd_start(_Args()) == 0
    out = capsys.readouterr().out
    assert "has died — restarting it once" in out  # reports why
    assert cleared and started


def test_start_never_touches_unowned_process(monkeypatch):
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "read_pid", lambda: None)  # nothing we own
    killed = []
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    assert local_runtime.cmd_start(_Args()) == 0
    assert killed == []  # never kill/restart a process we don't own


def test_status_reports_owned_pid_when_api_down(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "docker_available", lambda: False)
    monkeypatch.setattr(local_runtime, "api_running", lambda url: False)  # /health-validated
    monkeypatch.setattr(local_runtime, "read_pid", lambda: 555)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    assert local_runtime.cmd_status(_Args()) == 0
    out = capsys.readouterr().out
    assert "Owned API PID   : 555 (died)" in out


# --- persistent warm runtime (Phase 3D.1) ------------------------------------------


def test_start_warms_llm_automatically(monkeypatch, capsys):
    """start preloads the model (keep_alive=-1) right after Ollama is ready."""
    warmed = {}
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "prewarm_llm_enabled", lambda: True)
    monkeypatch.setattr(
        local_runtime, "warm_llm",
        lambda url, model: warmed.update(url=url, model=model) or True,
    )
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    assert local_runtime.cmd_start(_Args()) == 0
    assert warmed["model"] == local_runtime.configured_model()
    assert warmed["url"] == local_runtime.ollama_url()


def test_start_fails_when_llm_warmup_fails(monkeypatch, capsys):
    """A CPU-offloaded / failed warmup fails start — the API never launches."""
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "prewarm_llm_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: False)
    monkeypatch.setattr(
        local_runtime, "start_api_process",
        lambda host, port: (_ for _ in ()).throw(AssertionError("API must not start")),
    )
    assert local_runtime.cmd_start(_Args()) == 1
    assert "LLM warmup failed" in capsys.readouterr().out


def test_start_prewarms_stt_when_voice_enabled(monkeypatch, capsys):
    prewarmed = {}
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "voice_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt_enabled", lambda: True)
    monkeypatch.setattr(
        local_runtime, "prewarm_stt", lambda url: prewarmed.update(url=url) or True
    )
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    assert local_runtime.cmd_start(_Args()) == 0
    assert prewarmed["url"] == local_runtime.api_url()


def test_stt_prewarm_failure_never_fails_start(monkeypatch, capsys):
    """STT problems are reported but must not stop the API (/health unaffected)."""
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "voice_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt", lambda url: False)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    assert local_runtime.cmd_start(_Args()) == 0  # start still succeeds
    out = capsys.readouterr().out
    assert "STT prewarm failed" in out
    assert "/health is unaffected" in out
    assert "Runtime is up" in out


def test_restart_recovers_warm_state(monkeypatch, capsys):
    """restart = stop + start, and start re-warms the model automatically."""
    warm_calls = []
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: warm_calls.append(model) or True)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    monkeypatch.setattr(local_runtime, "keep_ollama_running", lambda: True)
    assert local_runtime.cmd_restart(_Args()) == 0
    assert warm_calls == [local_runtime.configured_model()]


def test_warm_subcommand(monkeypatch):
    calls = []
    monkeypatch.setattr(local_runtime, "warm_llm", lambda url, model: calls.append("llm") or True)
    monkeypatch.setattr(local_runtime, "voice_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt_enabled", lambda: True)
    monkeypatch.setattr(local_runtime, "prewarm_stt", lambda url: calls.append("stt") or True)
    parser = local_runtime.build_parser()
    args = parser.parse_args(["warm"])
    assert args.func is local_runtime.cmd_warm
    assert args.func(args) == 0
    assert calls == ["llm", "stt"]


def test_model_status_subcommand(monkeypatch):
    monkeypatch.setattr(local_runtime, "llm_model_status", lambda url, model: True)
    parser = local_runtime.build_parser()
    args = parser.parse_args(["model-status"])
    assert args.func(args) == 0


def test_unload_subcommand_releases_vram_only(monkeypatch):
    """unload delegates to the HTTP keep_alive=0 path — no docker/volume commands."""
    unloaded = {}
    monkeypatch.setattr(
        local_runtime, "unload_llm",
        lambda url, model: unloaded.update(url=url, model=model) or True,
    )
    ran = []
    monkeypatch.setattr(
        local_runtime.subprocess, "run",
        lambda *a, **k: ran.append(a) or (_ for _ in ()).throw(AssertionError("no subprocess")),
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["unload"])
    assert args.func(args) == 0
    assert unloaded["model"] == local_runtime.configured_model()
    assert ran == []  # never shells out — cannot touch containers or volumes


def test_wake_subcommand_launches_listener(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        local_runtime.fifi_wake, "run", lambda argv: launched.update(argv=argv) or 0
    )
    parser = local_runtime.build_parser()
    args = parser.parse_args(["wake"])
    assert args.func is local_runtime.cmd_wake
    assert args.func(args) == 0
    assert launched["argv"] == ["--server", local_runtime.api_url()]


def test_start_never_auto_starts_wake_listener(monkeypatch):
    """Wake listening is explicit-only: start must never launch it."""
    _all_up(monkeypatch)
    monkeypatch.setattr(local_runtime, "start_api_process", lambda host, port: object())
    monkeypatch.setattr(
        local_runtime.fifi_wake, "run",
        lambda argv: (_ for _ in ()).throw(AssertionError("wake must not auto-start")),
    )
    assert local_runtime.cmd_start(_Args()) == 0


def test_config_defaults_are_safe_and_local():
    monkey_free = local_runtime  # values read from env/.env; defaults asserted here
    assert monkey_free.api_host() in ("127.0.0.1", "localhost")
    # In a clean checkout these default true; .env may override, so only assert type.
    assert isinstance(monkey_free.auto_start_ollama(), bool)
    assert isinstance(monkey_free.require_gpu(), bool)
    assert isinstance(monkey_free.keep_ollama_running(), bool)
