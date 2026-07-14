"""Containerized Voice Lab: compose services, isolated image, and the
host|docker runtime switch (Phase 3D.x).

Covers the containerization contract WITHOUT Docker running — everything here is
static file assertions plus local_runtime dispatch with all process/HTTP/docker
calls stubbed. Nothing real is built, started, or downloaded.
"""

import sys
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import local_runtime  # noqa: E402


def compose_text() -> str:
    return (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")


def voice_lab_block() -> str:
    """Just the voice-lab service block (up to the next sibling service)."""
    body = compose_text().split("\n  voice-lab:", 1)[1]
    return body.split("\n  voice-model-seed:", 1)[0]


def seed_block() -> str:
    body = compose_text().split("\n  voice-model-seed:", 1)[1]
    return body.split("\n  voice-model-download:", 1)[0]


def download_block() -> str:
    body = compose_text().split("\n  voice-model-download:", 1)[1]
    return body.split("\nvolumes:", 1)[0]


# --- isolated image -----------------------------------------------------------------


def test_voice_lab_dockerfile_is_isolated_and_matches_setup():
    dockerfile = (PROJECT_ROOT / "voice_lab" / "Dockerfile").read_text(encoding="utf-8")
    # Same CUDA/PyTorch index setup.py uses (Blackwell/sm_120).
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    # Installs the proven requirements + the optional Qwen deps.
    assert "requirements.txt" in dockerfile
    assert "requirements-qwen.txt" in dockerfile
    # Runs uvicorn on 0.0.0.0:8766.
    assert "0.0.0.0" in dockerfile and "8766" in dockerfile
    # HF cache pinned to the mounted volume; SAPI/playback disabled in-container.
    assert "HF_HOME=/models/hf" in dockerfile
    assert "HF_HUB_CACHE=/models/hf/hub" in dockerfile
    assert "VOICE_LAB_DISABLE_SAPI=true" in dockerfile
    # The model cache must NOT be copied into the image (volume only).
    assert "COPY voice_lab/models" not in dockerfile


def test_root_dockerfile_was_not_turned_into_a_voice_image():
    """Requirement 1: the ROOT Dockerfile stays the simulated main API."""
    root = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "app.main:app" in root
    assert "8000" in root
    assert 'ENABLE_REAL_WINDOWS_TOOLS=false' in root
    assert "voice_lab" not in root  # never repurposed for the Voice Lab
    assert "cu128" not in root


# --- voice-lab service --------------------------------------------------------------


def test_voice_lab_service_binding_and_profile():
    block = voice_lab_block()
    assert "profiles:" in block and "- voice" in block
    assert '"127.0.0.1:8766:8766"' in block  # loopback only
    assert "restart: unless-stopped" in block
    assert "/health" in block  # healthcheck target
    assert "healthcheck:" in block
    assert "uvicorn" in block and "0.0.0.0" in block and "8766" in block


def test_voice_lab_service_env_and_mounts():
    block = voice_lab_block()
    assert "HF_HOME: /models/hf" in block
    assert "HF_HUB_CACHE: /models/hf/hub" in block
    assert "VOICE_LAB_PLAYBACK_ENABLED: \"false\"" in block  # never touch host audio
    assert "VOICE_LAB_DISABLE_SAPI: \"true\"" in block       # SAPI is host-only
    assert "voice-lab-models:/models/hf" in block            # named model volume
    # Shared state, read-write (profile selection + previews persist to host).
    assert "./config/voices:/srv/agent-automaton/config/voices" in block
    assert "./voice_lab/storage:/srv/agent-automaton/voice_lab/storage" in block


def test_voice_lab_config_voices_mount_is_read_write():
    """Profile activation must persist to the host — the mount is NOT :ro."""
    block = voice_lab_block()
    for line in block.splitlines():
        if "./config/voices:" in line:
            assert not line.rstrip().endswith(":ro"), line


def test_voice_lab_gpu_reservation_matches_ollama():
    block = voice_lab_block()
    assert "driver: nvidia" in block
    assert "count: all" in block
    assert "capabilities: [gpu]" in block
    assert "NVIDIA_VISIBLE_DEVICES: all" in block
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in block


# --- seed service -------------------------------------------------------------------


def test_voice_model_seed_service():
    block = seed_block()
    assert "- voice-seed" in block                       # profile
    assert "network_mode: none" in block                 # never use the network
    assert "./voice_lab/models/hf:/seed/hf:ro" in block  # host cache, read-only
    assert "voice-lab-models:/models/hf" in block        # destination volume
    assert "seed_models.py" in block


# --- download service ---------------------------------------------------------------


def test_voice_model_download_service():
    block = download_block()
    assert "- voice-download" in block                   # profile
    assert "voice-lab-models:/models/hf" in block        # same model volume
    assert "models.json" in block                        # ids from configuration
    assert "download_models.py" in block


# --- volume + naming ----------------------------------------------------------------


def test_voice_lab_models_volume_declared():
    volumes = compose_text().split("\nvolumes:", 1)[1]
    assert "voice-lab-models:" in volumes
    assert "ollama-models:" in volumes  # existing volume untouched


def test_no_docker_compose_yml_only_compose_yml():
    assert (PROJECT_ROOT / "compose.yml").exists()
    assert not (PROJECT_ROOT / "docker-compose.yml").exists()


# --- .dockerignore ------------------------------------------------------------------


def test_dockerignore_keeps_model_cache_out_of_build_context():
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "voice_lab/models/" in ignore     # the 14 GB cache never in context
    assert "voice_lab/.venv/" in ignore
    assert "voice_lab/storage/" in ignore
    assert "*.wav" in ignore                 # generated audio
    assert "*.log" in ignore                 # logs


# --- runtime switch: host | docker --------------------------------------------------


def test_runtime_defaults_to_host(monkeypatch):
    monkeypatch.setattr(local_runtime.llm_smoke, "config_value", lambda k, d="": d)
    assert local_runtime.voice_lab_runtime() == "host"


def test_runtime_reads_docker(monkeypatch):
    monkeypatch.setattr(
        local_runtime.llm_smoke, "config_value",
        lambda k, d="": "docker" if k == "VOICE_LAB_RUNTIME" else d,
    )
    assert local_runtime.voice_lab_runtime() == "docker"


def test_docker_mode_starts_compose_service(monkeypatch):
    monkeypatch.setattr(local_runtime, "voice_lab_runtime", lambda: "docker")
    monkeypatch.setattr(local_runtime, "docker_available", lambda: True)
    # down until the container comes up, then healthy.
    states = iter([None, {"engines_available": {}, "active_profile": "fifi_warm"}])
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: next(states, None))
    started = {}
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_container",
        lambda: started.setdefault("up", True) or True,
    )
    monkeypatch.setattr(
        local_runtime, "wait_for_voice_lab",
        lambda *a, **k: {"engines_available": {}, "active_profile": "fifi_warm"},
    )
    assert local_runtime._voice_lab_up() is True
    assert started == {"up": True}


def test_docker_mode_reuses_running_worker(monkeypatch):
    """A worker already answering on 8766 is reused, never duplicated."""
    monkeypatch.setattr(local_runtime, "voice_lab_runtime", lambda: "docker")
    monkeypatch.setattr(
        local_runtime, "voice_lab_health", lambda: {"status": "ok", "active_profile": "x"}
    )
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_container",
        lambda: (_ for _ in ()).throw(AssertionError("must not start a duplicate")),
    )
    assert local_runtime._voice_lab_up() is True


def test_host_worker_refused_when_runtime_is_docker(monkeypatch, capsys):
    """Requirement 6/8: never start a second HOST worker while docker owns 8766."""
    monkeypatch.setattr(local_runtime, "voice_lab_runtime", lambda: "docker")
    monkeypatch.setattr(
        local_runtime.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch a host worker")),
    )
    assert local_runtime.start_voice_lab_process() is None
    assert "refusing to start a HOST worker" in capsys.readouterr().out


def test_voice_stop_docker_stops_container(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "voice_lab_runtime", lambda: "docker")
    stopped = {}
    monkeypatch.setattr(
        local_runtime, "stop_voice_lab_container",
        lambda: stopped.setdefault("down", True) or True,
    )
    # Must not touch any host PID in docker mode.
    monkeypatch.setattr(
        local_runtime, "stop_owned_worker",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no host PID in docker mode")),
    )

    class Args:
        pass

    assert local_runtime.cmd_voice_stop(Args()) == 0
    assert stopped == {"down": True}
    assert "container" in capsys.readouterr().out


def test_voice_status_docker_reports_container(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "voice_lab_runtime", lambda: "docker")
    monkeypatch.setattr(local_runtime, "voice_lab_container_status", lambda: "running")
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "voice_lab_worker_status", lambda: None)
    monkeypatch.setattr(local_runtime.llm_smoke, "config_value", lambda k, d="": d)

    class Args:
        pass

    assert local_runtime.cmd_voice_status(Args()) == 0
    out = capsys.readouterr().out
    assert "Runtime         : docker" in out
    assert "Container       : running" in out


# --- main API fallback stays functional ---------------------------------------------


def test_main_api_tts_still_falls_back_when_worker_down(settings, monkeypatch):
    """Requirement 8: the main HOST API keeps its Windows/simulated fallback when
    the Voice Lab worker (host OR docker) is unreachable — it never raises."""
    import httpx

    from app.voice import tts

    settings.enable_voice = True
    settings.tts_engine = "voice_lab"

    def boom(*a, **k):
        raise httpx.ConnectError("worker down")

    monkeypatch.setattr(tts.httpx, "post", boom)
    service = tts.TextToSpeechService(engine="voice_lab")
    result = service.speak("hola")
    # Degrades gracefully (windows or simulated), reporting the fallback origin.
    assert result.get("fallback_from") == "voice_lab"
    assert "error" not in result or result.get("simulated") or result.get("spoke")
