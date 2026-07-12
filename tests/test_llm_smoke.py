"""Unit tests for scripts/llm_smoke.py and scripts/docker_llm.py.
All subprocess and HTTP calls are mocked — no Docker, Ollama, or GPU needed."""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import docker_llm  # noqa: E402
import llm_smoke  # noqa: E402


def fake_response(status_code=200, payload=None, url="http://x") -> httpx.Response:
    return httpx.Response(
        status_code, json=payload or {}, request=httpx.Request("GET", url)
    )


# --- ollama / model helpers ------------------------------------------------------


def test_ollama_available_true(monkeypatch):
    monkeypatch.setattr(llm_smoke.httpx, "get", lambda url, timeout: fake_response(200))
    assert llm_smoke.ollama_available("http://x") is True


def test_ollama_available_handles_connection_error(monkeypatch):
    def boom(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_smoke.httpx, "get", boom)
    assert llm_smoke.ollama_available("http://x") is False


def test_model_present(monkeypatch):
    payload = {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3:8b"}]}
    monkeypatch.setattr(
        llm_smoke.httpx, "get", lambda url, timeout: fake_response(200, payload)
    )
    assert llm_smoke.model_present("http://x", "qwen2.5:7b") is True
    assert llm_smoke.model_present("http://x", "mistral:7b") is False


def test_ensure_model_pulls_when_missing(monkeypatch):
    pulled = []
    monkeypatch.setattr(llm_smoke, "model_present", lambda base, model: False)
    monkeypatch.setattr(
        llm_smoke, "pull_model", lambda base, model: pulled.append(model) or True
    )
    ok, detail = llm_smoke.ensure_model("http://x", "qwen2.5:7b")
    assert ok
    assert pulled == ["qwen2.5:7b"]


def test_ensure_model_skip_pull_fails_instead_of_downloading(monkeypatch):
    monkeypatch.setattr(llm_smoke, "model_present", lambda base, model: False)
    monkeypatch.setattr(
        llm_smoke,
        "pull_model",
        lambda base, model: pytest.fail("pull_model must not be called"),
    )
    ok, detail = llm_smoke.ensure_model("http://x", "qwen2.5:7b", skip_pull=True)
    assert not ok
    assert "skip-pull" in detail


# --- safety guards ---------------------------------------------------------------


def test_verify_api_safe_refuses_real_tools():
    error = llm_smoke.verify_api_safe({"real_windows_tools_enabled": True})
    assert error is not None and "refusing" in error
    assert llm_smoke.verify_api_safe({"real_windows_tools_enabled": False}) is None


def test_start_api_never_enables_real_tools(monkeypatch):
    captured = {}

    def fake_popen(cmd, cwd=None, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return object()

    monkeypatch.setattr(llm_smoke.subprocess, "Popen", fake_popen)
    llm_smoke.start_api(8123, "http://x", "some-model")
    assert captured["env"]["ENABLE_REAL_WINDOWS_TOOLS"] == "false"
    assert captured["env"]["ENABLE_LLM_PLANNER"] == "true"
    assert captured["env"]["ENABLE_RESPONSE_GENERATOR"] == "true"
    assert captured["env"]["LLM_PLANNER_MODEL"] == "some-model"
    assert "uvicorn" in captured["cmd"]


def test_destructive_case_forbids_execution():
    destructive = [c for c in llm_smoke.COMMAND_CASES if "delete" in c["text"]]
    assert destructive, "smoke test must include a destructive attempt"
    assert "executed" in destructive[0]["forbid_status"]
    assert "delete_file" in destructive[0]["forbid_tools"]


# --- command evaluation (pure logic) ------------------------------------------------


def test_evaluate_command():
    case = {"expect_intent": "open_folder", "expect_status": ["simulated"]}
    good = {
        "planner": "llm_planner",
        "intent": "open_folder",
        "status": "simulated",
        "tool": "open_folder",
        "assistant_message": "Listo.",
    }
    assert llm_smoke.evaluate_command(case, good)[0] is True
    assert llm_smoke.evaluate_command(case, {**good, "intent": "open_app"})[0] is False
    assert llm_smoke.evaluate_command(case, {**good, "status": "executed"})[0] is False
    assert llm_smoke.evaluate_command(case, {**good, "assistant_message": ""})[0] is False

    forbid = {"forbid_status": ["executed"], "forbid_tools": ["delete_file"]}
    assert llm_smoke.evaluate_command(forbid, {**good, "status": "executed"})[0] is False
    assert llm_smoke.evaluate_command(forbid, {**good, "tool": "delete_file"})[0] is False
    assert llm_smoke.evaluate_command(forbid, good)[0] is True


# --- docker helper --------------------------------------------------------------------


def test_compose_up_command_is_profile_gated():
    assert docker_llm.COMPOSE_UP_OLLAMA == [
        "docker", "compose", "--profile", "llm", "up", "-d", "ollama",
    ]


def test_wait_for_ollama(monkeypatch):
    monkeypatch.setattr(llm_smoke, "ollama_available", lambda url: True)
    assert docker_llm.wait_for_ollama("http://x", timeout_seconds=1.0) is True
    monkeypatch.setattr(llm_smoke, "ollama_available", lambda url: False)
    assert docker_llm.wait_for_ollama("http://x", timeout_seconds=0.0) is False


def test_docker_llm_aborts_cleanly_without_docker(monkeypatch, capsys):
    monkeypatch.setattr(docker_llm, "docker_available", lambda: False)
    assert docker_llm.main() == 1
    assert "Docker is not available" in capsys.readouterr().out


# --- GPU-first behavior ----------------------------------------------------------


@pytest.fixture
def docker_llm_ready(monkeypatch):
    """Pretend docker/compose/ollama are all up; leave GPU + smoke to each test."""
    monkeypatch.setattr(docker_llm, "docker_available", lambda: True)
    monkeypatch.setattr(docker_llm, "start_ollama", lambda: True)
    monkeypatch.setattr(docker_llm, "wait_for_ollama", lambda url, **kw: True)
    monkeypatch.setattr(docker_llm, "get_ollama_container_id", lambda: "abc123def")
    # Model present + warmup succeeds by default (Phase 3D.1); tests override.
    monkeypatch.setattr(
        docker_llm.llm_smoke, "ensure_model", lambda url, model, **kw: (True, "model present")
    )
    monkeypatch.setattr(
        docker_llm.prewarm_runtime, "warm_llm", lambda url, model, **kw: True
    )


def test_refuses_cpu_when_gpu_missing_by_default(docker_llm_ready, monkeypatch, capsys):
    monkeypatch.delenv("ALLOW_CPU_OLLAMA", raising=False)
    monkeypatch.setattr(llm_smoke, "read_env_file", lambda: {})  # .env can't opt in either
    monkeypatch.setattr(docker_llm, "detect_gpu", lambda cid: (False, "no GPU"))
    monkeypatch.setattr(
        docker_llm.llm_smoke,
        "main",
        lambda argv: pytest.fail("smoke test must not run on CPU by default"),
    )
    assert docker_llm.main() == 1
    out = capsys.readouterr().out
    assert "GPU detected    : NO" in out
    assert "ALLOW_CPU_OLLAMA=true" in out  # actionable escape hatch mentioned


def test_allows_cpu_only_with_explicit_opt_in(docker_llm_ready, monkeypatch, capsys):
    monkeypatch.setenv("ALLOW_CPU_OLLAMA", "true")
    monkeypatch.setattr(docker_llm, "detect_gpu", lambda cid: (False, "no GPU"))
    monkeypatch.setattr(docker_llm.llm_smoke, "main", lambda argv: 0)
    assert docker_llm.main() == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "CPU-only" in out


def test_gpu_detected_runs_smoke_and_reports(docker_llm_ready, monkeypatch, capsys):
    monkeypatch.delenv("ALLOW_CPU_OLLAMA", raising=False)
    monkeypatch.setattr(
        docker_llm, "detect_gpu", lambda cid: (True, "NVIDIA GeForce RTX 5060 Ti")
    )
    monkeypatch.setattr(docker_llm.llm_smoke, "main", lambda argv: 0)
    assert docker_llm.main() == 0
    out = capsys.readouterr().out
    assert "Compose profile : llm" in out
    assert "Container       : abc123def" in out
    assert "GPU detected    : yes" in out
    assert "NVIDIA GeForce RTX 5060 Ti" in out
    assert "Smoke test      : PASS" in out


def test_docker_llm_warms_model_before_smoke(docker_llm_ready, monkeypatch):
    """Phase 3D.1: the model is preloaded (keep_alive=-1) before the smoke test."""
    order = []
    monkeypatch.setattr(
        docker_llm, "detect_gpu", lambda cid: (True, "NVIDIA GeForce RTX 5060 Ti")
    )
    monkeypatch.setattr(
        docker_llm.prewarm_runtime, "warm_llm",
        lambda url, model, **kw: order.append("warm") or True,
    )
    monkeypatch.setattr(
        docker_llm.llm_smoke, "main", lambda argv: order.append("smoke") or 0
    )
    assert docker_llm.main() == 0
    assert order == ["warm", "smoke"]


def test_docker_llm_fails_when_warmup_fails(docker_llm_ready, monkeypatch):
    monkeypatch.setattr(
        docker_llm, "detect_gpu", lambda cid: (True, "NVIDIA GeForce RTX 5060 Ti")
    )
    monkeypatch.setattr(
        docker_llm.prewarm_runtime, "warm_llm", lambda url, model, **kw: False
    )
    monkeypatch.setattr(
        docker_llm.llm_smoke, "main",
        lambda argv: pytest.fail("smoke must not run when the warmup failed"),
    )
    assert docker_llm.main() == 1


def test_detect_gpu_prefers_nvidia_smi(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        class Result:
            returncode = 0
            stdout = "NVIDIA GeForce RTX 5060 Ti\n"
            stderr = ""

        assert "nvidia-smi" in cmd
        return Result()

    monkeypatch.setattr(docker_llm.subprocess, "run", fake_run)
    assert docker_llm.detect_gpu("abc") == (True, "NVIDIA GeForce RTX 5060 Ti")


def test_detect_gpu_falls_back_to_logs(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        class Result:
            returncode = 0 if cmd[:2] == ["docker", "logs"] else 1
            stdout = (
                'msg="inference compute" id=GPU-1 library=cuda name="NVIDIA GeForce RTX 5060 Ti"'
                if cmd[:2] == ["docker", "logs"]
                else ""
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(docker_llm.subprocess, "run", fake_run)
    detected, detail = docker_llm.detect_gpu("abc")
    assert detected is True
    assert "NVIDIA" in detail


def test_detect_gpu_reports_absence(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        class Result:
            returncode = 1
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(docker_llm.subprocess, "run", fake_run)
    detected, detail = docker_llm.detect_gpu("abc")
    assert detected is False
    assert "no NVIDIA GPU" in detail
