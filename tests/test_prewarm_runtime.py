"""Unit tests for scripts/prewarm_runtime.py — persistent GPU model warmup.

All HTTP is mocked: no Docker, Ollama, GPU, or API process is needed. These
tests pin the warmup contract:
- warmup sends an EMPTY /api/generate with keep_alive=-1 (indefinite);
- the model must be verified through /api/ps and be FULLY GPU-loaded;
- CPU offload fails the warmup unless explicitly permitted;
- unload uses keep_alive=0 and can never delete models or volumes;
- STT prewarm reports the /voice/preflight result and never raises;
- compose keeps Ollama pinned warm (keep-alive env + model-keeper service).
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prewarm_runtime  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GB = 2**30


class FakeResponse:
    def __init__(self, payload=None, error=False):
        self._payload = payload or {}
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise httpx.HTTPError("boom")

    def json(self):
        return self._payload


def _entry(name="qwen2.5:7b", size=6 * GB, vram=6 * GB, expires="never"):
    return {"name": name, "size": size, "size_vram": vram, "expires_at": expires}


def _wire(monkeypatch, *, ps_models=None, post_error=False, ps_error=False):
    """Route httpx.post/get inside prewarm_runtime; returns the captured posts."""
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append({"url": url, "json": json})
        return FakeResponse(error=post_error)

    def fake_get(url, timeout=None):
        assert url.endswith("/api/ps")
        return FakeResponse({"models": ps_models or []}, error=ps_error)

    monkeypatch.setattr(prewarm_runtime.httpx, "post", fake_post)
    monkeypatch.setattr(prewarm_runtime.httpx, "get", fake_get)
    return posts


# --- LLM warmup --------------------------------------------------------------------


def test_warm_sends_empty_generate_with_indefinite_keep_alive(monkeypatch):
    posts = _wire(monkeypatch, ps_models=[_entry()])
    assert prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lambda *_: None
    )
    assert len(posts) == 1
    assert posts[0]["url"] == "http://x/api/generate"
    assert posts[0]["json"] == {"model": "qwen2.5:7b", "prompt": "", "keep_alive": -1}


def test_warm_reports_model_processor_vram_and_duration(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry(vram=6 * GB, size=6 * GB)])
    lines = []
    assert prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lines.append
    )
    text = "\n".join(lines)
    assert "Model loaded    : qwen2.5:7b" in text
    assert "Processor       : gpu" in text
    assert "VRAM usage      : 6.0 GB" in text
    assert "Load duration   :" in text
    assert "stays loaded until an explicit unload" in text


def test_warm_fails_when_cpu_offloaded_and_not_permitted(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry(vram=0)])
    lines = []
    assert not prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lines.append
    )
    text = "\n".join(lines)
    assert "CPU-offloaded" in text
    assert "ALLOW_CPU_OLLAMA" in text  # the fix is spelled out


def test_warm_partial_gpu_offload_also_fails(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry(vram=3 * GB, size=6 * GB)])
    lines = []
    assert not prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lines.append
    )
    assert "gpu+cpu (50% in VRAM)" in "\n".join(lines)


def test_warm_cpu_offload_passes_only_when_explicitly_permitted(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry(vram=0)])
    lines = []
    assert prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=True, out=lines.append
    )
    assert "WARNING" in "\n".join(lines)


def test_warm_fails_when_model_not_listed_by_ps(monkeypatch):
    _wire(monkeypatch, ps_models=[])  # generate succeeded but nothing stayed loaded
    lines = []
    assert not prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lines.append
    )
    assert "did not stay in memory" in "\n".join(lines)


def test_warm_fails_cleanly_on_http_error(monkeypatch):
    _wire(monkeypatch, post_error=True)
    lines = []
    assert not prewarm_runtime.warm_llm(
        "http://x", "qwen2.5:7b", keep_alive=-1, allow_cpu=False, out=lines.append
    )
    assert "LLM warmup FAILED" in "\n".join(lines)


def test_keep_alive_defaults_to_indefinite(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_MODEL_LOADED", "true")
    assert prewarm_runtime.warm_keep_alive() == -1
    monkeypatch.setenv("OLLAMA_KEEP_MODEL_LOADED", "false")
    assert prewarm_runtime.warm_keep_alive() == "5m"


def test_full_gpu_required_by_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUIRE_FULL_GPU", "true")
    monkeypatch.setenv("ALLOW_CPU_OLLAMA", "false")
    assert not prewarm_runtime.cpu_offload_allowed()
    monkeypatch.setenv("ALLOW_CPU_OLLAMA", "true")
    assert prewarm_runtime.cpu_offload_allowed()
    monkeypatch.setenv("ALLOW_CPU_OLLAMA", "false")
    monkeypatch.setenv("OLLAMA_REQUIRE_FULL_GPU", "false")
    assert prewarm_runtime.cpu_offload_allowed()


# --- unload ------------------------------------------------------------------------


def test_unload_uses_keep_alive_zero(monkeypatch):
    posts = _wire(monkeypatch, ps_models=[])
    lines = []
    assert prewarm_runtime.unload_model("http://x", "qwen2.5:7b", out=lines.append)
    assert posts[0]["json"] == {"model": "qwen2.5:7b", "prompt": "", "keep_alive": 0}
    assert "Nothing was deleted" in "\n".join(lines)


def test_unload_never_shells_out():
    """Unload is HTTP-only: the script cannot run docker, rm, or any volume
    command at all — nothing it does can delete the model volume."""
    assert not hasattr(prewarm_runtime, "subprocess")
    source = (PROJECT_ROOT / "scripts" / "prewarm_runtime.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "docker " not in source.lower()


def test_unload_warns_when_keeper_rewarms(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry()])  # still loaded after the unload call
    lines = []
    assert prewarm_runtime.unload_model("http://x", "qwen2.5:7b", out=lines.append)
    assert "model-keeper" in "\n".join(lines)


# --- model status ------------------------------------------------------------------


def test_model_status_reports_loaded_model(monkeypatch):
    _wire(monkeypatch, ps_models=[_entry()])
    lines = []
    assert prewarm_runtime.model_status("http://x", "qwen2.5:7b", out=lines.append)
    text = "\n".join(lines)
    assert "Processor       : gpu" in text
    assert "VRAM usage" in text


def test_model_status_when_not_loaded(monkeypatch):
    _wire(monkeypatch, ps_models=[])
    lines = []
    assert not prewarm_runtime.model_status("http://x", "qwen2.5:7b", out=lines.append)
    assert "NOT loaded" in "\n".join(lines)


# --- STT prewarm -------------------------------------------------------------------


def _wire_preflight(monkeypatch, payload=None, error=False):
    def fake_get(url, timeout=None):
        assert url.endswith("/voice/preflight")
        return FakeResponse(payload or {}, error=error)

    monkeypatch.setattr(prewarm_runtime.httpx, "get", fake_get)


def test_stt_prewarm_reports_preflight(monkeypatch):
    _wire_preflight(
        monkeypatch,
        {
            "passed": True,
            "device": "cuda",
            "compute_type": "float16",
            "model": "small",
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
        },
    )
    lines = []
    assert prewarm_runtime.prewarm_stt("http://api", out=lines.append)
    text = "\n".join(lines)
    assert "device=cuda" in text
    assert "compute=float16" in text
    assert "gpu=NVIDIA GeForce RTX 5060 Ti" in text
    assert "passed=True" in text


def test_stt_prewarm_failure_is_reported_not_raised(monkeypatch):
    _wire_preflight(monkeypatch, {"passed": False, "message": "CUDA broke"})
    lines = []
    assert not prewarm_runtime.prewarm_stt("http://api", out=lines.append)
    text = "\n".join(lines)
    assert "STT prewarm FAILED" in text
    assert "/health is unaffected" in text  # failure never stops the API


def test_stt_prewarm_skips_when_voice_disabled(monkeypatch):
    _wire_preflight(monkeypatch, {"status": "disabled"})
    lines = []
    assert prewarm_runtime.prewarm_stt("http://api", out=lines.append)
    assert "voice is disabled" in "\n".join(lines)


def test_stt_prewarm_survives_connection_error(monkeypatch):
    _wire_preflight(monkeypatch, error=True)
    lines = []
    assert not prewarm_runtime.prewarm_stt("http://api", out=lines.append)
    assert "STT prewarm FAILED" in "\n".join(lines)


# --- compose: pinned-warm Ollama + model-keeper -------------------------------------


def compose_text() -> str:
    return (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")


def test_compose_ollama_keeps_model_loaded():
    ollama_block = compose_text().split("ollama:", 1)[1].split("model-keeper:", 1)[0]
    assert 'OLLAMA_KEEP_ALIVE: "-1"' in ollama_block
    assert 'OLLAMA_MAX_LOADED_MODELS: "1"' in ollama_block
    assert 'OLLAMA_NUM_PARALLEL: "1"' in ollama_block
    # GPU-first configuration and the persistent model volume are untouched.
    assert "driver: nvidia" in ollama_block
    assert "ollama-models:/root/.ollama" in ollama_block


def test_compose_model_keeper_rewarms_after_restart():
    keeper_block = compose_text().split("model-keeper:", 1)[1]
    assert "restart: unless-stopped" in keeper_block  # survives Ollama restarts
    assert "llm" in keeper_block.split("depends_on")[0]  # same profile as ollama
    assert "/api/generate" in keeper_block
    assert '\\"keep_alive\\":-1' in keeper_block
    assert "/api/ps" in keeper_block  # checks before warming


def test_compose_model_keeper_cannot_touch_volumes_or_host():
    import re

    # Bound the keeper block at the next service (voice-lab) so it stays precise
    # now that model-keeper is no longer the last service in the file. Strip
    # comment lines — the following service's documentation may mention docker.
    raw = compose_text().split("model-keeper:", 1)[1].split("\n  voice-lab:", 1)[0]
    keeper_block = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("#")
    )
    assert "volumes:" not in keeper_block  # no mounts at all
    assert "docker" not in keeper_block  # no docker socket / CLI
    assert not re.search(r"\brm\b", keeper_block)  # no delete commands
    assert "delete" not in keeper_block


def test_no_duplicate_model_configuration():
    """Planner and response generator share one model — the keeper warms that
    single model; nothing configures a second copy."""
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    models = {
        line.split("=", 1)[1].strip()
        for line in env_example.splitlines()
        if line.startswith(("LLM_PLANNER_MODEL=", "RESPONSE_MODEL=", "DEFAULT_MODEL="))
    }
    assert models == {"qwen2.5:7b"}
    assert "${LLM_PLANNER_MODEL:-qwen2.5:7b}" in compose_text()
