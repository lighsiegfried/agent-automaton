"""GPU coordinator: headroom policy, ollama-unload opt-in, cache detection."""

import pytest

from app import gpu


@pytest.fixture(autouse=True)
def _no_real_probes(monkeypatch):
    monkeypatch.setattr(gpu, "ollama_loaded_models", lambda: [])
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda: {
        "name": "NVIDIA GeForce RTX 5060 Ti", "total_gb": 16.0, "used_gb": 8.0, "free_gb": 8.0,
    })
    monkeypatch.delenv("VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD", raising=False)
    # Cache detection now honors HF_HOME (Docker points it at the volume). Clear
    # it so MODELS_DIR-based tests stay deterministic regardless of the shell.
    monkeypatch.delenv("HF_HOME", raising=False)


def test_headroom_ok_when_vram_fits():
    decision = gpu.headroom_check(5.0)
    assert decision["ok"] is True


def test_headroom_refuses_with_actionable_message(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda: {
        "name": "X", "total_gb": 16.0, "used_gb": 14.0, "free_gb": 2.0,
    })
    monkeypatch.setattr(
        gpu, "ollama_loaded_models", lambda: [{"name": "qwen2.5:7b", "vram_gb": 4.4}]
    )
    decision = gpu.headroom_check(5.0)
    assert decision["ok"] is False
    message = decision["message"]
    assert "VRAM insuficiente" in message
    assert "local_runtime.py unload" in message  # concrete fix offered
    assert "VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD" in message  # and the opt-in
    assert ":\\" not in message  # no paths


def test_ollama_unload_is_opt_in_and_default_off(monkeypatch):
    assert gpu.allow_temporary_ollama_unload() is False  # DEFAULT FALSE
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda: {
        "name": "X", "total_gb": 16.0, "used_gb": 14.0, "free_gb": 2.0,
    })
    monkeypatch.setattr(
        gpu, "ollama_loaded_models", lambda: [{"name": "qwen2.5:7b", "vram_gb": 4.4}]
    )
    # Even with per-job consent, the global setting keeps it off:
    decision = gpu.headroom_check(5.0, allow_ollama_unload=True)
    assert decision["ok"] is False
    assert "unload_ollama_first" not in decision

    # Both the setting AND per-job consent are required:
    monkeypatch.setenv("VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD", "true")
    without_consent = gpu.headroom_check(5.0, allow_ollama_unload=False)
    assert without_consent["ok"] is False
    with_consent = gpu.headroom_check(5.0, allow_ollama_unload=True)
    assert with_consent["ok"] is True
    assert with_consent["unload_ollama_first"] is True


def test_headroom_allows_when_gpu_not_probeable(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda: {})
    decision = gpu.headroom_check(5.0)
    assert decision["ok"] is True  # engine-level OOM guard is the backstop


def test_unload_ollama_is_http_only():
    source = (gpu.__file__ and open(gpu.__file__, encoding="utf-8").read()) or ""
    assert "taskkill" not in source and "terminate" not in source
    assert "keep_alive" in source  # frees the model via the API, never a kill


# --- cache detection ---------------------------------------------------------------------


def test_model_cache_status(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "MODELS_DIR", tmp_path)
    assert gpu.model_cache_status("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") == "not_installed"
    snapshot = (
        tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        / "snapshots" / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert gpu.model_cache_status("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") == "cached"
    # engine-local ids never need a download
    assert gpu.model_cache_status("") == "cached"


def test_download_watcher_reports_real_bytes(monkeypatch):
    """When a download IS happening, progress carries actual byte growth."""
    import time

    from app.engines.qwen3_tts import Qwen3TtsEngine
    from app import progress

    sizes = iter([1000, 5_000_000, 12_000_000])
    monkeypatch.setattr(gpu, "cache_dir_size_bytes", lambda: next(sizes, 12_000_000))
    events = []
    progress.set_reporter(lambda step, **f: events.append((step, f)))
    try:
        stop = Qwen3TtsEngine._watch_download("Qwen/X", poll_seconds=0.02)
        time.sleep(0.15)
        stop.set()
    finally:
        progress.clear_reporter()
    downloads = [f for step, f in events if step == "downloading" and "bytes_downloaded" in f]
    assert downloads, "watcher never reported download bytes"
    assert downloads[0]["bytes_downloaded"] > 0


def test_resource_report_has_no_paths_or_secrets(monkeypatch):
    monkeypatch.setattr(gpu, "whisper_state", lambda: {"reachable": False, "loaded": False})
    report = gpu.resource_report()
    import json

    text = json.dumps(report)
    assert ":\\" not in text and "C:/" not in text
    assert "voice_design" in text and "clone" in text and "kokoro" in text
    assert report["allow_temporary_ollama_unload"] is False
