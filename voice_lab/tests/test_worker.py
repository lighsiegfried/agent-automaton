"""Worker endpoint tests — engines, playback, and profiles all faked/local."""

import json

import pytest
from fastapi.testclient import TestClient

import app.main as worker
from conftest import VOICE_LAB_ROOT, FakeEngine


def _scan_for_absolute_paths(value) -> list[str]:
    """Collect any response string that looks like an absolute/internal path."""
    hits: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            hits += _scan_for_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            hits += _scan_for_absolute_paths(child)
    elif isinstance(value, str):
        if value.startswith(("/audio/", "/previews/")):
            return hits  # relative API routes, not filesystem paths
        if ":\\" in value or value.startswith(("/", "\\")) or str(VOICE_LAB_ROOT) in value:
            hits.append(value)
    return hits


@pytest.fixture
def engines(monkeypatch):
    fakes = {
        "windows_sapi": FakeEngine("windows_sapi"),
        "kokoro": FakeEngine("kokoro"),
        "qwen3_tts": FakeEngine("qwen3_tts"),
    }
    monkeypatch.setattr(worker, "get_engine", lambda name: fakes[name])
    monkeypatch.setattr(
        worker, "loaded_engines", lambda: {n: e.loaded for n, e in fakes.items()}
    )
    monkeypatch.setattr(
        worker,
        "unload_all",
        lambda only=None: [
            e.name
            for e in ([fakes[only]] if only else fakes.values())
            if e.unload()
        ],
    )
    return fakes


@pytest.fixture
def played(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        worker, "play_wav", lambda path: calls.append(str(path)) or {"played": True}
    )
    return calls


@pytest.fixture
def client(engines, played, manager, monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "manager", manager)
    monkeypatch.setattr(worker, "PREVIEWS_DIR", tmp_path / "previews")
    with TestClient(worker.app) as test_client:
        yield test_client


# --- read endpoints ------------------------------------------------------------------


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["app"] == "fifi-voice-lab"
    assert set(data["engines_available"]) == {"windows_sapi", "kokoro", "qwen3_tts"}
    assert _scan_for_absolute_paths(data) == []


def test_profiles_lists_shared_profiles(client, manager):
    manager.select("fifi_warm")
    data = client.get("/profiles").json()
    assert [p["name"] for p in data["profiles"]] == ["fifi_calm", "fifi_warm"]
    assert data["active"] == "fifi_warm"
    assert _scan_for_absolute_paths(data) == []


def test_status_reports_loaded_engines_without_paths(client, engines):
    data = client.get("/status").json()
    assert data["status"] == "ok"
    assert data["engines_loaded"] == {
        "windows_sapi": False, "kokoro": False, "qwen3_tts": False,
    }
    assert "uptime_seconds" in data
    assert _scan_for_absolute_paths(data) == []


# --- synthesis -----------------------------------------------------------------------


def test_preview_uses_active_profile_and_plays(client, manager, engines, played):
    manager.select("fifi_calm")
    data = client.post("/preview", json={}).json()
    assert data["status"] == "ok"
    assert data["profile"] == "fifi_calm"
    assert data["engine"] == "kokoro"
    assert data["audio_valid"] is True
    assert data["fallback_used"] is False
    assert data["playback"] == {"played": True}
    assert len(played) == 1
    assert _scan_for_absolute_paths(data) == []


def test_synthesize_without_play(client, engines, played):
    data = client.post(
        "/synthesize", json={"text": "hola", "profile": "fifi_warm", "play": False}
    ).json()
    assert data["status"] == "ok"
    assert "playback" not in data
    assert played == []
    assert data["file"].startswith("storage/") or "/" not in data["file"]


def test_engine_failure_falls_back_to_windows_sapi(client, engines):
    engines["kokoro"].fail = "synthesize"
    data = client.post("/synthesize", json={"text": "hola", "profile": "fifi_warm"}).json()
    assert data["status"] == "ok"
    assert data["engine"] == "windows_sapi"  # profile fallback engine
    assert data["fallback_used"] is True
    assert "kokoro" in data["fallback_reason"]
    assert _scan_for_absolute_paths(data) == []


def test_total_failure_is_a_structured_error(client, engines):
    engines["kokoro"].fail = "synthesize"
    engines["windows_sapi"].dep_ok = False
    response = client.post("/synthesize", json={"text": "hola", "profile": "fifi_warm"})
    assert response.status_code == 200  # never a 500
    data = response.json()
    assert data["status"] == "error"
    assert "kokoro" in data["message"] and "windows_sapi" in data["message"]
    assert _scan_for_absolute_paths(data) == []


def test_fallback_order_selected_then_kokoro_then_sapi(client, engines, manager, voices_dir):
    """Phase 3D.0.1 order: selected neural engine -> kokoro -> windows_sapi."""
    import json as json_module

    payload = json_module.loads(
        (voices_dir / "profiles" / "fifi_warm.json").read_text(encoding="utf-8")
    )
    payload.update(name="fifi_qwen", engine="qwen3_tts")
    (voices_dir / "profiles" / "fifi_qwen.json").write_text(
        json_module.dumps(payload), encoding="utf-8"
    )
    engines["qwen3_tts"].fail = "synthesize"
    data = client.post("/synthesize", json={"text": "hola", "profile": "fifi_qwen"}).json()
    assert data["status"] == "ok"
    assert data["engine"] == "kokoro"  # kokoro BEFORE windows_sapi
    assert data["requested_engine"] == "qwen3_tts"
    assert data["fallback_used"] is True
    assert engines["windows_sapi"].synth_calls == 0  # never reached


def test_gpu_oom_falls_back_safely(client, engines):
    """A CUDA OOM is a structured failure that walks the chain, never a crash."""
    engines["kokoro"].fail = "oom"
    data = client.post("/synthesize", json={"text": "hola", "profile": "fifi_warm"}).json()
    assert data["status"] == "ok"
    assert data["engine"] == "windows_sapi"
    assert "out of memory" in data["fallback_reason"].lower()


def test_warm_endpoint_is_async_and_idempotent(client, engines, played, manager, fast_infra):
    """Phase 3D.0.4: /warm answers 202 with a pollable job; a second call for
    an already-warm engine answers ready IMMEDIATELY without a second load."""
    response = client.post("/warm", json={"profile": "fifi_warm"})
    assert response.status_code == 202
    job = response.json()
    assert job["job_id"] and job["initial_status"] in ("queued", "preparing")
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    assert final["result"]["ready"] is True
    assert final["result"]["engine"] == "kokoro"
    assert engines["kokoro"].loaded  # engine is now warm
    assert engines["kokoro"].load_calls == 1
    assert played == []  # warm never plays audio
    assert _scan_for_absolute_paths(final) == []

    again = client.post("/warm", json={"profile": "fifi_warm"})
    assert again.status_code == 200  # idempotent: immediate ready
    data = again.json()
    assert data["status"] == "ready" and data["already_loaded"] is True
    assert data["last_warm_job"]["status"] == "completed"
    assert engines["kokoro"].load_calls == 1  # NEVER a duplicate load


def test_warm_duplicate_request_returns_same_job(client, engines, manager, fast_infra):
    import threading

    gate = threading.Event()
    original = engines["kokoro"]._load

    def slow_load(profile):
        gate.wait(5)
        return original(profile)

    engines["kokoro"]._load = slow_load
    first = client.post("/warm", json={"profile": "fifi_warm"})
    assert first.status_code == 202
    try:
        second = client.post("/warm", json={"profile": "fifi_warm"})
        assert second.status_code == 202
        body = second.json()
        assert body["job_id"] == first.json()["job_id"]  # SAME job, no dup load
        assert body["deduplicated"] is True
        # /health stays fast and truthful while warming
        health = client.get("/health").json()
        assert health["status"] == "ok"
    finally:
        gate.set()
    _wait_job(client, first.json()["job_id"])


def test_preview_uses_the_exact_comparison_sentence(client, engines):
    client.post("/preview", json={"profile": "fifi_warm"})
    assert engines["kokoro"].texts == [
        "Hola, soy Fifi. El sistema está listo y puedo ayudarte con tus tareas. "
        "¿En qué trabajaremos hoy?"
    ]


def test_unknown_profile_is_a_clean_error(client):
    data = client.post("/synthesize", json={"text": "hola", "profile": "ghost"}).json()
    assert data["status"] == "error"
    assert "Unknown profile" in data["message"]
    assert _scan_for_absolute_paths(data) == []


# --- selection + unload --------------------------------------------------------------


def test_profile_select_updates_active_json(client, manager):
    data = client.post("/profile/select", json={"profile": "fifi_calm"}).json()
    assert data["status"] == "ok"
    assert data["active"]["profile"] == "fifi_calm"
    on_disk = json.loads(manager.active_path.read_text(encoding="utf-8"))
    assert on_disk["profile"] == "fifi_calm"
    assert _scan_for_absolute_paths(data) == []


def test_profile_select_unknown_is_clean(client, manager):
    data = client.post("/profile/select", json={"profile": "ghost"}).json()
    assert data["status"] == "error"
    assert manager.active() is None  # nothing was written


def test_unload_releases_loaded_engines(client, engines):
    client.post("/synthesize", json={"text": "hola", "profile": "fifi_warm"})
    assert engines["kokoro"].loaded
    data = client.post("/unload", json={}).json()
    assert data == {"status": "ok", "unloaded": ["kokoro"]}
    assert not engines["kokoro"].loaded


# --- Voice Designer endpoints (Phase 3D.0.2) ------------------------------------------


@pytest.fixture
def wired_designer(client, monkeypatch, tmp_path):
    from app.designer import VoiceDesigner
    from conftest import write_wav

    def fake_synth(text, profile, out_path):
        write_wav(out_path)
        return {"status": "ok", "engine": "qwen3_tts", "seconds": 0.4, "duration_seconds": 3.0}

    instance = VoiceDesigner(
        root=tmp_path,
        identities_dir=tmp_path / "identities",
        profiles_dir=tmp_path / "profiles",
        designs_dir=tmp_path / "designs",
        synth=fake_synth,
    )
    monkeypatch.setattr(worker, "designer", instance)
    monkeypatch.setattr(worker, "list_designs", lambda: [])
    return instance


def test_ui_is_served_locally(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Voice Designer" in response.text
    assert "Activar para Fifi" in response.text  # the activation button exists


def test_designer_state_endpoint(client, wired_designer):
    data = client.get("/designer").json()
    assert data["status"] == "ok"
    assert data["default_mode"] == "voice_design_then_clone"
    assert "voice_design" in data["modes"] and "reference_clone" in data["modes"]
    assert set(data["options"]) >= {"age", "gender", "timbre", "pitch", "energy",
                                    "speed", "accent", "emotion", "personality"}
    assert _scan_for_absolute_paths(data) == []


def test_designer_generate_freeze_and_activate_flow(client, wired_designer, manager):
    generated = client.post(
        "/designer/generate",
        json={"profile_name": "fifi_luna", "language": "es", "count": 3,
              "timbre": "aterciopelado", "preview_text": "Hola, soy Fifi."},
    ).json()
    assert generated["status"] == "ok"
    assert len(generated["variants"]) == 3
    assert _scan_for_absolute_paths(generated) == []

    variant = generated["variants"][0]
    audio = client.get(
        f"/designer/audio?profile_name=fifi_luna&variant_id={variant['id']}"
    )
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")

    frozen = client.post(
        "/designer/freeze", json={"profile_name": "fifi_luna", "variant_id": variant["id"]}
    ).json()
    assert frozen["status"] == "ok"
    assert frozen["design"]["reference"]["text"] == "Hola, soy Fifi."
    assert _scan_for_absolute_paths(frozen) == []

    # Activation is atomic and visible via /profile/active. The frozen profile
    # lives in the designer's (tmp) profiles dir — point the manager there too.
    manager.profiles_dir = wired_designer.profiles_dir
    selected = client.post("/profile/select", json={"profile": "fifi_luna"}).json()
    assert selected["status"] == "ok"
    active = client.get("/profile/active").json()
    assert active["profile"] == "fifi_luna"
    assert active["updated_utc"]  # change metadata for hot reload


def test_designer_audio_rejects_bad_names(client, wired_designer):
    response = client.get("/designer/audio?profile_name=..%2Fetc&variant_id=x")
    data = response.json()
    assert data["status"] == "error"
    assert ":\\" not in data["message"] and "/" not in data.get("message", "")[:2]


def test_designer_delete_variant_endpoint(client, wired_designer, tmp_path):
    generated = client.post(
        "/designer/generate",
        json={"profile_name": "fifi_luna", "language": "es", "count": 3},
    ).json()
    variant = generated["variants"][0]
    deleted = client.delete(
        f"/designer/variant?profile_name=fifi_luna&variant_id={variant['id']}"
    ).json()
    assert deleted["status"] == "ok"
    assert not (tmp_path / variant["file"]).exists()


def test_designer_reference_requires_authorization(client, wired_designer):
    response = client.post(
        "/designer/reference",
        data={"profile_name": "voz_ajena", "language": "es", "transcript": "hola",
              "authorization_statement": ""},
        files={"file": ("ref.wav", b"RIFFdata", "audio/wav")},
    ).json()
    assert response["status"] == "error"
    assert "authorization" in response["message"]


# --- background jobs / auditions / models (Phase 3D.0.3) --------------------------------


import time as _time


def _wait_job(client, job_id, timeout=15.0):
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in ("completed", "failed", "cancelled"):
            return job
        _time.sleep(0.03)
    raise AssertionError(f"job {job_id} never finished: {job}")


@pytest.fixture
def fast_infra(monkeypatch, tmp_path):
    """Fast, isolated job infrastructure: no nvidia-smi, tmp dirs everywhere."""
    from app import auditions as auditions_module
    from app import gpu as gpu_module
    from app import resources as resources_module
    import app.jobs as jobs_module

    GiB = resources_module.GiB
    monkeypatch.setattr(gpu_module, "gpu_snapshot", lambda: {
        "name": "RTX 5060 Ti", "total_gb": 16.0, "used_gb": 8.0, "free_gb": 8.0,
    })
    monkeypatch.setattr(gpu_module, "system_ram_snapshot", lambda: {
        "total_gb": 16.0, "used_gb": 8.0, "free_gb": 8.0,
    })
    # Canonical resource service (3D.0.4) — byte-exact fakes, plenty of room.
    monkeypatch.setattr(resources_module, "system_ram", lambda: {
        "total_bytes": 16 * GiB, "available_bytes": 8 * GiB, "used_bytes": 8 * GiB,
    })
    monkeypatch.setattr(resources_module, "dedicated_vram", lambda: {
        "total_bytes": 16 * GiB, "used_bytes": 8 * GiB, "free_bytes": 8 * GiB,
        "name": "RTX 5060 Ti", "driver": "610.62", "source": "test",
    })
    monkeypatch.setattr(resources_module, "disk", lambda: {
        "total_bytes": 200 * GiB, "free_bytes": 50 * GiB, "used_bytes": 150 * GiB,
    })
    monkeypatch.setattr(gpu_module, "ollama_loaded_models", lambda: [])
    monkeypatch.setattr(gpu_module, "whisper_state", lambda: {"reachable": False, "loaded": False})
    monkeypatch.setattr(jobs_module, "JOBS_TMP_DIR", tmp_path / "jobs-tmp")
    monkeypatch.setattr(jobs_module, "JOBS_META_DIR", tmp_path / "jobs-meta")
    monkeypatch.setattr(auditions_module, "AUDITIONS_DIR", tmp_path / "auditions")
    return tmp_path


def test_profile_preview_job_with_cache(client, engines, manager, fast_infra):
    """First audition synthesizes; the identical second one is a cache hit."""
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Hola prueba."}
    ).json()
    first = _wait_job(client, job["job_id"])
    assert first["status"] == "completed"
    result = first["result"]
    assert result["cache_hit"] is False
    assert result["engine"] == "kokoro"  # the engine ACTUALLY used
    assert result["requested_engine"] == "kokoro"
    assert result["seconds"] is not None and result["profile"] == "fifi_warm"
    audio = client.get(f"/previews/audition/{result['key']}")
    assert audio.status_code == 200  # playable in the browser

    again = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Hola prueba."}
    ).json()
    second = _wait_job(client, again["job_id"])
    assert second["result"]["cache_hit"] is True  # no new synthesis
    assert engines["kokoro"].synth_calls == 1

    forced = client.post(
        "/jobs/profile-preview",
        json={"profile": "fifi_warm", "text": "Hola prueba.", "force": True},
    ).json()
    assert _wait_job(client, forced["job_id"])["result"]["cache_hit"] is False
    assert engines["kokoro"].synth_calls == 2  # Regenerar prueba re-synthesizes


def test_preview_reports_fallback_engine(client, engines, manager, fast_infra):
    engines["kokoro"].fail = "synthesize"
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Con fallback."}
    ).json()
    final = _wait_job(client, job["job_id"])
    result = final["result"]
    assert result["requested_engine"] == "kokoro"
    assert result["engine"] == "windows_sapi"  # fallback clearly reported
    assert result["fallback_used"] is True


def test_compare_job_generates_both_profiles(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-compare",
        json={"profile_a": "fifi_warm", "profile_b": "fifi_calm", "text": "La misma frase."},
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    result = final["result"]
    assert result["a"]["profile"] == "fifi_warm"
    assert result["b"]["profile"] == "fifi_calm"
    assert result["a"]["key"] != result["b"]["key"]
    assert result["a"]["text"] == result["b"]["text"] == "La misma frase."
    assert _scan_for_absolute_paths(result) == []


def test_unknown_profile_preview_refused_at_preflight(client, engines, manager, fast_infra):
    """Phase 3D.0.3a: an unknown profile never even enters the queue — the
    refusal is structured (error_code + Spanish user_message), never generic."""
    response = client.post("/jobs/profile-preview", json={"profile": "ghost"})
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "unknown_profile"
    assert "no existe" in data["user_message"]
    assert data["retryable"] is False
    assert "worker" in data and "memory" in data
    assert "job_id" not in data  # nothing was queued


def test_health_stays_fast_while_heavy_job_runs(client, engines, manager, fast_infra):
    import threading

    gate = threading.Event()
    original = engines["kokoro"]._synthesize

    def slow_synth(text, profile, out_path):
        gate.wait(5)
        return original(text, profile, out_path)

    engines["kokoro"]._synthesize = slow_synth
    job = client.post("/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Lento."}).json()
    try:
        started = _time.monotonic()
        health = client.get("/health").json()
        elapsed = _time.monotonic() - started
        assert health["status"] == "ok"
        assert elapsed < 1.0  # never blocked by the running job
        assert health["worker"] in ("busy", "warming")  # busy is NOT unhealthy
        ready = client.get("/ready").json()
        assert "worker" in ready
    finally:
        gate.set()
    _wait_job(client, job["job_id"])
    assert client.get("/health").json()["worker"] == "idle"


def test_voice_design_job_blocked_by_insufficient_vram(client, engines, manager, monkeypatch, fast_infra):
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "headroom_check", lambda *a, **k: {
        "ok": False, "message": "VRAM insuficiente: libera el modelo de Ollama",
    })
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_test_vram", "language": "es"}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "failed"
    assert final["error_code"] == "insufficient_vram"
    assert "VRAM insuficiente" in final["error"]
    assert engines["qwen3_tts"].synth_calls == 0  # refused BEFORE any load


def test_freeze_job_unloads_voicedesign_when_tight(client, engines, manager, wired_designer, monkeypatch, fast_infra):
    """Sequential policy: VoiceDesign frees the card before the clone loads."""
    import app.main as worker_module
    from app import gpu as gpu_module
    from app.engines import base as base_module

    generated = client.post(
        "/designer/generate", json={"profile_name": "fifi_seq", "language": "es", "count": 3},
    ).json()
    variant = generated["variants"][0]

    unloaded = []

    class FakeQwen:
        _models = {"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": object()}

        def unload_model(self, model_id):
            unloaded.append(model_id)
            return True

    monkeypatch.setitem(base_module._instances, "qwen3_tts", FakeQwen())
    monkeypatch.setattr(gpu_module, "headroom_check", lambda *a, **k: {"ok": False, "message": "tight"})
    # Freeze must write into the manager's VALID store (the preflight checks it).
    wired_designer.profiles_dir = manager.profiles_dir

    job = client.post(
        "/jobs/freeze", json={"profile_name": "fifi_seq", "variant_id": variant["id"]}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    assert unloaded == ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]  # a. design freed
    assert final["result"]["warm"]["status"] == "ok"  # d. clone warmed after


def test_job_cancellation_endpoint(client, engines, manager, fast_infra):
    import threading

    gate = threading.Event()
    original = engines["kokoro"]._synthesize

    def slow_synth(text, profile, out_path):
        from app import progress

        for _ in range(100):
            progress.check_cancelled()
            if gate.wait(0.05):
                break
        return original(text, profile, out_path)

    engines["kokoro"]._synthesize = slow_synth
    job = client.post("/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Cancélame."}).json()
    _time.sleep(0.15)
    cancelled = client.post(f"/jobs/{job['job_id']}/cancel").json()
    final = _wait_job(client, job["job_id"])
    gate.set()
    assert final["status"] == "cancelled"
    active = client.get("/jobs/active").json()
    assert all(j["job_id"] != job["job_id"] for j in active["jobs"])


def test_models_status_endpoint(client, engines, fast_infra, monkeypatch):
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "model_cache_status", lambda mid: "cached")
    data = client.get("/models/status").json()
    assert data["status"] == "ok"
    assert data["gpu"]["name"] == "RTX 5060 Ti"
    assert data["voice_lab"]["voice_design"]["cache"] == "cached"  # cached ≠ downloading
    assert data["allow_temporary_ollama_unload"] is False
    assert _scan_for_absolute_paths(data) == []


def test_models_unload_endpoint_keeps_cache(client, engines, manager, fast_infra):
    client.post("/synthesize", json={"text": "hola", "profile": "fifi_warm"})
    assert engines["kokoro"].loaded
    data = client.post("/models/unload", json={}).json()
    assert data["status"] == "ok"
    assert "kokoro" in data["unloaded"]
    assert "no se borran" in data["note"]  # cached files untouched


# --- profile details + deletion ----------------------------------------------------------


def test_profiles_details_metadata_and_secrecy(client, manager, fast_infra):
    manager.select("fifi_warm")
    data = client.get("/profiles/details").json()
    assert data["active"] == "fifi_warm"
    profiles = {p["name"]: p for p in data["profiles"]}
    warm = profiles["fifi_warm"]
    assert warm["active"] is True  # the badge
    assert warm["mode"] == "built-in"
    assert warm["deletable"] is False
    assert warm["fallback_chain"][0] == "kokoro"
    assert warm["created"] and warm["updated"]
    assert _scan_for_absolute_paths(data) == []


def test_builtin_profile_deletion_refused(client, manager, fast_infra):
    response = client.delete("/profiles/fifi_warm").json()
    assert response["status"] == "error"
    assert "integrado" in response["message"]
    assert (manager.profiles_dir / "fifi_warm.json").exists()  # untouched


def test_user_profile_deletion(client, engines, manager, wired_designer, fast_infra):
    generated = client.post(
        "/designer/generate", json={"profile_name": "fifi_borrable", "language": "es", "count": 3},
    ).json()
    frozen = client.post(
        "/designer/freeze",
        json={"profile_name": "fifi_borrable", "variant_id": generated["variants"][0]["id"]},
    ).json()
    assert frozen["status"] == "ok"
    manager.profiles_dir = wired_designer.profiles_dir

    import app.main as worker_module

    original_designer = worker_module.designer
    response = client.delete("/profiles/fifi_borrable").json()
    assert response["status"] == "ok"
    assert not (wired_designer.profiles_dir / "fifi_borrable.json").exists()
    assert not (wired_designer.identities_dir / "fifi_borrable").exists()


def test_active_profile_cannot_be_deleted(client, engines, manager, wired_designer, fast_infra):
    generated = client.post(
        "/designer/generate", json={"profile_name": "fifi_activa", "language": "es", "count": 3},
    ).json()
    client.post(
        "/designer/freeze",
        json={"profile_name": "fifi_activa", "variant_id": generated["variants"][0]["id"]},
    )
    manager.profiles_dir = wired_designer.profiles_dir
    manager.select("fifi_activa")
    response = client.delete("/profiles/fifi_activa").json()
    assert response["status"] == "error"
    assert "activa" in response["message"]


# --- isolation / binding -------------------------------------------------------------


def test_worker_binds_loopback_only():
    from app.config import get_settings

    settings = get_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8766


def test_worker_caches_are_pinned_inside_voice_lab(monkeypatch):
    from app import config as lab_config

    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    lab_config.pin_cache_env()
    import os

    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        assert os.environ[var].startswith(str(lab_config.VOICE_LAB_ROOT))


# --- Phase 3D.0.3a: identity, preflight, pressure, exclusive mode -----------------------


from app.build import API_BUILD  # noqa: E402


def _make_qwen_profile(voices_dir, name="fifi_qwen"):
    import json as json_module

    payload = json_module.loads(
        (voices_dir / "profiles" / "fifi_warm.json").read_text(encoding="utf-8")
    )
    payload.update(name=name, engine="qwen3_tts")
    (voices_dir / "profiles" / f"{name}.json").write_text(
        json_module.dumps(payload), encoding="utf-8"
    )
    return name


def test_worker_info_identity(client):
    import os as os_module

    data = client.get("/worker/info").json()
    assert data["status"] == "ok"
    assert data["pid"] == os_module.getpid()
    assert data["api_build"] == API_BUILD
    assert isinstance(data["venv_isolated"], bool)
    for key in ("source_fingerprint", "repo_fingerprint", "python_fingerprint"):
        assert isinstance(data[key], str) and len(data[key]) == 12
    assert data["started_utc"] and data["version"]
    assert _scan_for_absolute_paths(data) == []


def test_ready_reports_store_env_and_engines(client, fast_infra):
    data = client.get("/ready").json()
    assert data["ready"] is True
    assert data["profile_store"]["valid"] is True
    assert data["profile_store"]["profile_count"] == 2
    assert data["environment_ok"] is True
    assert set(data["engines_available"]) == {"windows_sapi", "kokoro", "qwen3_tts"}
    assert data["api_build"] == API_BUILD
    assert _scan_for_absolute_paths(data) == []


def test_ui_build_mismatch_is_refused(client, engines, fast_infra):
    response = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_x", "ui_build": "stale-build"}
    )
    assert response.status_code == 409
    data = response.json()
    assert data["error_code"] == "version_mismatch"
    assert data["user_message"] == (
        "La interfaz y el worker tienen versiones diferentes. Reinicia Voice Lab."
    )
    assert data["retryable"] is False
    assert "job_id" not in data  # never queued
    assert client.get("/jobs/active").json()["jobs"] == []


def test_matching_ui_build_is_accepted(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-preview",
        json={"profile": "fifi_warm", "text": "Compatibles.", "ui_build": API_BUILD},
    ).json()
    assert job["job_id"]
    assert _wait_job(client, job["job_id"])["status"] == "completed"


def test_conflicting_heavy_job_is_refused_with_details(
    client, engines, manager, voices_dir, fast_infra
):
    import threading

    _make_qwen_profile(voices_dir)
    gate = threading.Event()
    original = engines["qwen3_tts"]._synthesize

    def slow_synth(text, profile, out_path):
        gate.wait(5)
        return original(text, profile, out_path)

    engines["qwen3_tts"]._synthesize = slow_synth
    first = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_qwen", "text": "Pesado."}
    ).json()
    assert first["job_id"]
    try:
        response = client.post(
            "/jobs/voice-design", json={"profile_name": "fifi_nueva", "language": "es"}
        )
        assert response.status_code == 409
        data = response.json()
        assert data["error_code"] == "heavy_job_running"
        assert data["retryable"] is True
        assert data["active_heavy_job"]["job_id"] == first["job_id"]
        assert "trabajo pesado" in data["user_message"]
    finally:
        gate.set()
    _wait_job(client, first["job_id"])


def test_insufficient_system_ram_is_refused(client, engines, manager, monkeypatch, fast_infra):
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "system_ram", lambda: {
        "total_bytes": 16 * GiB, "available_bytes": 1 * GiB, "used_bytes": 15 * GiB,
    })
    response = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_ram", "language": "es"}
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "insufficient_system_ram"
    assert data["user_message"] == "No hay suficiente memoria RAM disponible."
    assert data["retryable"] is True
    assert data["memory"]["constrained"] == "ram"
    assert "system_ram" in data["technical"]  # resource type + measured + required
    assert "requerido" in data["technical"]
    assert data["preflight"]["checks"]["system_ram"]["ok"] is False
    assert client.get("/jobs/active").json()["jobs"] == []  # never queued
    assert engines["qwen3_tts"].synth_calls == 0


def test_insufficient_vram_is_refused(client, engines, manager, monkeypatch, fast_infra):
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "dedicated_vram", lambda: {
        "total_bytes": 16 * GiB, "used_bytes": 15 * GiB, "free_bytes": 1 * GiB,
        "name": "RTX 5060 Ti", "source": "test",
    })
    response = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_vram", "language": "es"}
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "insufficient_vram"
    assert data["user_message"] == "No hay suficiente memoria VRAM disponible."
    assert data["memory"]["constrained"] == "vram"
    assert client.get("/jobs/active").json()["jobs"] == []


def test_insufficient_disk_is_refused_with_storage_message(
    client, engines, manager, monkeypatch, fast_infra
):
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "disk", lambda: {
        "total_bytes": 200 * GiB, "free_bytes": 2 * GiB, "used_bytes": 198 * GiB,
    })
    response = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_disk", "language": "es"}
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "insufficient_disk"
    assert data["user_message"] == "No hay suficiente espacio de almacenamiento."
    assert data["memory"]["constrained"] == "disk"
    # disk is NOT confused with RAM or VRAM anywhere in the message
    assert "RAM" not in data["user_message"] and "VRAM" not in data["user_message"]


def test_used_vram_alone_never_blocks_when_free_passes_threshold(
    client, engines, manager, wired_designer, monkeypatch, fast_infra
):
    """The 3D.0.3 false rejection: 8-11 GB VRAM in USE must not block while
    the FREE amount passes the byte-exact threshold (2.5 GB)."""
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "dedicated_vram", lambda: {
        # 11.2 GiB used, 4.8 GiB free — exactly the live nvidia-smi picture.
        "total_bytes": 16 * GiB, "used_bytes": int(11.2 * GiB),
        "free_bytes": int(4.8 * GiB), "name": "RTX 5060 Ti", "source": "test",
    })
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "headroom_check", lambda *a, **k: {"ok": True})
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_ok", "language": "es"}
    ).json()
    assert job.get("job_id"), job  # accepted — no false insufficient-VRAM
    assert _wait_job(client, job["job_id"])["status"] == "completed"


def test_resident_model_skips_memory_gate(
    client, engines, manager, wired_designer, monkeypatch, fast_infra
):
    """A model already on the GPU needs no new memory — low RAM must not
    block work with an already-loaded model."""
    from app import gpu as gpu_module
    from app.engines import base as base_module
    from app.engines.qwen3_tts import DEFAULT_MODEL_ID

    class FakeQwen:
        _models = {DEFAULT_MODEL_ID: object()}

        def unload_model(self, model_id):
            return False

    monkeypatch.setitem(base_module._instances, "qwen3_tts", FakeQwen())
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "system_ram", lambda: {
        "total_bytes": 16 * GiB, "available_bytes": 1 * GiB, "used_bytes": 15 * GiB,
    })
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_res", "language": "es"}
    ).json()
    assert job.get("job_id"), job  # accepted despite low RAM
    assert _wait_job(client, job["job_id"])["status"] == "completed"


def test_cached_model_reports_cache_never_download(
    client, engines, manager, wired_designer, monkeypatch, fast_infra
):
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "model_cache_status", lambda mid: "cached")
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_cache", "language": "es"}
    ).json()
    assert job["model_cache"] == "cached"  # found locally: no download claimed
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    assert final["bytes_downloaded"] is None


def test_store_degraded_is_reported_and_never_repaired(client, engines, manager, fast_infra):
    manager.active_path.parent.mkdir(parents=True, exist_ok=True)
    manager.active_path.write_text(
        json.dumps({"schema_version": 1, "profile": "ghost"}), encoding="utf-8"
    )
    ready = client.get("/ready").json()
    assert ready["ready"] is False
    assert ready["profile_store"]["valid"] is False
    assert "ghost" in ready["profile_store"]["error"]

    response = client.post("/jobs/profile-preview", json={"profile": "fifi_warm"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "profile_store_invalid"

    # The broken selection is REPORTED, never silently replaced.
    on_disk = json.loads(manager.active_path.read_text(encoding="utf-8"))
    assert on_disk["profile"] == "ghost"


def test_heavy_job_records_resource_snapshot(
    client, engines, manager, voices_dir, fast_infra
):
    _make_qwen_profile(voices_dir, "fifi_snap")
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_snap", "text": "Con snapshot."}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    snapshot = final["resources_before"]
    assert snapshot["ram"]["free_gb"] == 8.0
    assert snapshot["gpu"]["name"] == "RTX 5060 Ti"
    assert "kokoro_loaded" in snapshot and "qwen_models_loaded" in snapshot
    assert _scan_for_absolute_paths(final) == []


def test_light_job_skips_resource_snapshot(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Ligero."}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert "resources_before" not in final


def test_exclusive_design_mode_frees_unused_models(monkeypatch, fast_infra):
    from types import SimpleNamespace

    import app.main as worker_module
    from app import gpu as gpu_module
    from app.engines import base as base_module
    from app.engines.qwen3_tts import CLONE_MODEL_ID

    freed = {"kokoro": [], "models": []}

    class FakeQwen:
        _models = {CLONE_MODEL_ID: object()}

        def unload_model(self, model_id):
            freed["models"].append(model_id)
            return True

    monkeypatch.setitem(base_module._instances, "qwen3_tts", FakeQwen())
    monkeypatch.setattr(
        base_module, "unload_all", lambda only=None: freed["kokoro"].append(only) or []
    )
    monkeypatch.setattr(
        worker_module, "get_settings",
        lambda: SimpleNamespace(exclusive_design_mode=True, min_free_system_ram_gb=3.0),
    )
    from app import resources as resources_module

    monkeypatch.setattr(
        resources_module, "check_resources",
        lambda: {"checks": {"system_ram": {"ok": True, "free_gb": 8.0, "required_gb": 2.0}},
                 "blocking": None, "message": ""},
    )
    monkeypatch.setattr(gpu_module, "headroom_check", lambda *a, **k: {"ok": True})

    worker_module._gate_headroom("voice_design", False)
    assert freed["kokoro"] == ["kokoro"]  # kokoro unloaded first
    assert freed["models"] == [CLONE_MODEL_ID]  # unused clone model freed too


# --- Phase 3D.0.4: resources endpoint, structured preflight, playable previews ---------


def test_resources_status_endpoint(client, engines, fast_infra):
    data = client.get("/resources/status").json()
    assert data["status"] == "ok"
    assert data["system_ram_available_gb"] == 8.0
    assert data["dedicated_vram_free_gb"] == 8.0
    assert data["disk_free_gb"] == 50.0
    assert data["gpu_name"] == "RTX 5060 Ti"
    assert data["thresholds"]["min_free_vram_gb"] == 2.0
    assert data["thresholds"]["min_free_disk_gb"] == 5.0
    assert data["blocking"] is None
    assert _scan_for_absolute_paths(data) == []


def test_jobs_preflight_endpoint_structure(client, engines, manager, fast_infra):
    data = client.post("/jobs/preflight", json={"kind": "voice_design"}).json()
    assert data["allowed"] is True
    assert data["blocking_reason"] is None
    for key in ("profile_store", "engine", "model_cache", "system_ram",
                "vram", "disk", "job_queue", "worker_environment"):
        assert key in data["checks"], key
    summary = data["summary"]
    assert summary["resultado"] == "listo"
    assert summary["ram_libre_gb"] == 8.0
    assert summary["vram_libre_gb"] == 8.0
    assert summary["almacenamiento_libre_gb"] == 50.0
    assert summary["motor_solicitado"] == "qwen3_tts"
    assert _scan_for_absolute_paths(data) == []


def test_jobs_preflight_blocked_reports_reason(client, engines, manager, monkeypatch, fast_infra):
    from app import resources as resources_module

    GiB = resources_module.GiB
    monkeypatch.setattr(resources_module, "dedicated_vram", lambda: {
        "total_bytes": 16 * GiB, "used_bytes": 15 * GiB, "free_bytes": 1 * GiB,
        "name": "X", "source": "test",
    })
    data = client.post("/jobs/preflight", json={"kind": "voice_design"}).json()
    assert data["allowed"] is False
    assert data["blocking_reason"] == "insufficient_vram"
    assert data["summary"]["resultado"] == "bloqueado"
    assert data["summary"]["motivo"] == "No hay suficiente memoria VRAM disponible."


def test_loaded_idle_model_is_never_a_rejection(client, engines, manager, monkeypatch, fast_infra):
    """Spec 8: a loaded idle model is valid — preflight reports loaded=True
    and stays allowed (generation reuses the instance)."""
    from app.engines import base as base_module
    from app.engines.qwen3_tts import DEFAULT_MODEL_ID

    class FakeQwen:
        _models = {DEFAULT_MODEL_ID: object()}

    monkeypatch.setitem(base_module._instances, "qwen3_tts", FakeQwen())
    data = client.post("/jobs/preflight", json={"kind": "voice_design"}).json()
    assert data["allowed"] is True
    assert data["checks"]["model_cache"]["loaded"] is True
    assert data["checks"]["vram"].get("skipped")  # resident model: no new VRAM


def test_preview_result_has_playable_audio_url(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "URL estable."}
    ).json()
    result = _wait_job(client, job["job_id"])["result"]
    assert result["preview_id"] == result["key"]
    assert result["audio_url"].startswith("/audio/previews/")
    assert "?v=" in result["audio_url"]  # versioned: regeneration busts caches
    assert result["requested_profile"] == "fifi_warm"
    assert result["actual_engine"] == "kokoro"
    assert result["synthesis_seconds"] is not None
    assert result["audio_duration_seconds"] is not None
    assert result["created_at"]
    assert _scan_for_absolute_paths(result) == []

    audio = client.get(result["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert int(audio.headers["content-length"]) > 44  # nonzero, beyond WAV header
    assert audio.headers["cache-control"] == "no-store"
    assert audio.content[:4] == b"RIFF"  # a real WAV, playable in the browser


def test_preview_audio_missing_is_404(client, fast_infra):
    response = client.get("/audio/previews/fifi_warm-deadbeef")
    assert response.status_code == 404
    assert response.json()["error_code"] == "preview_not_found"


def test_preview_audio_rejects_traversal(client, fast_infra):
    # Encoded slashes: either the route refuses to match (FastAPI 404) or the
    # strict id validator rejects it — never a file read outside auditions/.
    response = client.get("/audio/previews/..%2F..%2Fsecrets")
    assert response.status_code == 404
    # A literal dotted id hits OUR validator explicitly:
    direct = client.get("/audio/previews/..%5C..%5Csecrets")
    assert direct.status_code == 404
    plain = client.get("/audio/previews/badid!!")
    assert plain.status_code == 404
    assert plain.json()["error_code"] == "invalid_preview_id"


def test_preview_cache_hit_and_regenerate_bypass(client, engines, manager, fast_infra):
    body = {"profile": "fifi_warm", "text": "Caché de preview."}
    first = _wait_job(client, client.post("/jobs/profile-preview", json=body).json()["job_id"])
    assert first["result"]["cache_hit"] is False
    second = _wait_job(client, client.post("/jobs/profile-preview", json=body).json()["job_id"])
    assert second["result"]["cache_hit"] is True
    assert engines["kokoro"].synth_calls == 1  # reused, no new synthesis
    forced = _wait_job(client, client.post(
        "/jobs/profile-preview", json={**body, "force": True}
    ).json()["job_id"])
    assert forced["result"]["cache_hit"] is False  # Regenerar bypasses the cache
    assert engines["kokoro"].synth_calls == 2


def test_preview_cache_invalidated_when_profile_changes(client, engines, manager, voices_dir, fast_infra):
    """The cache hash covers the profile VERSION: editing the profile file
    yields a fresh preview, never a stale voice."""
    body = {"profile": "fifi_warm", "text": "Versión del perfil."}
    _wait_job(client, client.post("/jobs/profile-preview", json=body).json()["job_id"])
    profile_path = voices_dir / "profiles" / "fifi_warm.json"
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    data["speed"] = 1.2  # a synthesis-setting change
    profile_path.write_text(json.dumps(data), encoding="utf-8")
    again = _wait_job(client, client.post("/jobs/profile-preview", json=body).json()["job_id"])
    assert again["result"]["cache_hit"] is False  # fingerprint changed
    assert engines["kokoro"].synth_calls == 2


def test_generation_validates_every_variant_and_reports_partial_failure(
    client, engines, manager, wired_designer, fast_infra
):
    """Spec 22: never completed unless every requested variant has VALID audio;
    partial failure names the failing variants."""
    calls = {"n": 0}
    real_write = __import__("conftest").write_wav

    def flaky_synth(text, profile, out_path):
        calls["n"] += 1
        if calls["n"] == 2:  # second variant produces an empty (invalid) file
            out_path.write_bytes(b"")
            return {"status": "ok", "engine": "qwen3_tts", "seconds": 0.1}
        real_write(out_path)
        return {"status": "ok", "engine": "qwen3_tts", "seconds": 0.1,
                "duration_seconds": 0.3}

    wired_designer.synth = flaky_synth
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_parcial", "language": "es"}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "failed"  # NOT completed with 2/3 outputs
    assert final["error_code"] == "partial_generation_failed"
    assert "variante 2" in final["error"]  # the failing variant is identified
    assert "2 variantes válidas" in final["error"]


def test_generation_with_zero_valid_outputs_never_completes(
    client, engines, manager, wired_designer, fast_infra
):
    def broken_synth(text, profile, out_path):
        out_path.write_bytes(b"not-a-wav")
        return {"status": "ok", "engine": "qwen3_tts", "seconds": 0.1}

    wired_designer.synth = broken_synth
    job = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_roto", "language": "es"}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "failed"
    assert final["status"] != "completed"  # zero audio files can never "finish"


def test_job_endpoints_answer_202(client, engines, manager, fast_infra):
    response = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "202."}
    )
    assert response.status_code == 202  # asynchronous acceptance, not 200
    _wait_job(client, response.json()["job_id"])


def test_heavy_concurrency_is_bounded(monkeypatch, tmp_path):
    import threading

    import app.jobs as jobs_module
    from app import gpu as gpu_module

    monkeypatch.setattr(jobs_module, "JOBS_TMP_DIR", tmp_path / "jt")
    monkeypatch.setattr(jobs_module, "JOBS_META_DIR", tmp_path / "jm")
    monkeypatch.setattr(gpu_module, "gpu_snapshot", lambda: {})
    monkeypatch.setattr(gpu_module, "pre_job_snapshot", lambda: {})
    bounded = jobs_module.JobManager(heavy_slots=1)
    gate = threading.Event()
    running = []

    def slow(job, tmp_dir):
        running.append(job["job_id"])
        gate.wait(5)
        return {}

    first = bounded.submit("heavy", slow, heavy=True)
    deadline = _time.monotonic() + 3
    while not running and _time.monotonic() < deadline:
        _time.sleep(0.02)
    second = bounded.submit("heavy", slow, heavy=True)
    _time.sleep(0.3)
    assert running == [first["job_id"]]  # the second heavy job waits for the slot
    gate.set()
    deadline = _time.monotonic() + 5
    while _time.monotonic() < deadline:
        states = {bounded.get(j["job_id"])["status"] for j in (first, second)}
        if states == {"completed"}:
            break
        _time.sleep(0.05)
    assert bounded.get(second["job_id"])["status"] == "completed"


# --- Phase 3D.0.3a HOTFIX: disconnect, stale jobs, heartbeat, persistence ----------------


def test_health_carries_worker_instance_identity(client):
    from app.jobs import WORKER_INSTANCE_ID, WORKER_STARTED_UTC

    data = client.get("/health").json()
    assert data["instance_id"] == WORKER_INSTANCE_ID
    assert data["started_utc"] == WORKER_STARTED_UTC
    info = client.get("/worker/info").json()
    assert info["instance_id"] == WORKER_INSTANCE_ID


def test_job_records_carry_instance_and_heartbeat(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Identidad."}
    ).json()
    from app.jobs import WORKER_INSTANCE_ID

    assert job["worker_instance_id"] == WORKER_INSTANCE_ID
    assert job["worker_started_utc"]
    assert job["heartbeat_utc"]
    final = _wait_job(client, job["job_id"])
    assert final["stalled"] is False


def test_job_metadata_is_persisted_atomically(client, engines, manager, fast_infra):
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Persistente."}
    ).json()
    final = _wait_job(client, job["job_id"])
    meta_path = fast_infra / "jobs-meta" / f"{job['job_id']}.json"
    assert meta_path.is_file()
    persisted = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    assert persisted["worker_instance_id"] == final["worker_instance_id"]
    assert "_monotonic_start" not in persisted  # public metadata only
    assert not list((fast_infra / "jobs-meta").glob("*.tmp"))  # atomic writes


def test_restart_marks_inflight_jobs_interrupted(monkeypatch, tmp_path):
    """A restarted worker NEVER pretends to resume an in-memory GPU job."""
    import app.jobs as jobs_module

    meta = tmp_path / "meta"
    tmp_jobs = tmp_path / "tmp-jobs"
    monkeypatch.setattr(jobs_module, "JOBS_META_DIR", meta)
    monkeypatch.setattr(jobs_module, "JOBS_TMP_DIR", tmp_jobs)
    meta.mkdir(parents=True)
    (tmp_jobs / "deadbeef0001").mkdir(parents=True)
    (tmp_jobs / "deadbeef0001" / "partial.wav").write_bytes(b"x")
    (meta / "deadbeef0001.json").write_text(json.dumps({
        "job_id": "deadbeef0001", "job_type": "voice_design", "status": "loading",
        "message": "Cargando…", "model": "Qwen/X", "worker_instance_id": "oldworker111",
    }), encoding="utf-8")
    (meta / "deadbeef0002.json").write_text(json.dumps({
        "job_id": "deadbeef0002", "job_type": "profile_preview", "status": "completed",
        "result": {"ok": True}, "worker_instance_id": "oldworker111",
    }), encoding="utf-8")

    restarted = jobs_module.JobManager()  # simulates the new worker process
    interrupted = restarted.get("deadbeef0001")
    assert interrupted["status"] == "interrupted"
    assert interrupted["error_code"] == "worker_restarted"
    assert "se reinició" in interrupted["error"]
    assert not (tmp_jobs / "deadbeef0001").exists()  # temp files cleaned
    history = restarted.get("deadbeef0002")
    assert history["status"] == "completed"  # completed history survives
    assert history["result"] == {"ok": True}
    # Interrupted/history jobs are terminal: the worker starts idle.
    assert restarted.worker_status() == "idle"
    assert restarted.active() == []

    on_disk = json.loads((meta / "deadbeef0001.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "interrupted"  # persisted for the UI to see


def test_interrupted_job_served_after_restart_via_endpoint(
    client, engines, manager, monkeypatch, fast_infra
):
    """The UI polls /jobs/{id} after reconnect — it must get the interrupted
    record, not a 404, when metadata survived the restart."""
    import app.jobs as jobs_module
    import app.main as worker_module

    meta = fast_infra / "jobs-meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "beefbeef0001.json").write_text(json.dumps({
        "job_id": "beefbeef0001", "job_type": "voice_design", "status": "transferring",
        "worker_instance_id": "deadworker000",
    }), encoding="utf-8")
    monkeypatch.setattr(worker_module, "jobs", jobs_module.JobManager())
    data = client.get("/jobs/beefbeef0001").json()
    assert data["status"] == "interrupted"
    assert data["worker_instance_id"] == "deadworker000"  # NOT this worker's id


def test_running_job_heartbeat_advances_and_stall_detection(monkeypatch, tmp_path):
    import app.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_TMP_DIR", tmp_path / "jt")
    monkeypatch.setattr(jobs_module, "JOBS_META_DIR", tmp_path / "jm")
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "gpu_snapshot", lambda: {})
    manager2 = jobs_module.JobManager()
    import threading

    gate = threading.Event()
    job = manager2.submit("slow", lambda j, t: gate.wait(5) or {}, heavy=False)
    _time.sleep(0.4)
    first = manager2.get(job["job_id"])
    assert first["status"] not in ("completed", "failed")
    assert first["stalled"] is False  # heartbeat is fresh
    # Backdate the heartbeat far past the threshold: stalled must flip on.
    # (The 2 s heartbeat cadence cannot catch up within this assertion.)
    with manager2._registry_lock:
        manager2._jobs[job["job_id"]]["_hb_monotonic"] = _time.monotonic() - 999
    stalled = manager2.get(job["job_id"])
    assert stalled["stalled"] is True
    gate.set()
    deadline = _time.monotonic() + 5
    while manager2.get(job["job_id"])["status"] != "completed" and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert manager2.get(job["job_id"])["stalled"] is False  # terminal, never stalled


def test_failed_job_carries_safe_diagnostic(client, engines, manager, fast_infra):
    engines["kokoro"].fail = "synthesize"
    engines["windows_sapi"].dep_ok = False
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_warm", "text": "Diagnóstico."}
    ).json()
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "failed"
    diagnostic = final["diagnostic"]
    assert diagnostic["exception_type"] == "JobError"
    assert diagnostic["last_stage"]
    assert "ram" in diagnostic and "vram" in diagnostic
    assert "Traceback" not in json.dumps(final)  # no stack traces exposed
    assert _scan_for_absolute_paths(final) == []


def test_ui_has_reconnection_and_interrupted_handling():
    """The designer page ships the hotfix: adaptive reconnection, instance
    tracking, interrupted-job messaging, and backend revalidation on load."""
    html = (VOICE_LAB_ROOT / "app" / "static" / "designer.html").read_text(encoding="utf-8")
    assert "healthTick" in html
    assert "instance_id" in html
    assert "El worker se reinició durante la operación." in html
    assert "adoptActiveJobs" in html
    assert "reconectando" in html
    assert "Math.min(10000" in html  # backoff capped at 10 s
    assert "stalled" in html


# --- Phase 3D.0.5: coordinator modes, optimize, low-memory fallback ---------------------


def test_mode_endpoint_get_and_switch(client):
    assert client.get("/mode").json()["mode"] == "daily"
    switched = client.post("/mode", json={"mode": "designer"}).json()
    assert switched["status"] == "ok"
    assert switched["previous_mode"] == "daily"
    assert switched["mode"] == "designer"
    assert client.get("/mode").json()["mode"] == "designer"
    # Any spelling normalizes: low_memory -> low-memory.
    assert client.post("/mode", json={"mode": "low_memory"}).json()["mode"] == "low-memory"


def test_low_memory_refuses_heavy_design_job(client, engines, manager, fast_infra):
    """low-memory mode never loads a heavy Qwen model — a VoiceDesign job is
    refused at preflight (structured, never queued)."""
    client.post("/mode", json={"mode": "low-memory"})
    response = client.post(
        "/jobs/voice-design", json={"profile_name": "fifi_lm", "language": "es"}
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "mode_low_memory"
    assert "baja memoria" in data["user_message"]
    assert data["retryable"] is False
    assert client.get("/jobs/active").json()["jobs"] == []  # never queued
    assert engines["qwen3_tts"].synth_calls == 0


def test_low_memory_preview_of_qwen_profile_falls_back_to_kokoro(
    client, engines, manager, voices_dir, fast_infra
):
    """A qwen profile in low-memory mode is served by Kokoro — the request is
    accepted as a LIGHT job (not refused), and the heavy engine is never used."""
    _make_qwen_profile(voices_dir, "fifi_qlm")
    client.post("/mode", json={"mode": "low-memory"})
    job = client.post(
        "/jobs/profile-preview", json={"profile": "fifi_qlm", "text": "Baja memoria."}
    ).json()
    assert job["job_id"]  # accepted, not refused
    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed"
    assert final["result"]["actual_engine"] == "kokoro"  # collapsed away from qwen
    assert engines["qwen3_tts"].synth_calls == 0  # heavy engine never touched


def test_synthesize_qwen_profile_low_memory_uses_kokoro(
    client, engines, manager, voices_dir, fast_infra
):
    _make_qwen_profile(voices_dir, "fifi_qsyn")
    client.post("/mode", json={"mode": "low-memory"})
    data = client.post("/synthesize", json={"text": "hola", "profile": "fifi_qsyn"}).json()
    assert data["status"] == "ok"
    assert data["engine"] == "kokoro"
    assert data["requested_engine"] == "qwen3_tts"
    assert data["fallback_used"] is True
    assert engines["qwen3_tts"].synth_calls == 0


def test_models_optimize_unloads_non_required_and_keeps_cache(
    client, engines, manager, monkeypatch, fast_infra
):
    from app.engines import base as base_module
    from app.engines.qwen3_tts import CLONE_MODEL_ID, DEFAULT_MODEL_ID

    class FakeQwen:
        def __init__(self):
            self._models = {DEFAULT_MODEL_ID: object(), CLONE_MODEL_ID: object()}
            self.unloaded = []

        def unload_model(self, mid):
            self._models.pop(mid, None)
            self.unloaded.append(mid)
            return True

    fake = FakeQwen()
    monkeypatch.setitem(base_module._instances, "qwen3_tts", fake)
    monkeypatch.setattr(base_module, "loaded_engines", lambda: {"kokoro": True, "qwen3_tts": True})
    monkeypatch.setattr(base_module, "unload_all", lambda only=None: [only] if only else [])
    manager.select("fifi_warm")  # active voice = Kokoro
    data = client.post("/models/optimize", json={}).json()
    assert data["status"] == "ok"
    assert data["kept_engine"] == "kokoro"
    assert data["unloaded_engines"] == []  # kokoro is the active engine — kept
    assert set(data["unloaded_models"]) == {DEFAULT_MODEL_ID, CLONE_MODEL_ID}
    assert fake._models == {}  # both heavy models freed
    assert "no se borran" in data["note"]  # cached weights untouched


def test_coordinator_status_endpoint_has_req7_fields(client, engines, manager, fast_infra):
    manager.select("fifi_warm")
    data = client.get("/coordinator/status").json()
    assert data["status"] == "ok"
    assert data["mode"] == "daily"
    assert data["active_voice"] == "fifi_warm"
    assert data["required_tts_engine"] == "kokoro"
    assert data["system_ram_available_gb"] == 8.0  # from fast_infra fakes
    assert data["dedicated_vram_free_gb"] == 8.0
    assert data["shared_gpu_memory"] is None  # never counted as VRAM
    for key in ("mode", "active_voice", "required_tts_engine", "loaded_voice_lab_model",
                "ollama_loaded", "whisper_loaded", "current_heavy_job", "fallback_status"):
        assert key in data, key
    assert _scan_for_absolute_paths(data) == []
