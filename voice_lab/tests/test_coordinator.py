"""Model coordinator tests (Phase 3D.0.5) — pure policy, no GPU or downloads.

The daily-use policy is exercised entirely with a fake Qwen engine and mocked
resource snapshots: mode gates, the effective fallback chain, duplicate-load
prevention, sequential Qwen switching (LRU eviction), idle unloading, and the
enforce-policy / optimize path. Cached weights are never touched here because
the fake unload only drops dict entries.
"""

from types import SimpleNamespace

import pytest

from app import coordinator as coord_mod
from app.coordinator import ModelCoordinator
from app.engines import base as base_module
from app.engines.qwen3_tts import CLONE_MODEL_ID, DEFAULT_MODEL_ID
from app.profiles.schema import VoiceProfile


# --- fakes / helpers -----------------------------------------------------------------


class FakeQwen:
    """A stand-in for the Qwen engine: _models is the resident set; unload_model
    drops one and records it (no torch, no gc, no CUDA)."""

    def __init__(self, models=()):
        self._models = {m: object() for m in models}
        self.unloaded = []

    def unload_model(self, model_id):
        if model_id in self._models:
            del self._models[model_id]
            self.unloaded.append(model_id)
            return True
        return False


def profile(engine="kokoro", model="", fallback="windows_sapi", name="p"):
    speaker = "af_heart" if engine == "kokoro" else ""
    style = "voz cálida" if model == DEFAULT_MODEL_ID else ""
    return VoiceProfile(
        name=name, engine=engine, model=model, speaker=speaker,
        style_instruction=style, fallback_engine=fallback,
    )


@pytest.fixture
def coord(tmp_path):
    return ModelCoordinator(state_dir=tmp_path / "coord")


@pytest.fixture
def install_qwen(monkeypatch):
    def _install(models=()):
        fake = FakeQwen(models)
        monkeypatch.setitem(base_module._instances, "qwen3_tts", fake)
        return fake
    return _install


def _settings(monkeypatch, **overrides):
    base = dict(
        min_free_system_ram_gb=2.0, min_free_vram_gb=2.0, idle_unload_seconds=300.0,
        max_loaded_heavy_models=1, allow_kokoro_fallback=True,
        designer_auto_unload_ollama=False, mode="daily",
    )
    base.update(overrides)
    monkeypatch.setattr(coord_mod, "get_settings", lambda: SimpleNamespace(**base))
    return base


# --- mode parsing + persistence ------------------------------------------------------


def test_mode_defaults_to_daily(coord):
    assert coord.mode == "daily"


def test_set_mode_normalizes_and_persists(tmp_path):
    c = ModelCoordinator(state_dir=tmp_path / "s")
    assert c.set_mode("LOW_MEMORY") == "low-memory"
    assert c.mode == "low-memory"
    # A fresh coordinator on the SAME state dir reads the persisted mode.
    assert ModelCoordinator(state_dir=tmp_path / "s").mode == "low-memory"


def test_set_mode_unknown_falls_back_to_daily(coord):
    assert coord.set_mode("banana") == "daily"


# --- effective fallback chain / required engine --------------------------------------


def test_daily_chain_selected_then_kokoro_then_windows(coord):
    chain = coord.effective_fallback_chain(profile(engine="qwen3_tts", model=CLONE_MODEL_ID))
    assert chain == ["qwen3_tts", "kokoro", "windows_sapi"]
    assert coord.required_engine(profile(engine="qwen3_tts", model=CLONE_MODEL_ID)) == "qwen3_tts"


def test_kokoro_profile_requires_only_kokoro(coord):
    p = profile(engine="kokoro")
    assert coord.effective_fallback_chain(p) == ["kokoro", "windows_sapi"]
    assert coord.required_engine(p) == "kokoro"


def test_low_memory_drops_qwen_from_chain(coord):
    coord.set_mode("low-memory")
    p = profile(engine="qwen3_tts", model=CLONE_MODEL_ID)
    assert coord.effective_fallback_chain(p) == ["kokoro", "windows_sapi"]
    assert coord.required_engine(p) == "kokoro"
    assert coord.profile_is_heavy(p) is False  # a qwen profile is LIGHT here


def test_allow_kokoro_fallback_false_drops_kokoro(coord, monkeypatch):
    _settings(monkeypatch, allow_kokoro_fallback=False)
    p = profile(engine="qwen3_tts", model=CLONE_MODEL_ID)
    assert coord.effective_fallback_chain(p) == ["qwen3_tts", "windows_sapi"]
    # low-memory + no kokoro fallback => Windows host TTS only.
    coord.set_mode("low-memory")
    assert coord.effective_fallback_chain(p) == ["windows_sapi"]


def test_profile_is_heavy_daily(coord):
    assert coord.profile_is_heavy(profile(engine="qwen3_tts", model=CLONE_MODEL_ID)) is True
    assert coord.profile_is_heavy(profile(engine="kokoro")) is False


# --- mode gates ----------------------------------------------------------------------


def test_mode_allows_heavy(coord):
    assert coord.mode_allows_heavy("voice_design")[0] is True
    coord.set_mode("low-memory")
    allowed, reason = coord.mode_allows_heavy("voice_design")
    assert allowed is False
    assert "baja memoria" in reason


def test_designer_ollama_consent_requires_designer_mode_and_flag(coord, monkeypatch):
    _settings(monkeypatch, designer_auto_unload_ollama=True)
    assert coord.designer_ollama_consent("voice_design") is False  # daily
    coord.set_mode("designer")
    assert coord.designer_ollama_consent("voice_design") is True
    assert coord.designer_ollama_consent("clone") is False  # only VoiceDesign
    _settings(monkeypatch, designer_auto_unload_ollama=False)
    assert coord.designer_ollama_consent("voice_design") is False  # flag off


# --- duplicate-load prevention + sequential switching --------------------------------


def test_resident_model_is_reused_not_reloaded(coord, install_qwen):
    fake = install_qwen([DEFAULT_MODEL_ID])
    # Target already resident: nothing is evicted, nothing reloaded.
    result = coord.prepare_for_heavy_load(DEFAULT_MODEL_ID)
    assert result["unloaded"] == []
    assert fake.unloaded == []
    assert DEFAULT_MODEL_ID in fake._models


def test_sequential_switch_evicts_previous_unused_model(coord, install_qwen):
    """max_loaded_heavy_models=1: loading the clone first frees VoiceDesign
    (release -> gc -> CUDA clear inside the real unload_model)."""
    fake = install_qwen([DEFAULT_MODEL_ID])
    result = coord.prepare_for_heavy_load(CLONE_MODEL_ID)
    assert result["unloaded"] == [DEFAULT_MODEL_ID]
    assert fake.unloaded == [DEFAULT_MODEL_ID]
    assert DEFAULT_MODEL_ID not in fake._models


def test_lru_eviction_keeps_recently_used(coord, install_qwen, monkeypatch):
    """With room for 2 heavy models, loading a 3rd evicts the LEAST recently
    used, not the most recent."""
    _settings(monkeypatch, max_loaded_heavy_models=2)
    fake = install_qwen([DEFAULT_MODEL_ID, CLONE_MODEL_ID])
    # Deterministic last-used stamps (Windows monotonic() resolution is too
    # coarse to distinguish rapid successive calls): clone older, VoiceDesign new.
    coord._model_last_used = {CLONE_MODEL_ID: 100.0, DEFAULT_MODEL_ID: 200.0}
    # Loading a third heavy model must keep 1 other (limit=1) — the LRU (clone) goes.
    result = coord.prepare_for_heavy_load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    assert result["unloaded"] == [CLONE_MODEL_ID]
    assert DEFAULT_MODEL_ID in fake._models  # recently used survives


def test_prepare_no_qwen_instance_is_noop(coord):
    assert coord.prepare_for_heavy_load(CLONE_MODEL_ID) == {"unloaded": [], "kept": []}


# --- idle unloading ------------------------------------------------------------------


def test_idle_tick_below_threshold_does_nothing(coord, install_qwen):
    fake = install_qwen([DEFAULT_MODEL_ID])
    coord.note_activity()  # last activity = now
    assert coord.idle_unload_tick(now=coord._last_activity + 10, active_jobs=0, idle_seconds=300) == []
    assert DEFAULT_MODEL_ID in fake._models


def test_idle_tick_unloads_heavy_after_threshold(coord, install_qwen):
    fake = install_qwen([DEFAULT_MODEL_ID, CLONE_MODEL_ID])
    base_t = coord._last_activity
    freed = coord.idle_unload_tick(now=base_t + 400, active_jobs=0, idle_seconds=300)
    assert set(freed) == {DEFAULT_MODEL_ID, CLONE_MODEL_ID}
    assert fake._models == {}


def test_idle_tick_never_unloads_while_a_job_runs(coord, install_qwen):
    fake = install_qwen([DEFAULT_MODEL_ID])
    freed = coord.idle_unload_tick(now=coord._last_activity + 9999, active_jobs=1, idle_seconds=300)
    assert freed == []
    assert DEFAULT_MODEL_ID in fake._models  # a running job is not idleness


def test_idle_disabled_when_threshold_zero(coord, install_qwen):
    fake = install_qwen([DEFAULT_MODEL_ID])
    assert coord.idle_unload_tick(now=coord._last_activity + 99999, active_jobs=0, idle_seconds=0) == []
    assert DEFAULT_MODEL_ID in fake._models


# --- enforce policy (model-optimize) -------------------------------------------------


def test_enforce_policy_keeps_active_clone_unloads_rest(coord, install_qwen, monkeypatch):
    fake = install_qwen([DEFAULT_MODEL_ID, CLONE_MODEL_ID])
    loaded = {"kokoro": True, "qwen3_tts": True}
    monkeypatch.setattr(base_module, "loaded_engines", lambda: loaded)
    monkeypatch.setattr(base_module, "unload_all", lambda only=None: [only] if loaded.get(only) else [])
    # Active voice = a frozen Qwen clone -> keep the clone model, drop VoiceDesign + Kokoro.
    active = profile(engine="qwen3_tts", model=CLONE_MODEL_ID)
    result = coord.enforce_policy(active, "test")
    assert result["kept_engine"] == "qwen3_tts"
    assert result["kept_model"] == CLONE_MODEL_ID
    assert result["unloaded_models"] == [DEFAULT_MODEL_ID]
    assert result["unloaded_engines"] == ["kokoro"]
    assert list(fake._models) == [CLONE_MODEL_ID]


def test_enforce_policy_kokoro_active_unloads_all_qwen(coord, install_qwen, monkeypatch):
    fake = install_qwen([DEFAULT_MODEL_ID, CLONE_MODEL_ID])
    monkeypatch.setattr(base_module, "loaded_engines", lambda: {"kokoro": True})
    monkeypatch.setattr(base_module, "unload_all", lambda only=None: [])
    active = profile(engine="kokoro")
    result = coord.enforce_policy(active, "test")
    assert result["kept_engine"] == "kokoro"
    assert set(result["unloaded_models"]) == {DEFAULT_MODEL_ID, CLONE_MODEL_ID}
    assert fake._models == {}  # both heavy models freed; kokoro kept (active)


# --- status shape --------------------------------------------------------------------


def test_status_has_req7_fields(coord, install_qwen, monkeypatch):
    install_qwen([CLONE_MODEL_ID])
    from app import resources as resources_module

    monkeypatch.setattr(resources_module, "snapshot", lambda: {
        "system_ram_available_gb": 6.0, "dedicated_vram_free_gb": 7.5,
        "gpu_name": "RTX 5060 Ti", "vram_source": "nvidia-smi",
        "shared_gpu_memory": None,
        "engines_loaded": {"kokoro": False, "qwen3_tts": True},
        "ollama": {"loaded": [{"name": "qwen2.5:7b", "vram_gb": 4.4}]},
        "whisper": {"reachable": True, "loaded": True},
    })
    jobs = SimpleNamespace(active=lambda: [], active_heavy=lambda: None)
    status = coord.status(profile(engine="qwen3_tts", model=CLONE_MODEL_ID), jobs=jobs)
    for key in (
        "mode", "active_voice", "required_tts_engine", "loaded_voice_lab_model",
        "ollama_loaded", "whisper_loaded", "system_ram_available_gb",
        "dedicated_vram_free_gb", "current_heavy_job", "fallback_status",
    ):
        assert key in status, key
    assert status["mode"] == "daily"
    assert status["required_tts_engine"] == "qwen3_tts"
    assert status["loaded_voice_lab_model"] == CLONE_MODEL_ID
    assert status["ollama_loaded"] == ["qwen2.5:7b"]
    assert status["whisper_loaded"] is True
    assert status["shared_gpu_memory"] is None  # never counted as VRAM
    assert status["dedicated_vram_free_gb"] == 7.5


def test_status_reports_low_memory_fallback(coord, monkeypatch):
    from app import resources as resources_module

    monkeypatch.setattr(resources_module, "snapshot", lambda: {
        "engines_loaded": {}, "ollama": {"loaded": []}, "whisper": {"loaded": False},
    })
    coord.set_mode("low-memory")
    status = coord.status(profile(engine="qwen3_tts", model=CLONE_MODEL_ID))
    assert status["required_tts_engine"] == "kokoro"  # collapsed away from qwen
    assert "fallback activo" in status["fallback_status"]
