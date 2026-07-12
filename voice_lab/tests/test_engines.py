"""Engine contract: lazy loading, structured failures, safe unload, no commands."""

from pathlib import Path

import pytest

from app.engines import base
from app.profiles.schema import VoiceProfile
from conftest import PROFILE_TEMPLATE, VOICE_LAB_ROOT, FakeEngine


def profile(**overrides) -> VoiceProfile:
    return VoiceProfile(**{**PROFILE_TEMPLATE, "name": "fifi_test", **overrides})


def test_engines_load_lazily(tmp_path):
    engine = FakeEngine("kokoro")
    assert not engine.loaded  # constructing an engine loads nothing
    result = engine.synthesize("hola", profile(), tmp_path / "out.wav")
    assert result["status"] == "ok"
    assert engine.loaded and engine.load_calls == 1
    engine.synthesize("otra", profile(), tmp_path / "out2.wav")
    assert engine.load_calls == 1  # loaded once, reused


def test_missing_dependencies_are_structured_not_raised(tmp_path):
    engine = FakeEngine("kokoro", dep_ok=False)
    result = engine.synthesize("hola", profile(), tmp_path / "out.wav")
    assert result == {
        "status": "unavailable",
        "engine": "kokoro",
        "message": "kokoro deps missing",
    }
    assert not engine.loaded


def test_load_failure_is_structured(tmp_path):
    engine = FakeEngine("qwen3_tts", fail="load")
    result = engine.synthesize("hola", profile(), tmp_path / "out.wav")
    assert result["status"] == "unavailable"
    assert "model missing" in result["message"]


def test_synthesis_crash_is_structured(tmp_path):
    engine = FakeEngine("kokoro", fail="synthesize")
    result = engine.synthesize("hola", profile(), tmp_path / "out.wav")
    assert result["status"] == "error"
    assert "RuntimeError" in result["message"]


def test_unload_reports_and_resets(tmp_path):
    engine = FakeEngine("kokoro")
    engine.synthesize("hola", profile(), tmp_path / "out.wav")
    assert engine.unload() is True  # something was loaded
    assert engine.unload() is False  # already unloaded
    assert not engine.loaded


def test_registry_returns_singletons(monkeypatch):
    monkeypatch.setattr(base, "_instances", {})
    first = base.get_engine("windows_sapi")
    assert base.get_engine("windows_sapi") is first
    with pytest.raises(base.EngineUnavailable):
        base.get_engine("bash")


def test_unload_all_only_touches_loaded_engines(monkeypatch, tmp_path):
    kokoro = FakeEngine("kokoro")
    sapi = FakeEngine("windows_sapi")
    monkeypatch.setattr(base, "_instances", {"kokoro": kokoro, "windows_sapi": sapi})
    kokoro.synthesize("hola", profile(), tmp_path / "out.wav")
    assert base.unload_all() == ["kokoro"]  # sapi was never loaded
    assert base.unload_all("kokoro") == []  # already unloaded


# --- kokoro: language / speaker validation (pure functions, no torch) ---------------


def test_kokoro_language_codes():
    from app.engines.kokoro import resolve_lang_code

    assert resolve_lang_code("es") == "e"
    assert resolve_lang_code("en") == "a"
    assert resolve_lang_code("") == "a"
    assert resolve_lang_code("klingon") == "a"  # unknown -> english, never crash


def test_kokoro_empty_speaker_gets_language_default():
    from app.engines.kokoro import validate_voice

    assert validate_voice("es", "") == ("ef_dora", "")
    assert validate_voice("en", "") == ("af_heart", "")


def test_kokoro_rejects_language_mismatched_speaker():
    from app.engines.kokoro import validate_voice

    voice, error = validate_voice("es", "af_heart")  # English voice, Spanish profile
    assert voice == ""
    assert "af_heart" in error and "ef_" in error  # clear, actionable message


def test_kokoro_accepts_language_consistent_speakers():
    from app.engines.kokoro import validate_voice

    for language, speaker in (("es", "ef_dora"), ("es", "em_alex"), ("en", "am_adam")):
        assert validate_voice(language, speaker) == (speaker, "")


# --- qwen3_tts stays optional with safe OOM handling ---------------------------------


def test_qwen_is_optional_and_not_in_default_requirements():
    requirements = (VOICE_LAB_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        line = line.split("#")[0].strip()
        assert not line.startswith(("qwen-tts", "transformers", "accelerate"))
    optional = (VOICE_LAB_ROOT / "requirements-qwen.txt").read_text(encoding="utf-8")
    assert "qwen-tts" in optional

    from app.engines.qwen3_tts import INSTALL_HINT

    assert "--with-qwen" in INSTALL_HINT  # opt-in is spelled out


def test_qwen_language_mapping():
    from app.engines.qwen3_tts import map_language

    assert map_language("es") == "Spanish"
    assert map_language("en") == "English"
    assert map_language("") == "English"
    assert map_language("klingon") == "English"  # unknown -> supported default


def test_qwen_model_variant_detection():
    from app.engines.qwen3_tts import DEFAULT_MODEL_ID, is_voice_design

    assert is_voice_design("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    assert is_voice_design("")  # empty model falls back to the default...
    assert is_voice_design(DEFAULT_MODEL_ID)  # ...which is the VoiceDesign model
    assert not is_voice_design("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def test_voice_design_requires_an_instruction():
    """A VoiceDesign profile without a style_instruction is a clear error —
    the instruction IS the voice."""
    source = (VOICE_LAB_ROOT / "app" / "engines" / "qwen3_tts.py").read_text(encoding="utf-8")
    assert "no style_instruction" in source
    assert "generate_voice_design" in source
    assert "generate_custom_voice" in source


def test_qwen_handles_gpu_oom_explicitly():
    source = (VOICE_LAB_ROOT / "app" / "engines" / "qwen3_tts.py").read_text(encoding="utf-8")
    assert "OutOfMemoryError" in source  # caught, not propagated
    assert "empty_cache" in source  # VRAM released
    assert "EngineUnavailable(OOM_HINT)" in source  # structured -> fallback


# --- importing engine modules must stay cheap and safe ------------------------------


def test_engine_modules_import_without_neural_dependencies():
    """Top-level imports must not require torch/kokoro/transformers — this test
    runs in the MAIN environment where none of them are installed."""
    import app.engines.kokoro  # noqa: F401
    import app.engines.qwen3_tts  # noqa: F401
    import app.engines.windows_sapi  # noqa: F401


def test_engines_never_execute_commands():
    """Engines synthesize audio and nothing else — no shell, no subprocess."""
    engines_dir = VOICE_LAB_ROOT / "app" / "engines"
    for path in engines_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("subprocess", "os.system", "os.exec", "shell=True", "Popen"):
            assert forbidden not in source, f"{path.name} contains {forbidden!r}"


def test_engine_availability_in_isolated_main_env():
    """In the main venv the neural engines report unavailable with an install
    hint (their deps live only in voice_lab/.venv) — they never crash."""
    from app.engines.kokoro import KokoroEngine
    from app.engines.qwen3_tts import Qwen3TtsEngine

    for engine_cls in (KokoroEngine, Qwen3TtsEngine):
        ok, reason = engine_cls().available()
        if not ok:  # true in CI/main env; a fully-installed lab may report ok
            assert "voice_lab/scripts/setup.py" in reason
            assert ":\\" not in reason  # hint contains no absolute paths
