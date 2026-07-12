"""Profile schema + manager: validation, atomic activation, shared files."""

import json
import os

import pytest

from app.profiles.manager import ProfileError, ProfileManager
from app.profiles.schema import VoiceProfile
from conftest import PROFILE_TEMPLATE, PROJECT_ROOT


# --- schema -------------------------------------------------------------------------


def test_schema_accepts_valid_profile():
    profile = VoiceProfile(**{**PROFILE_TEMPLATE, "name": "fifi_test"})
    assert profile.engine == "kokoro"
    assert profile.fallback_engine == "windows_sapi"


def test_schema_rejects_unknown_engine():
    with pytest.raises(ValueError):
        VoiceProfile(**{**PROFILE_TEMPLATE, "name": "x", "engine": "shell"})


def test_schema_rejects_filesystem_paths():
    """Shared profile files can never smuggle paths into responses."""
    for field, value in (
        ("model", "C:\\secret\\model.bin"),
        ("speaker", "/etc/passwd"),
        ("style_instruction", "\\\\server\\share"),
    ):
        with pytest.raises(ValueError):
            VoiceProfile(**{**PROFILE_TEMPLATE, "name": "x", field: value})


def test_schema_rejects_bad_names_and_speeds():
    with pytest.raises(ValueError):
        VoiceProfile(**{**PROFILE_TEMPLATE, "name": "Fifi Warm!"})  # spaces/caps
    with pytest.raises(ValueError):
        VoiceProfile(**{**PROFILE_TEMPLATE, "name": "x", "speed": 9.0})


# --- manager ------------------------------------------------------------------------


def test_list_and_get(manager):
    names = [p.name for p in manager.list_profiles()]
    assert names == ["fifi_calm", "fifi_warm"]  # sorted, both valid
    assert manager.get_profile("fifi_warm").speaker == "af_heart"


def test_get_unknown_profile_is_a_clean_error(manager):
    with pytest.raises(ProfileError) as excinfo:
        manager.get_profile("does_not_exist")
    assert "Available:" in str(excinfo.value)


def test_broken_profile_file_is_skipped(manager, voices_dir):
    (voices_dir / "profiles" / "broken.json").write_text("{not json", encoding="utf-8")
    names = [p.name for p in manager.list_profiles()]
    assert "broken" not in names
    assert names == ["fifi_calm", "fifi_warm"]  # still served


def test_select_writes_active_json_atomically(manager, voices_dir, monkeypatch):
    replaced = {}
    real_replace = os.replace

    def tracking_replace(src, dst):
        replaced["src"] = str(src)
        replaced["dst"] = str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)
    payload = manager.select("fifi_calm")

    # Written via temp-file + os.replace (atomic), in the SAME directory.
    assert replaced["dst"] == str(voices_dir / "active.json")
    assert os.path.dirname(replaced["src"]) == str(voices_dir)
    # No temp files left behind.
    assert list(voices_dir.glob("*.tmp")) == []

    on_disk = json.loads((voices_dir / "active.json").read_text(encoding="utf-8"))
    assert on_disk == payload
    assert on_disk["profile"] == "fifi_calm"
    assert on_disk["engine"] == "kokoro"
    assert on_disk["fallback_engine"] == "windows_sapi"
    assert manager.active_profile_name() == "fifi_calm"
    # Shared file stays path-free.
    assert ":\\" not in json.dumps(on_disk)


def test_select_unknown_profile_changes_nothing(manager, voices_dir):
    manager.select("fifi_warm")
    before = (voices_dir / "active.json").read_text(encoding="utf-8")
    with pytest.raises(ProfileError):
        manager.select("nope")
    assert (voices_dir / "active.json").read_text(encoding="utf-8") == before


def test_active_is_none_when_missing_or_corrupt(manager, voices_dir):
    assert manager.active() is None  # never written yet
    (voices_dir / "active.json").write_text("{broken", encoding="utf-8")
    assert manager.active() is None
    assert manager.active_profile_name() is None


# --- the real shared profiles under config/voices/ ----------------------------------


def _repo_manager() -> ProfileManager:
    return ProfileManager(
        profiles_dir=PROJECT_ROOT / "config" / "voices" / "profiles",
        active_path=PROJECT_ROOT / "config" / "voices" / "active.json",
    )


def test_repo_ships_the_fifi_profiles():
    profiles = {p.name: p for p in _repo_manager().list_profiles()}
    # Built-ins must always be present; the Voice Designer may add MORE
    # (user-designed identities like fifi_luna) — never assert an exact set.
    assert set(profiles) >= {
        "fifi_warm", "fifi_calm", "fifi_professional", "fifi_friendly",
        "fifi_es_warm", "fifi_es_calm", "fifi_es_vivaz", "fifi_es_alex",
        "fifi_es_santa",
        "fifi_aurora", "fifi_serena", "fifi_nova", "fifi_ejecutiva",
    }
    for profile in profiles.values():
        # Neural voices always degrade to another engine, ending in SAPI.
        assert profile.fallback_engine in ("windows_sapi", "kokoro")
        assert profile.provenance  # provenance is required content
    assert profiles["fifi_professional"].engine == "qwen3_tts"
    assert profiles["fifi_professional"].speaker == "Serena"  # real CustomVoice timbre
    # The committed active.json points at an existing profile.
    assert _repo_manager().active_profile_name() in profiles


def test_all_three_kokoro_spanish_speakers_are_registered():
    """Phase 3D.0.2: ef_dora, em_alex, em_santa each have a base profile."""
    spanish_kokoro = {
        p.speaker for p in _repo_manager().list_profiles()
        if p.engine == "kokoro" and p.language == "es"
    }
    assert spanish_kokoro == {"ef_dora", "em_alex", "em_santa"}


def test_designed_identities_are_genuinely_distinct():
    """The four identities differ by VoiceDesign instruction — real timbre
    design, not speed variants of one speaker."""
    profiles = {p.name: p for p in _repo_manager().list_profiles()}
    identities = ["fifi_aurora", "fifi_serena", "fifi_nova", "fifi_ejecutiva"]
    instructions = set()
    for name in identities:
        profile = profiles[name]
        assert profile.engine == "qwen3_tts"
        assert "VoiceDesign" in profile.model
        assert profile.language == "es"
        assert len(profile.style_instruction) > 40  # a real design, not a tag
        assert profile.speed == 1.0  # identity comes from the design text...
        assert profile.fallback_engine == "kokoro"  # ...and degrades to kokoro
        instructions.add(profile.style_instruction)
    assert len(instructions) == 4  # all four designs are different


def test_spanish_profiles_vary_speed_and_tone():
    """Phase 3D.0.1: at least three usable Spanish Kokoro profiles with
    different speed/tone settings and language-consistent speakers."""
    manager = ProfileManager(
        profiles_dir=PROJECT_ROOT / "config" / "voices" / "profiles",
        active_path=PROJECT_ROOT / "config" / "voices" / "active.json",
    )
    spanish = [
        p for p in manager.list_profiles()
        if p.language == "es" and p.engine == "kokoro"
    ]
    assert len(spanish) >= 3
    assert len({(p.speaker, p.speed) for p in spanish}) == len(spanish)  # all distinct
    assert len({p.speed for p in spanish}) >= 3  # real speed variety
    assert len({p.speaker for p in spanish}) >= 2  # tone variety (different voices)
    from app.engines.kokoro import validate_voice

    for profile in spanish:
        voice, error = validate_voice(profile.language, profile.speaker)
        assert error == "" and voice == profile.speaker  # speaker matches language
