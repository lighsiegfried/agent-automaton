"""Voice catalog: base voices vs style variants, grouping, and availability."""

from app.engines.catalog import (
    KOKORO_SPEAKERS,
    QWEN_CUSTOM_SPEAKERS,
    base_speaker_profiles,
    build_catalog,
    designed_identity_profiles,
)
from app.profiles.schema import VoiceProfile
from conftest import PROFILE_TEMPLATE


def _profile(name, **overrides) -> VoiceProfile:
    return VoiceProfile(**{**PROFILE_TEMPLATE, "name": name, **overrides})


DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

SAMPLE_PROFILES = [
    _profile("fifi_es_warm", language="es", speaker="ef_dora", speed=1.0),
    _profile("fifi_es_calm", language="es", speaker="ef_dora", speed=0.85),  # variant
    _profile("fifi_es_alex", language="es", speaker="em_alex", speed=1.0),
    _profile("fifi_es_santa", language="es", speaker="em_santa", speed=1.0),
    _profile(
        "fifi_aurora",
        engine="qwen3_tts",
        model=DESIGN_MODEL,
        language="es",
        speaker="",
        style_instruction="Voz femenina cálida y luminosa de adulta joven.",
        fallback_engine="kokoro",
    ),
    _profile(
        "fifi_professional",
        engine="qwen3_tts",
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        language="en",
        speaker="Serena",
        style_instruction="Professional.",
    ),
]


def test_catalog_covers_all_spanish_kokoro_speakers():
    ids = {s["id"] for s in KOKORO_SPEAKERS if s["language"] == "es"}
    assert ids == {"ef_dora", "em_alex", "em_santa"}


def test_catalog_groups_variants_under_their_base_voice():
    catalog = build_catalog(SAMPLE_PROFILES)
    dora = next(e for e in catalog if e["id"] == "ef_dora")
    # Both dora profiles are listed UNDER the one base voice — a style variant
    # is never presented as a separate voice.
    assert dora["profiles"] == ["fifi_es_calm", "fifi_es_warm"]
    assert dora["kind"] == "speaker"
    assert not any(e["id"] == "fifi_es_calm" for e in catalog)


def test_catalog_entries_carry_language_gender_timbre_availability():
    catalog = build_catalog(SAMPLE_PROFILES)
    for entry in catalog:
        assert entry["engine"] in ("kokoro", "qwen3_tts", "windows_sapi")
        assert {"id", "language", "gender", "timbre", "kind", "available"} <= set(entry)


def test_designed_identities_are_their_own_catalog_entries():
    catalog = build_catalog(SAMPLE_PROFILES)
    aurora = next(e for e in catalog if e["id"] == "fifi_aurora")
    assert aurora["kind"] == "designed"
    assert aurora["engine"] == "qwen3_tts"
    assert "cálida" in aurora["timbre"]  # the instruction IS the timbre


def test_qwen_custom_speakers_catalogued():
    serena = next(e for e in build_catalog(SAMPLE_PROFILES) if e["id"] == "Serena")
    assert serena["kind"] == "speaker"
    assert serena["profiles"] == ["fifi_professional"]
    assert {s["id"] for s in QWEN_CUSTOM_SPEAKERS} >= {"Serena", "Vivian", "Ryan"}


def test_base_speaker_profiles_collapse_variants():
    base = base_speaker_profiles(SAMPLE_PROFILES, language="es")
    names = sorted(p.name for p in base)
    # One per base speaker; the speed-1.0 variant wins; designed excluded.
    assert names == ["fifi_es_alex", "fifi_es_santa", "fifi_es_warm"]


def test_designed_identity_profiles_selects_only_designs():
    identities = designed_identity_profiles(SAMPLE_PROFILES)
    assert [p.name for p in identities] == ["fifi_aurora"]
