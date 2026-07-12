"""Voice catalog — every speaker/timbre the Voice Lab knows about.

A SPEAKER (base voice) is a distinct timbre: a Kokoro voice id, a Qwen
CustomVoice named timbre, or a Qwen VoiceDesign identity (a voice defined by a
natural-language instruction). A PROFILE that reuses a speaker with different
speed/style settings is a STYLE VARIANT of that base voice — the catalog and
the `speakers` CLI never present variants as separate voices.

Static data only — importing this module loads nothing heavy.
"""

from typing import Any

from app.engines.base import get_engine
from app.profiles.schema import VoiceProfile

# --- Kokoro base voices (curated; ids are prefixed language+gender) ------------------

KOKORO_SPEAKERS = [
    {"id": "ef_dora", "language": "es", "gender": "female", "timbre": "warm mid-range, natural"},
    {"id": "em_alex", "language": "es", "gender": "male", "timbre": "clear mid-range, direct"},
    {"id": "em_santa", "language": "es", "gender": "male", "timbre": "deep, relaxed, unhurried"},
    {"id": "af_heart", "language": "en", "gender": "female", "timbre": "warm, rounded (default)"},
    {"id": "af_bella", "language": "en", "gender": "female", "timbre": "bright, upbeat"},
    {"id": "af_nicole", "language": "en", "gender": "female", "timbre": "soft, breathy"},
    {"id": "af_sarah", "language": "en", "gender": "female", "timbre": "even, composed"},
    {"id": "am_adam", "language": "en", "gender": "male", "timbre": "solid mid-range"},
    {"id": "am_michael", "language": "en", "gender": "male", "timbre": "low, calm"},
    {"id": "bf_emma", "language": "en-gb", "gender": "female", "timbre": "clear British"},
]

# --- Qwen3-TTS CustomVoice premium timbres (from the official model card) ------------

QWEN_CUSTOM_SPEAKERS = [
    {"id": "Vivian", "language": "zh", "gender": "female", "timbre": "bright, slightly edgy, young"},
    {"id": "Serena", "language": "zh", "gender": "female", "timbre": "warm, gentle, young"},
    {"id": "Uncle_Fu", "language": "zh", "gender": "male", "timbre": "seasoned, low, mellow"},
    {"id": "Dylan", "language": "zh", "gender": "male", "timbre": "youthful Beijing, clear"},
    {"id": "Eric", "language": "zh", "gender": "male", "timbre": "lively Chengdu, husky brightness"},
    {"id": "Ryan", "language": "en", "gender": "male", "timbre": "dynamic, strong rhythm"},
    {"id": "Aiden", "language": "en", "gender": "male", "timbre": "sunny American, clear midrange"},
    {"id": "Ono_Anna", "language": "ja", "gender": "female", "timbre": "playful, light, nimble"},
    {"id": "Sohee", "language": "ko", "gender": "female", "timbre": "warm, rich emotion"},
]


def _profiles_by_speaker(profiles: list[VoiceProfile]) -> dict[tuple[str, str], list[VoiceProfile]]:
    grouped: dict[tuple[str, str], list[VoiceProfile]] = {}
    for profile in profiles:
        grouped.setdefault((profile.engine, profile.speaker), []).append(profile)
    return grouped


def build_catalog(
    profiles: list[VoiceProfile], availability: dict[str, bool] | None = None
) -> list[dict[str, Any]]:
    """The full speaker catalog with availability and the profiles per voice.

    Returns one entry per BASE VOICE:
      {engine, id, language, gender, timbre, kind, available, profiles: [...]}
    kind: "speaker" (fixed timbre) | "designed" (VoiceDesign identity).

    `availability` lets callers inject the WORKER's engine availability (the
    CLI runs in the main venv, where neural engines are never installed).
    """
    grouped = _profiles_by_speaker(profiles)
    catalog: list[dict[str, Any]] = []

    if availability is None:
        availability = {
            name: get_engine(name).available()[0]
            for name in ("kokoro", "qwen3_tts", "windows_sapi")
        }
    else:
        availability = {
            name: bool(availability.get(name))
            for name in ("kokoro", "qwen3_tts", "windows_sapi")
        }

    for speaker in KOKORO_SPEAKERS:
        users = grouped.get(("kokoro", speaker["id"]), [])
        catalog.append(
            {
                "engine": "kokoro",
                "kind": "speaker",
                "available": availability["kokoro"],
                **speaker,
                "profiles": [p.name for p in sorted(users, key=lambda p: p.name)],
            }
        )

    for speaker in QWEN_CUSTOM_SPEAKERS:
        users = grouped.get(("qwen3_tts", speaker["id"]), [])
        catalog.append(
            {
                "engine": "qwen3_tts",
                "kind": "speaker",
                "available": availability["qwen3_tts"],
                **speaker,
                "profiles": [p.name for p in sorted(users, key=lambda p: p.name)],
            }
        )

    # VoiceDesign identities: each designed profile IS its own base voice —
    # the style_instruction defines the timbre, not a fixed speaker id.
    from app.engines.qwen3_tts import is_voice_design

    for profile in sorted(profiles, key=lambda p: p.name):
        if profile.engine == "qwen3_tts" and is_voice_design(profile.model) and profile.style_instruction:
            catalog.append(
                {
                    "engine": "qwen3_tts",
                    "kind": "designed",
                    "available": availability["qwen3_tts"],
                    "id": profile.name,
                    "language": profile.language,
                    "gender": "designed",
                    "timbre": profile.style_instruction,
                    "profiles": [profile.name],
                }
            )

    catalog.append(
        {
            "engine": "windows_sapi",
            "kind": "speaker",
            "available": availability["windows_sapi"],
            "id": "(system voices)",
            "language": "any",
            "gender": "system",
            "timbre": "whatever Windows SAPI voices are installed",
            "profiles": [],
        }
    )
    return catalog


def base_speaker_profiles(profiles: list[VoiceProfile], language: str | None = None) -> list[VoiceProfile]:
    """One canonical profile per fixed base speaker (for compare-speakers).

    Style variants collapse to the variant closest to speed 1.0; designed
    identities are excluded (compare-identities covers those).
    """
    from app.engines.qwen3_tts import is_voice_design

    grouped: dict[tuple[str, str], list[VoiceProfile]] = {}
    for profile in profiles:
        if language and profile.language != language:
            continue
        if not profile.speaker:
            continue
        if profile.engine == "qwen3_tts" and is_voice_design(profile.model):
            continue
        grouped.setdefault((profile.engine, profile.speaker), []).append(profile)
    return [
        min(candidates, key=lambda p: abs(p.speed - 1.0))
        for candidates in grouped.values()
    ]


def designed_identity_profiles(profiles: list[VoiceProfile]) -> list[VoiceProfile]:
    """All VoiceDesign identity profiles (for compare-identities)."""
    from app.engines.qwen3_tts import is_voice_design

    return sorted(
        (
            p for p in profiles
            if p.engine == "qwen3_tts" and is_voice_design(p.model) and p.style_instruction
        ),
        key=lambda p: p.name,
    )
