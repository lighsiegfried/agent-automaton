"""Voice Designer: variants, freeze/clone workflow, cleanup, portability, safety.

Everything is mocked — synthesis is an injected fake, no model/GPU/audio."""

import json

import pytest

from app.designer import DesignerError, VoiceDesigner, compose_instruction
from app.profiles.designs import clone_authorized, load_design
from app.profiles.schema import VoiceProfile
from conftest import write_wav


@pytest.fixture
def designer(tmp_path):
    calls = []

    def fake_synth(text, profile, out_path):
        calls.append({"text": text, "profile": profile, "path": out_path})
        write_wav(out_path)
        return {"status": "ok", "engine": "qwen3_tts", "seconds": 0.5, "duration_seconds": 4.0}

    instance = VoiceDesigner(
        root=tmp_path,
        identities_dir=tmp_path / "identities",
        profiles_dir=tmp_path / "profiles",
        designs_dir=tmp_path / "designs",
        synth=fake_synth,
    )
    instance.calls = calls
    return instance


def _request(**overrides):
    base = {
        "profile_name": "fifi_luna",
        "mode": "voice_design_then_clone",
        "language": "es",
        "age": "adulta joven",
        "gender": "femenina",
        "timbre": "aterciopelado",
        "pitch": "medio-grave",
        "energy": "tranquila",
        "speed": "pausada",
        "accent": "neutro",
        "emotion": "serena",
        "personality": "soñadora",
        "instruction": "como una astrónoma contando historias del cielo",
        "preview_text": "Hola, soy Fifi y esta es mi nueva voz.",
        "count": 3,
    }
    base.update(overrides)
    return base


# --- instruction composition ---------------------------------------------------------


def test_compose_instruction_merges_fields_and_freeform():
    instruction = compose_instruction(_request())
    assert instruction.startswith("Voz femenina de adulta joven")
    for fragment in ("aterciopelado", "medio-grave", "tranquila", "pausada",
                     "neutro", "serena", "soñadora", "astrónoma"):
        assert fragment in instruction


# --- variant generation ---------------------------------------------------------------


def test_generate_creates_three_or_four_variants(designer, tmp_path):
    manifest = designer.generate(_request(count=3))
    assert len(manifest["variants"]) == 3
    for variant in manifest["variants"]:
        assert (tmp_path / variant["file"]).is_file()  # relative to the root
        assert variant["text"] == "Hola, soy Fifi y esta es mi nueva voz."
    manifest4 = designer.generate(_request(profile_name="fifi_otra", count=4))
    assert len(manifest4["variants"]) == 4
    with pytest.raises(DesignerError):
        designer.generate(_request(profile_name="fifi_x", count=7))


def test_generate_validates_name_against_traversal(designer):
    for evil in ("../evil", "a/b", "UPPER", "name with spaces", "", "x" * 65):
        with pytest.raises(DesignerError):
            designer.generate(_request(profile_name=evil))


def test_generate_refuses_existing_profile_name(designer):
    designer.profiles_dir.mkdir(parents=True, exist_ok=True)
    (designer.profiles_dir / "fifi_luna.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DesignerError, match="already exists"):
        designer.generate(_request())


def test_manifest_and_responses_are_path_portable(designer, tmp_path):
    manifest = designer.generate(_request())
    text = json.dumps(manifest, ensure_ascii=False)
    assert str(tmp_path) not in text  # no absolute paths anywhere
    assert ":\\" not in text and not any(
        v["file"].startswith(("/", "\\")) for v in manifest["variants"]
    )


def test_regenerate_replaces_one_variant(designer, tmp_path):
    manifest = designer.generate(_request())
    target = manifest["variants"][1]
    designer.rename_variant("fifi_luna", target["id"], "mi favorita")
    updated = designer.regenerate("fifi_luna", target["id"])
    fresh = updated["variants"][1]
    assert fresh["id"] != target["id"]  # re-rolled
    assert fresh["label"] == "mi favorita"  # label survives the re-roll
    assert not (tmp_path / target["file"]).exists()  # old audio removed
    assert (tmp_path / fresh["file"]).is_file()
    assert [v["id"] for v in updated["variants"]][::2] == [
        manifest["variants"][0]["id"], manifest["variants"][2]["id"],
    ]  # the others are untouched


def test_delete_variant_removes_files(designer, tmp_path):
    manifest = designer.generate(_request())
    for variant in list(manifest["variants"]):
        manifest = designer.delete_variant("fifi_luna", variant["id"])
        assert not (tmp_path / variant["file"]).exists()
    # deleting the last variant cleans the whole working directory
    assert not (designer.identities_dir / "fifi_luna").exists()


def test_unknown_variant_is_clean(designer):
    designer.generate(_request())
    with pytest.raises(DesignerError, match="unknown variant"):
        designer.regenerate("fifi_luna", "nope")


# --- freeze: voice design then clone ---------------------------------------------------


def test_freeze_builds_clone_profile_and_design(designer, tmp_path):
    manifest = designer.generate(_request())
    chosen = manifest["variants"][2]
    result = designer.freeze("fifi_luna", chosen["id"])

    profile = VoiceProfile(**json.loads(
        (designer.profiles_dir / "fifi_luna.json").read_text(encoding="utf-8")
    ))
    assert profile.engine == "qwen3_tts"
    assert profile.model.endswith("-Base")  # the clone checkpoint
    assert profile.fallback_engine == "kokoro"
    assert profile.provenance["mode"] == "voice_design_then_clone"

    design = load_design("fifi_luna", designer.designs_dir)
    assert design["reference"]["text"] == chosen["text"]  # EXACT transcript kept
    assert design["reference"]["audio"] == "identities/fifi_luna/reference.wav"
    assert (tmp_path / design["reference"]["audio"]).is_file()
    assert design["chosen_variant"]["id"] == chosen["id"]
    ok, _reason = clone_authorized(design)
    assert ok  # synthetic reference: self-authorized

    # Audition variants are temporary: only the reference survives the freeze.
    for variant in manifest["variants"]:
        assert not (tmp_path / variant["file"]).exists()
    assert str(tmp_path) not in json.dumps(result, ensure_ascii=False)


def test_freeze_voice_design_mode_keeps_design_model(designer):
    manifest = designer.generate(_request(mode="voice_design"))
    designer.freeze("fifi_luna", manifest["variants"][0]["id"])
    profile = json.loads(
        (designer.profiles_dir / "fifi_luna.json").read_text(encoding="utf-8")
    )
    assert "VoiceDesign" in profile["model"]
    assert profile["style_instruction"]  # the instruction IS the voice


# --- reference clone: authorization gate ------------------------------------------------


def test_reference_clone_requires_authorization(designer):
    with pytest.raises(DesignerError, match="authorization"):
        designer.freeze_reference_clone(
            "voz_ajena", "es", b"RIFFdata", "hola mundo", authorization_statement=""
        )
    assert not (designer.profiles_dir / "voz_ajena.json").exists()  # nothing written


def test_reference_clone_with_authorization(designer, tmp_path):
    result = designer.freeze_reference_clone(
        "voz_propia", "es", b"RIFFdata", "hola mundo",
        authorization_statement="Grabación de mi propia voz, autorizada por mí (Wilson) el 2026-07-11.",
    )
    design = load_design("voz_propia", designer.designs_dir)
    ok, _ = clone_authorized(design)
    assert ok
    assert design["reference"]["authorization"]["type"] == "external"
    assert (tmp_path / design["reference"]["audio"]).is_file()
    assert result["profile"]["model"].endswith("-Base")


def test_clone_without_authorization_is_refused_by_the_engine():
    """The engine-side gate: designs lacking authorization refuse to clone."""
    unauthorized = {"reference": {"audio": "x.wav", "text": "hola", "authorization": {}}}
    ok, reason = clone_authorized(unauthorized)
    assert not ok
    assert "authorization" in reason


# --- generation failure surfaces cleanly ------------------------------------------------


def test_generation_failure_is_a_clean_error(tmp_path):
    failing = VoiceDesigner(
        root=tmp_path,
        identities_dir=tmp_path / "identities",
        profiles_dir=tmp_path / "profiles",
        designs_dir=tmp_path / "designs",
        synth=lambda text, profile, out_path: {
            "status": "unavailable", "engine": "qwen3_tts", "message": "not installed",
        },
    )
    with pytest.raises(DesignerError, match="not installed"):
        failing.generate(_request())
