"""Training-config validation, inheritance, and openWakeWord translation."""

import pytest
import yaml

from trainer import config as cfg
from trainer.config import ConfigError, TrainingConfig, from_mapping, load_config


# --- the real candidate configs ----------------------------------------------------


def test_fifi_candidate_loads_and_inherits_defaults():
    config = load_config("fifi")
    assert config.model_name == "fifi"
    assert "fifi" in [p.lower() for p in config.target_phrase]
    # Inherited from defaults.yaml.
    assert config.language == "es"
    assert config.random_seed == 1234
    assert config.positive.n_samples > 0
    assert config.validate() == []


def test_oye_fifi_overrides_hard_negatives():
    config = load_config("oye_fifi")
    assert config.model_name == "oye_fifi"
    assert any("oye" in p for p in config.target_phrase)
    # The phrase now CONTAINS "oye", so bare "oye" must not be a hard negative.
    assert "oye" not in config.hard_negative_phrases
    assert "oye Sofía" in config.hard_negative_phrases
    assert config.validate() == []


def test_both_candidates_are_valid():
    for name in ("fifi", "oye_fifi"):
        assert load_config(name).validate() == []


# --- inheritance / merge semantics -------------------------------------------------


def test_extends_deep_merges_maps_and_replaces_lists(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "model_name": "base",
                "target_phrase": ["base"],
                "positive": {"n_samples": 10, "n_samples_val": 5,
                             "piper_voices": ["a", "b"]},
                "hard_negative_phrases": ["x", "y"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "model_name": "child",
                # nested map: deep-merge (n_samples overridden, val inherited)
                "positive": {"n_samples": 99},
                # list: REPLACE, not append
                "hard_negative_phrases": ["z"],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(tmp_path / "child.yaml")
    assert config.model_name == "child"
    assert config.target_phrase == ["base"]              # inherited
    assert config.positive.n_samples == 99               # overridden
    assert config.positive.n_samples_val == 5            # deep-merged from base
    assert config.positive.piper_voices == ["a", "b"]    # inherited
    assert config.hard_negative_phrases == ["z"]         # replaced, not appended


def test_circular_extends_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nmodel_name: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nmodel_name: b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_extends_target_raises(tmp_path):
    (tmp_path / "c.yaml").write_text("extends: nope.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="extends target not found"):
        load_config(tmp_path / "c.yaml")


def test_unknown_key_in_section_raises(valid_mapping):
    valid_mapping["training"]["bogus"] = 1
    with pytest.raises(ConfigError, match="unknown keys in 'training'.*bogus"):
        from_mapping(valid_mapping)


def test_missing_config_file_raises():
    with pytest.raises(ConfigError, match="config not found"):
        load_config("does-not-exist")


# --- validation --------------------------------------------------------------------


def test_valid_mapping_has_no_problems(valid_mapping):
    assert from_mapping(valid_mapping).validate() == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda m: m.update(target_phrase=[]), "target_phrase"),
        (lambda m: m.update(target_phrase=["   "]), "non-empty string"),
        (lambda m: m.update(model_name=""), "model_name"),
        (lambda m: m.update(random_seed="x"), "random_seed"),
        (lambda m: m.update(language=""), "language"),
        (lambda m: m["positive"].update(n_samples=0), "positive.n_samples"),
        (lambda m: m["positive"].update(n_samples_val=-1), "positive.n_samples_val"),
        (lambda m: m["positive"].update(piper_voices=[]), "piper_voices"),
        (lambda m: m["augmentation"].update(speed=[1.2, 0.9]), "augmentation.speed"),
        (lambda m: m["augmentation"].update(noise_snr_db=[-1.0, 5.0]), "noise_snr_db"),
        (lambda m: m["augmentation"].update(pitch_semitones=[1.0]), "pitch_semitones"),
        (lambda m: m["training"].update(model_type="svm"), "model_type"),
        (lambda m: m["training"].update(steps=0), "training.steps"),
        (lambda m: m["training"].update(target_recall=2.0), "target_recall"),
        (lambda m: m["negatives"].update(background_duplication_rate=0), "duplication_rate"),
        (lambda m: m["evaluation"].update(min_recall=1.5), "min_recall"),
        (lambda m: m["evaluation"].update(conditions=[]), "conditions"),
        (lambda m: m["verifier"].update(threshold=9.0), "verifier.threshold"),
    ],
)
def test_validation_catches_bad_fields(valid_mapping, mutate, needle):
    mutate(valid_mapping)
    problems = from_mapping(valid_mapping).validate()
    assert any(needle in p for p in problems), (needle, problems)


def test_raise_if_invalid_aggregates(valid_mapping):
    valid_mapping.update(model_name="", target_phrase=[])
    with pytest.raises(ConfigError) as exc:
        from_mapping(valid_mapping).raise_if_invalid()
    assert "model_name" in str(exc.value)
    assert "target_phrase" in str(exc.value)


# --- openWakeWord translation ------------------------------------------------------


def test_to_openwakeword_config_has_expected_keys(valid_mapping):
    oww = from_mapping(valid_mapping).to_openwakeword_config(
        output_dir="/out/fifi", piper_sample_generator_path="/piper"
    )
    # Keys openWakeWord's train.py actually reads.
    for key in (
        "model_name", "target_phrase", "custom_negative_phrases", "n_samples",
        "n_samples_val", "steps", "max_negative_weight",
        "target_false_positives_per_hour", "output_dir", "background_paths",
        "background_paths_duplication_rate", "rir_paths",
        "false_positive_validation_data_path", "augmentation_rounds",
        "augmentation_batch_size", "tts_batch_size", "model_type", "layer_size",
        "batch_n_per_class", "piper_sample_generator_path",
    ):
        assert key in oww, f"missing openWakeWord key: {key}"
    assert oww["model_name"] == "fifi"
    assert oww["custom_negative_phrases"] == ["FIFA", "Fina"]
    assert oww["output_dir"] == "/out/fifi"
    assert oww["piper_sample_generator_path"] == "/piper"
    # duplication rate is one entry per background path.
    assert oww["background_paths_duplication_rate"] == [1]


def test_write_openwakeword_config_roundtrips(valid_mapping, tmp_path):
    config = from_mapping(valid_mapping)
    out = config.write_openwakeword_config(
        tmp_path / "gen" / "train.yaml", output_dir=tmp_path / "o", piper_sample_generator_path="/p"
    )
    assert out.is_file()
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "fifi"
    assert loaded["target_phrase"] == ["fifi"]


def test_config_module_exposes_project_relative_config_dir():
    assert (cfg.paths.CONFIG_DIR / "defaults.yaml").is_file()
