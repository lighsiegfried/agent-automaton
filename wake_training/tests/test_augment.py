"""Augmentation spec/plan — pure, no audiomentations needed."""

from trainer import augment
from trainer.config import AugmentationConfig


def _cfg(**over) -> AugmentationConfig:
    base = dict(
        rounds=2, batch_size=16, pitch_semitones=[-3.0, 3.0], speed=[0.9, 1.1],
        gain_db=[-6.0, 6.0], noise_snr_db=[5.0, 25.0], reverb=True,
    )
    base.update(over)
    return AugmentationConfig(**base)


def test_transform_specs_cover_pitch_speed_gain_noise_reverb():
    names = [s["name"] for s in augment.transform_specs(_cfg())]
    assert names == ["PitchShift", "TimeStretch", "Gain", "AddGaussianSNR", "RoomSimulator"]


def test_transform_specs_map_ranges():
    specs = {s["name"]: s for s in augment.transform_specs(_cfg())}
    assert specs["PitchShift"]["min_semitones"] == -3.0
    assert specs["PitchShift"]["max_semitones"] == 3.0
    assert specs["TimeStretch"]["min_rate"] == 0.9
    assert specs["TimeStretch"]["max_rate"] == 1.1
    assert specs["Gain"]["min_gain_db"] == -6.0
    assert specs["AddGaussianSNR"]["min_snr_db"] == 5.0
    assert specs["AddGaussianSNR"]["max_snr_db"] == 25.0


def test_reverb_disabled_omits_room_simulator():
    names = [s["name"] for s in augment.transform_specs(_cfg(reverb=False))]
    assert "RoomSimulator" not in names
    assert names == ["PitchShift", "TimeStretch", "Gain", "AddGaussianSNR"]


def test_plan_is_readable_and_includes_params():
    lines = augment.plan(_cfg())
    assert any("PitchShift" in ln and "-3.0" in ln for ln in lines)
    assert any("p=0.3" in ln for ln in lines)  # RoomSimulator probability


def test_to_openwakeword_augmentation_maps_core_knobs():
    oww = augment.to_openwakeword_augmentation(_cfg(rounds=3, batch_size=8))
    assert oww == {
        "augmentation_rounds": 3,
        "augmentation_batch_size": 8,
        "apply_reverb": True,
    }
