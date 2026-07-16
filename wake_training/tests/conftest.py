"""Shared fixtures for the wake-word trainer tests.

Puts wake_training/ on sys.path so ``import trainer.<module>`` resolves, and
provides a minimal valid config mapping the validation tests mutate. These tests
are deliberately dependency-light (PyYAML + numpy + onnxruntime at most) so they
run in the MAIN project's .venv without the isolated training stack installed.
"""

import sys
from pathlib import Path

import pytest

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(WAKE_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(WAKE_TRAINING_ROOT))


@pytest.fixture
def valid_mapping() -> dict:
    """A fully-populated, valid merged config mapping."""
    return {
        "model_name": "fifi",
        "target_phrase": ["fifi"],
        "language": "es",
        "random_seed": 1234,
        "positive": {
            "n_samples": 1000,
            "n_samples_val": 100,
            "piper_voices": ["es_ES-carlfm-x_low"],
            "length_scales": [1.0],
            "noise_scales": [0.667],
        },
        "augmentation": {
            "rounds": 1,
            "batch_size": 8,
            "pitch_semitones": [-2.0, 2.0],
            "speed": [0.95, 1.05],
            "gain_db": [-3.0, 3.0],
            "noise_snr_db": [10.0, 20.0],
            "reverb": True,
        },
        "hard_negative_phrases": ["FIFA", "Fina"],
        "negatives": {
            "background_paths": ["datasets/negative/speech"],
            "background_duplication_rate": 1,
            "rir_paths": ["datasets/negative/rir"],
            "false_positive_validation_data": "datasets/validation/neg.npy",
        },
        "training": {
            "steps": 100,
            "max_negative_weight": 100,
            "model_type": "dnn",
            "layer_size": 96,
            "batch_n_per_class": 128,
            "target_false_positives_per_hour": 0.2,
            "target_recall": 0.6,
        },
        "evaluation": {
            "target_false_positives_per_hour": 0.5,
            "min_recall": 0.7,
            "conditions": ["quiet", "noisy"],
        },
        "verifier": {
            "enabled": False,
            "threshold": 0.3,
            "reference_dir": "datasets/positive/user",
        },
    }
