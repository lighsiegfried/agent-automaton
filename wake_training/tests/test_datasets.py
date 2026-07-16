"""Dataset discovery, counts/durations, hard negatives, and status."""

import numpy as np
import soundfile as sf

from trainer import datasets
from trainer.config import load_config


def _wav(path, seconds=0.5, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * sr), dtype="float32"), sr)
    return path


def test_audio_files_recursive_sorted_and_filtered(tmp_path):
    _wav(tmp_path / "a.wav")
    _wav(tmp_path / "sub" / "b.flac")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = datasets.audio_files(tmp_path)
    assert [p.name for p in found] == ["a.wav", "b.flac"]
    assert datasets.count_audio(tmp_path) == 2


def test_audio_files_missing_dir_is_empty(tmp_path):
    assert datasets.audio_files(tmp_path / "nope") == []
    assert datasets.count_audio(tmp_path / "nope") == 0


def test_total_duration_sums_clips(tmp_path):
    _wav(tmp_path / "a.wav", seconds=0.5)
    _wav(tmp_path / "b.wav", seconds=1.0)
    assert datasets.total_duration_seconds(tmp_path) == 1.5


def test_directory_summary_reports_presence_and_size(tmp_path):
    _wav(tmp_path / "a.wav", seconds=1.0)
    summary = datasets.directory_summary(tmp_path)
    assert summary["exists"] is True
    assert summary["count"] == 1
    assert summary["duration_seconds"] == 1.0


def test_resolve_dataset_path_relative_vs_absolute(tmp_path):
    rel = datasets.resolve_dataset_path("datasets/negative/speech", root=tmp_path)
    assert rel == tmp_path / "datasets" / "negative" / "speech"
    absolute = (tmp_path / "x").resolve()
    assert datasets.resolve_dataset_path(absolute) == absolute


def test_hard_negatives_come_from_config():
    config = load_config("fifi")
    assert "FIFA" in datasets.hard_negatives(config)
    assert "Fina" in datasets.hard_negatives(config)


def test_write_hard_negative_manifest(tmp_path):
    config = load_config("fifi")
    out = datasets.write_hard_negative_manifest(config, tmp_path / "neg.txt")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert "FIFA" in lines


def test_dataset_status_reports_readiness(tmp_path):
    config = load_config("fifi")
    # No background audio yet -> not ready to train.
    status = datasets.dataset_status(config, root=tmp_path)
    assert status["ready_to_train"] is False
    assert status["false_positive_validation_data"]["present"] is False
    assert "FIFA" in status["hard_negative_phrases"]

    # Add background audio at the first configured background path.
    bg = datasets.resolve_dataset_path(config.negatives.background_paths[0], root=tmp_path)
    _wav(bg / "noise1.wav", seconds=3600.0 / 3600.0 * 60)  # 60s of "background"
    # And the false-positive validation feature file.
    fp = datasets.resolve_dataset_path(
        config.negatives.false_positive_validation_data, root=tmp_path
    )
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(b"\x00")
    status = datasets.dataset_status(config, root=tmp_path)
    assert status["ready_to_train"] is True
    assert status["background_hours"] > 0
    assert status["false_positive_validation_data"]["present"] is True
