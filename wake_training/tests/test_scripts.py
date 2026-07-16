"""Pure helpers in setup.py and record_samples.py (no venv, mic, or network)."""

import importlib.util
import sys
from pathlib import Path

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKE_TRAINING_ROOT / "scripts"))

import record_samples  # noqa: E402


def _load_setup():
    spec = importlib.util.spec_from_file_location(
        "wake_setup", WAKE_TRAINING_ROOT / "scripts" / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_torch_index_env_override(monkeypatch):
    setup = _load_setup()
    monkeypatch.delenv("WAKE_TRAIN_TORCH_CUDA_INDEX", raising=False)
    assert "cu128" in setup.torch_cuda_index()
    monkeypatch.setenv("WAKE_TRAIN_TORCH_CUDA_INDEX", "https://example/cu999")
    assert setup.torch_cuda_index() == "https://example/cu999"


def test_setup_python_version_ok():
    setup = _load_setup()
    ok, version = setup.python_version_ok()
    assert isinstance(ok, bool) and version


def test_sample_plan_varies_conditions():
    plan = record_samples.sample_plan(10)
    assert len(plan) == 10
    assert plan[0]["index"] == 1 and plan[-1]["index"] == 10
    # Multiple distinct distances/rooms are represented.
    assert len({p["distance"] for p in plan}) >= 2
    assert len({p["room"] for p in plan}) >= 2


def test_next_index_and_recording_path(tmp_path):
    assert record_samples.next_index(tmp_path) == 1
    (tmp_path / "fifi_001.wav").write_bytes(b"x")
    assert record_samples.next_index(tmp_path) == 2
    path = record_samples.recording_path(2, tmp_path)
    assert path.name == "fifi_002.wav"
