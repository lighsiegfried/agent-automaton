"""Training orchestration: seeding, command construction, output discovery.

No real training happens — the openWakeWord subprocess is a fake ``runner`` that
records the command and (optionally) creates the expected ONNX artifact.
"""

import types
from pathlib import Path

import pytest

from trainer import paths, pipeline
from trainer.config import load_config


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """Redirect all writable workspace dirs under tmp so tests leave no trace."""
    for attr in (
        "OUTPUTS_DIR", "REPORTS_DIR", "POSITIVE_DIR", "NEGATIVE_DIR",
        "VALIDATION_DIR", "USER_RECORDINGS_DIR", "PIPER_DIR", "DATASETS_DIR",
    ):
        monkeypatch.setattr(paths, attr, tmp_path / attr.lower())
    return tmp_path


class FakeRunner:
    """Stands in for subprocess.run: answers the find_spec probe and 'trains'."""

    def __init__(self, pkg_dir: Path, model_path: Path | None = None,
                 train_returncode: int = 0, create_model: bool = True):
        self.pkg_dir = str(pkg_dir)
        self.model_path = model_path
        self.train_returncode = train_returncode
        self.create_model = create_model
        self.calls: list = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if len(cmd) >= 2 and cmd[1] == "-c":            # the openWakeWord probe
            return types.SimpleNamespace(returncode=0, stdout=self.pkg_dir + "\n", stderr="")
        if self.create_model and self.model_path:        # the training run
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_path.write_bytes(b"onnx-bytes")
        return types.SimpleNamespace(returncode=self.train_returncode, stdout="", stderr="")


# --- reproducibility ---------------------------------------------------------------


def test_seed_everything_is_deterministic():
    import numpy as np

    pipeline.seed_everything(7)
    first = np.random.rand(3).tolist()
    pipeline.seed_everything(7)
    second = np.random.rand(3).tolist()
    assert first == second


def test_subprocess_env_pins_hashseed():
    env = pipeline.subprocess_env(99, base_env={"PATH": "/x"})
    assert env["PYTHONHASHSEED"] == "99"
    assert env["WAKE_TRAIN_SEED"] == "99"
    assert env["PATH"] == "/x"  # base env preserved


# --- command construction ----------------------------------------------------------


def test_build_train_command_orders_flags_canonically():
    cmd = pipeline.build_train_command(
        ["py", "train.py"], "cfg.yaml", stages=("train", "generate"), overwrite=True
    )
    assert cmd[:4] == ["py", "train.py", "--training_config", "cfg.yaml"]
    # canonical order regardless of the stages tuple order
    assert cmd[4:] == ["--generate_clips", "--train_model", "--overwrite"]


def test_locate_train_entry_prefers_train_py(tmp_path):
    pkg = tmp_path / "openwakeword"
    pkg.mkdir()
    (pkg / "train.py").write_text("# trainer", encoding="utf-8")
    fake = FakeRunner(pkg)
    entry = pipeline.locate_train_entry("py", runner=fake)
    assert entry == ["py", str(pkg / "train.py")]


def test_locate_train_entry_falls_back_to_module(tmp_path):
    pkg = tmp_path / "openwakeword"
    pkg.mkdir()  # no train.py file
    entry = pipeline.locate_train_entry("py", runner=FakeRunner(pkg))
    assert entry == ["py", "-m", "openwakeword.train"]


def test_locate_train_entry_missing_package_raises():
    empty = FakeRunner(Path(""))  # probe prints empty -> not installed
    empty.pkg_dir = ""
    with pytest.raises(RuntimeError, match="not installed"):
        pipeline.locate_train_entry("py", runner=empty)


# --- full run ----------------------------------------------------------------------


def test_run_training_happy_path_discovers_model_and_reports(workspace):
    config = load_config("fifi")
    model_path = paths.candidate_model_path("fifi")
    pkg = workspace / "pkg" / "openwakeword"
    pkg.mkdir(parents=True)
    (pkg / "train.py").write_text("# trainer", encoding="utf-8")
    fake = FakeRunner(pkg, model_path=model_path)

    result = pipeline.run_training(config, "fifi", venv_python="py", runner=fake)

    assert result["status"] == "ok"
    assert result["model_present"] is True
    assert result["return_code"] == 0
    # The generated openWakeWord config was written and referenced.
    assert Path(result["generated_config"]).is_file()
    # A provenance report was written.
    assert Path(result["report_path"]).is_file()
    # The training command targeted the generated config with all three stages.
    train_call = [c for c, _ in fake.calls if "--training_config" in c][0]
    assert "--generate_clips" in train_call
    assert "--augment_clips" in train_call
    assert "--train_model" in train_call


def test_run_training_errors_when_model_missing_despite_exit_zero(workspace):
    config = load_config("fifi")
    pkg = workspace / "pkg" / "openwakeword"
    pkg.mkdir(parents=True)
    (pkg / "train.py").write_text("# trainer", encoding="utf-8")
    fake = FakeRunner(pkg, model_path=paths.candidate_model_path("fifi"),
                      create_model=False)  # trainer 'succeeds' but produces nothing

    result = pipeline.run_training(config, "fifi", venv_python="py", runner=fake)
    assert result["status"] == "error"
    assert "not produced" in result["message"]


def test_run_training_rejects_invalid_config(workspace):
    config = load_config("fifi")
    config.target_phrase = []  # now invalid
    result = pipeline.run_training(config, "fifi", venv_python="py",
                                   runner=FakeRunner(workspace), entry=["py", "t"])
    assert result["status"] == "error"
    assert result["stage"] == "validate"


def test_run_training_with_explicit_entry_skips_locate(workspace):
    config = load_config("fifi")
    model_path = paths.candidate_model_path("fifi")
    fake = FakeRunner(workspace, model_path=model_path)
    result = pipeline.run_training(
        config, "fifi", venv_python="py", runner=fake, entry=["py", "train.py"]
    )
    assert result["status"] == "ok"
    # No probe call happened (entry supplied).
    assert all(c[1] != "-c" for c, _ in fake.calls)
