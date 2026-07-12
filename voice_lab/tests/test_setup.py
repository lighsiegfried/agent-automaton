"""setup.py diagnostics — GPU/VRAM/Python detection with subprocess mocked."""

import importlib.util
import sys

import pytest

from conftest import VOICE_LAB_ROOT


@pytest.fixture
def setup_module():
    path = VOICE_LAB_ROOT / "scripts" / "setup.py"
    spec = importlib.util.spec_from_file_location("voice_lab_setup", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["voice_lab_setup"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("voice_lab_setup", None)


class FakeCompleted:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_detect_nvidia_gpu_parses_smi_output(setup_module, monkeypatch):
    monkeypatch.setattr(setup_module.shutil, "which", lambda name: "C:/fake/nvidia-smi.exe")
    monkeypatch.setattr(
        setup_module.subprocess, "run",
        lambda *a, **k: FakeCompleted("NVIDIA GeForce RTX 5060 Ti, 16384 MiB, 581.57\n"),
    )
    gpu = setup_module.detect_nvidia_gpu()
    assert gpu == {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0, "driver": "581.57"}


def test_detect_nvidia_gpu_none_without_smi(setup_module, monkeypatch):
    monkeypatch.setattr(setup_module.shutil, "which", lambda name: None)
    assert setup_module.detect_nvidia_gpu() is None


def test_detect_nvidia_gpu_survives_smi_failure(setup_module, monkeypatch):
    monkeypatch.setattr(setup_module.shutil, "which", lambda name: "smi")
    monkeypatch.setattr(
        setup_module.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    assert setup_module.detect_nvidia_gpu() is None


def test_python_version_check(setup_module):
    ok, version = setup_module.python_version_ok()
    assert ok  # this suite runs on a supported interpreter
    assert version.count(".") == 2


def test_torch_installs_from_blackwell_capable_index(setup_module, monkeypatch):
    """RTX 50 (sm_120) needs cu128 wheels — pin the index used for GPU installs."""
    assert "cu128" in setup_module.TORCH_CUDA_INDEX
    installed = []
    monkeypatch.setattr(setup_module, "pip_install", lambda args: installed.append(args) or True)
    assert setup_module.install_torch(gpu_present=True)
    assert installed == [["torch", "--index-url", setup_module.TORCH_CUDA_INDEX]]
    installed.clear()
    assert setup_module.install_torch(gpu_present=False)
    assert installed == [["torch"]]  # CPU fallback, no CUDA index


def test_setup_never_touches_the_main_environment(setup_module):
    """Everything installs through the voice_lab venv interpreter only."""
    assert str(setup_module.VENV_DIR).endswith(".venv")
    assert str(setup_module.VENV_DIR.parent).endswith("voice_lab")
    source = (VOICE_LAB_ROOT / "scripts" / "setup.py").read_text(encoding="utf-8")
    # sys.executable appears exactly once: to CREATE the venv. Every pip
    # invocation goes through venv_python() — never the main interpreter.
    assert source.count("sys.executable") == 1
    assert '[sys.executable, "-m", "venv"' in source
    assert '[str(venv_python()), "-m", "pip", "install"' in source


def test_diagnose_flag_changes_nothing(setup_module, monkeypatch, capsys):
    monkeypatch.setattr(setup_module, "detect_nvidia_gpu", lambda: {"name": "X", "vram_gb": 16.0, "driver": "1"})
    monkeypatch.setattr(setup_module, "check_venv_import", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(
        setup_module.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("--diagnose must not run anything")),
    )
    monkeypatch.setattr(
        setup_module, "pip_install",
        lambda args: (_ for _ in ()).throw(AssertionError("--diagnose must not install")),
    )
    assert setup_module.main(["--diagnose"]) == 0
    out = capsys.readouterr().out
    assert "VRAM" in out and "16.0" in out
