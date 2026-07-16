"""One-shot setup for the ISOLATED wake-word training environment (Phase 3D.2).

Creates wake_training/.venv, installs the CUDA build of torch when an NVIDIA GPU
is present (cu128 for RTX 50 / Blackwell), installs requirements.txt into that
venv, clones piper-sample-generator (Spanish synthetic speech) into the
workspace, copies .env.example to .env, and prints diagnostics. The MAIN project
.venv is NEVER touched.

    python wake_training/scripts/setup.py                 # venv + deps + piper
    python wake_training/scripts/setup.py --download-data  # also fetch corpora/voices help
    python wake_training/scripts/setup.py --diagnose       # diagnostics only, change nothing

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = WAKE_TRAINING_ROOT / ".venv"
REQUIREMENTS = WAKE_TRAINING_ROOT / "requirements.txt"
PIPER_DIR = WAKE_TRAINING_ROOT / "piper-sample-generator"
PIPER_REPO = "https://github.com/rhasspy/piper-sample-generator"

DEFAULT_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
MIN_PYTHON = (3, 10)


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def torch_cuda_index() -> str:
    return os.environ.get("WAKE_TRAIN_TORCH_CUDA_INDEX", DEFAULT_TORCH_CUDA_INDEX)


def python_version_ok() -> tuple[bool, str]:
    v = sys.version_info
    return v >= MIN_PYTHON, f"{v.major}.{v.minor}.{v.micro}"


def detect_nvidia_gpu() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    return {"name": parts[0] if parts else "unknown",
            "vram": parts[1] if len(parts) > 1 else "?",
            "driver": parts[2] if len(parts) > 2 else "?"}


def pip_install(args: list[str]) -> bool:
    return subprocess.run(
        [str(venv_python()), "-m", "pip", "install", *args], cwd=WAKE_TRAINING_ROOT
    ).returncode == 0


def check_venv_import(module: str) -> tuple[bool, str]:
    if not venv_python().exists():
        return False, "venv missing"
    code = f"import {module}; print(getattr({module}, '__version__', 'ok'))"
    try:
        result = subprocess.run(
            [str(venv_python()), "-c", code], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "check failed"
    if result.returncode != 0:
        return False, "import failed"
    return True, (result.stdout or "").strip().splitlines()[-1][:60]


def print_diagnostics() -> dict:
    py_ok, py_version = python_version_ok()
    gpu = detect_nvidia_gpu()
    print("--- diagnostics ---")
    print(f"Python          : {py_version} ({'ok' if py_ok else 'needs >= 3.10'})")
    print(f"GPU             : {gpu['name'] + ' (' + gpu['vram'] + ')' if gpu else 'none detected'}")
    checks = {}
    if venv_python().exists():
        for module in ("torch", "openwakeword", "onnxruntime", "audiomentations", "soundfile"):
            ok, detail = check_venv_import(module)
            checks[module] = ok
            print(f"venv {module:<14}: {'ok ' + detail if ok else 'MISSING'}")
        cuda_ok, cuda_detail = (False, "")
        if checks.get("torch"):
            cuda_ok, cuda_detail = check_venv_import("torch")  # version only; deep check below
            ok, detail = _venv_probe("import torch; print(torch.cuda.is_available())")
            checks["cuda"] = ok and detail.strip() == "True"
            print(f"venv CUDA       : {detail if ok else 'torch cannot report CUDA'}")
    else:
        print("venv            : not created yet")
    print(f"piper generator : {'present' if PIPER_DIR.exists() else 'not cloned'}")
    return {"python_ok": py_ok, "gpu": gpu, "checks": checks}


def _venv_probe(code: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [str(venv_python()), "-c", code], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return result.returncode == 0, (result.stdout or "").strip()


def clone_piper() -> bool:
    if PIPER_DIR.exists():
        print("piper generator : already cloned — left untouched")
        return True
    if not shutil.which("git"):
        print("piper generator : git not found — clone it manually:")
        print(f"  git clone {PIPER_REPO} {PIPER_DIR}")
        return False
    print("piper generator : cloning rhasspy/piper-sample-generator ...")
    ok = subprocess.run(["git", "clone", "--depth", "1", PIPER_REPO, str(PIPER_DIR)]).returncode == 0
    if ok and (PIPER_DIR / "requirements.txt").is_file():
        pip_install(["-r", str(PIPER_DIR / "requirements.txt")])
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the isolated wake-word trainer.")
    parser.add_argument("--diagnose", action="store_true", help="diagnostics only")
    parser.add_argument("--skip-install", action="store_true", help="dirs/.env/diagnostics only")
    parser.add_argument("--download-data", action="store_true",
                        help="clone piper + print corpus download guidance")
    args = parser.parse_args(argv)

    print("=== Fifi wake-word trainer setup (isolated) ===")
    print(f"Location        : {WAKE_TRAINING_ROOT}")
    if args.diagnose:
        report = print_diagnostics()
        return 0 if report["python_ok"] else 1

    py_ok, py_version = python_version_ok()
    if not py_ok:
        print(f"Python {py_version} is too old — the trainer needs >= 3.10.")
        return 1

    from trainer import paths as trainer_paths

    trainer_paths.ensure_workspace_dirs()
    print("Directories     : datasets/ outputs/ reports/ ready")

    env_file = WAKE_TRAINING_ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(WAKE_TRAINING_ROOT / ".env.example", env_file)
        print("Config          : .env created from .env.example")

    if args.skip_install:
        print_diagnostics()
        return 0

    gpu = detect_nvidia_gpu()
    if not venv_python().exists():
        print("Virtualenv      : creating wake_training/.venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    if gpu:
        print(f"Torch           : installing CUDA build ({torch_cuda_index()}) ...")
        torch_ok = pip_install(["torch", "--index-url", torch_cuda_index()])
    else:
        print("Torch           : no GPU — installing CPU build.")
        torch_ok = pip_install(["torch"])
    if not torch_ok:
        print("Torch install FAILED — the main project environment was not touched.")
        return 1

    print("Dependencies    : installing requirements.txt into wake_training/.venv ...")
    if not pip_install(["-r", str(REQUIREMENTS)]):
        print("Install FAILED — the main project environment was not touched.")
        return 1

    clone_piper()
    if args.download_data:
        print("Corpora         : point config/*.yaml negatives.* at local 16 kHz audio,")
        print("                  or fetch openWakeWord's precomputed negatives + RIRs and")
        print("                  Spanish piper voices (es_ES-*, es_MX-*) — see README.md.")

    report = print_diagnostics()
    ready = report["checks"].get("openwakeword") and report["checks"].get("torch")
    if gpu and not report["checks"].get("cuda"):
        print("WARNING: a GPU is present but torch cannot use CUDA — training will be slow.")
    print("Trainer ready. Try:  wake_training/.venv/Scripts/python.exe "
          "wake_training/scripts/wake_trainer.py status" if ready
          else "Setup finished with problems — see diagnostics above.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
