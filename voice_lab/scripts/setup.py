"""One-shot Voice Lab environment setup — isolated from the main project.

Creates voice_lab/.venv, installs the CUDA build of torch when an NVIDIA GPU
is present (cu128 — required for RTX 50 / Blackwell), installs
voice_lab/requirements.txt into the venv, copies .env.example to .env if
missing, creates the storage/model directories, and prints setup diagnostics
(GPU, VRAM, driver, Python version, dependency status).

The main project's environment is NEVER touched; nothing is installed there.
Qwen3-TTS is optional and only installed with --with-qwen.

Usage:
  python voice_lab/scripts/setup.py                # full setup + diagnostics
  python voice_lab/scripts/setup.py --with-qwen    # also install Qwen3-TTS deps
  python voice_lab/scripts/setup.py --skip-install # dirs/.env/diagnostics only
  python voice_lab/scripts/setup.py --diagnose     # diagnostics only

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VOICE_LAB_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = VOICE_LAB_ROOT / ".venv"
REQUIREMENTS = VOICE_LAB_ROOT / "requirements.txt"
REQUIREMENTS_QWEN = VOICE_LAB_ROOT / "requirements-qwen.txt"

# CUDA 12.8 wheels support Blackwell (sm_120, e.g. RTX 5060 Ti).
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

MIN_PYTHON = (3, 10)


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


# --- diagnostics --------------------------------------------------------------------


def python_version_ok() -> tuple[bool, str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return sys.version_info >= MIN_PYTHON, version


def detect_nvidia_gpu() -> dict | None:
    """GPU name / VRAM / driver via nvidia-smi. None when no NVIDIA GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    gpu = {"name": parts[0] if parts else "unknown"}
    if len(parts) > 1:
        try:
            gpu["vram_gb"] = round(float(parts[1].split()[0]) / 1024, 1)  # "16384 MiB"
        except (ValueError, IndexError):
            gpu["vram_gb"] = None
    if len(parts) > 2:
        gpu["driver"] = parts[2]
    return gpu


def check_venv_import(module: str, probe: str = "") -> tuple[bool, str]:
    """(ok, detail) for importing a module inside the voice_lab venv."""
    if not venv_python().exists():
        return False, "venv missing"
    code = probe or f"import {module}; print(getattr({module}, '__version__', 'ok'))"
    try:
        result = subprocess.run(
            [str(venv_python()), "-c", code], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "check failed to run"
    if result.returncode != 0:
        reason = (result.stderr or "").strip().splitlines()
        return False, reason[-1][:120] if reason else "import failed"
    return True, (result.stdout or "").strip().splitlines()[-1][:120]


def print_diagnostics() -> dict:
    """Print the environment report; returns the findings for the caller."""
    py_ok, py_version = python_version_ok()
    gpu = detect_nvidia_gpu()
    print("--- diagnostics ---")
    print(f"Python          : {py_version} ({'ok' if py_ok else 'needs >= 3.10'})")
    if gpu:
        print(f"GPU             : {gpu['name']}")
        print(f"VRAM            : {gpu.get('vram_gb')} GB")
        print(f"Driver          : {gpu.get('driver')}")
    else:
        print("GPU             : none detected (Kokoro will run on CPU — slower)")

    checks = {}
    if venv_python().exists():
        for module in ("torch", "kokoro", "soundfile", "pyttsx3", "fastapi"):
            ok, detail = check_venv_import(module)
            checks[module] = ok
            print(f"venv {module:<11}: {'ok ' + detail if ok else 'MISSING (' + detail + ')'}")
        cuda_ok, cuda_detail = check_venv_import(
            "torch",
            "import torch; print(f'cuda={torch.cuda.is_available()} "
            "device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')",
        )
        checks["cuda"] = cuda_ok and "cuda=True" in cuda_detail
        print(f"venv CUDA       : {cuda_detail if cuda_ok else 'torch missing'}")
        qwen_ok, _ = check_venv_import("qwen_tts")
        checks["qwen_deps"] = qwen_ok
        print(f"venv qwen3_tts  : {'installed' if qwen_ok else 'not installed (optional; --with-qwen)'}")
    else:
        print("venv            : not created yet")
    return {"python_ok": py_ok, "gpu": gpu, "checks": checks}


# --- setup steps --------------------------------------------------------------------


def pip_install(args: list[str]) -> bool:
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", *args], cwd=VOICE_LAB_ROOT
    )
    return result.returncode == 0


def install_torch(gpu_present: bool) -> bool:
    """CUDA torch first (cu128 has Blackwell/sm_120 kernels); CPU as fallback."""
    if gpu_present:
        print("Torch           : installing CUDA build (cu128; ~3 GB, be patient)...")
        return pip_install(["torch", "--index-url", TORCH_CUDA_INDEX])
    print("Torch           : no GPU detected — installing CPU build.")
    return pip_install(["torch"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the isolated Voice Lab env.")
    parser.add_argument("--skip-install", action="store_true", help="dirs/.env/diagnostics only")
    parser.add_argument("--diagnose", action="store_true", help="diagnostics only, change nothing")
    parser.add_argument(
        "--with-qwen", action="store_true",
        help="also install the OPTIONAL Qwen3-TTS dependencies",
    )
    args = parser.parse_args(argv)

    print("=== Fifi Voice Lab setup (isolated) ===")
    print(f"Location        : {VOICE_LAB_ROOT}")

    if args.diagnose:
        report = print_diagnostics()
        return 0 if report["python_ok"] else 1

    py_ok, py_version = python_version_ok()
    if not py_ok:
        print(f"Python {py_version} is too old — Voice Lab needs >= 3.10.")
        return 1

    for sub in ("models", "storage/previews", "storage/cache", "storage/logs"):
        (VOICE_LAB_ROOT / sub).mkdir(parents=True, exist_ok=True)
    print("Directories     : models/ storage/{previews,cache,logs}/ ready")

    env_file = VOICE_LAB_ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(VOICE_LAB_ROOT / ".env.example", env_file)
        print("Config          : .env created from .env.example")
    else:
        print("Config          : .env already present — left untouched")

    if args.skip_install:
        print("Install         : skipped (--skip-install)")
        print_diagnostics()
        return 0

    gpu = detect_nvidia_gpu()
    if not venv_python().exists():
        print("Virtualenv      : creating voice_lab/.venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    else:
        print("Virtualenv      : voice_lab/.venv already exists")

    if not install_torch(gpu is not None):
        print("Torch install FAILED — the main project environment was not touched.")
        return 1
    print("Dependencies    : installing requirements.txt into voice_lab/.venv ...")
    if not pip_install(["-r", str(REQUIREMENTS)]):
        print("Install FAILED — the main project environment was not touched.")
        return 1
    if args.with_qwen:
        print("Qwen3-TTS       : installing OPTIONAL dependencies (--with-qwen)...")
        if not pip_install(["-r", str(REQUIREMENTS_QWEN)]):
            print("Qwen3-TTS install FAILED — Kokoro and the worker are unaffected.")
    else:
        print("Qwen3-TTS       : skipped (optional; rerun with --with-qwen to add it)")

    report = print_diagnostics()
    checks = report["checks"]
    ready = checks.get("torch") and checks.get("kokoro") and checks.get("fastapi")
    if gpu and not checks.get("cuda"):
        print("WARNING: a GPU is present but torch cannot use CUDA — Kokoro will run on CPU.")
    print(
        "Voice Lab ready. Start it with:  python scripts/local_runtime.py voice-start"
        if ready
        else "Setup finished with problems — see the diagnostics above."
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
