"""Start Dockerized Ollama (GPU-first) and run the unattended LLM smoke test.

This helper is Python (not PowerShell) on purpose: this machine's Group
Policy enforces AllSigned, which blocks unsigned .ps1 files entirely.

Steps:
  1. docker compose --profile llm up -d ollama   (profile-gated support service)
  2. wait until Ollama answers on 127.0.0.1:11434
  3. verify the NVIDIA GPU is visible INSIDE the container — CPU inference is
     refused unless ALLOW_CPU_OLLAMA=true (env or .env)
  4. pull the configured model if missing (delegated to llm_smoke)
  5. run scripts/llm_smoke.py against it

Docker only ever runs support services — it cannot control the Windows
desktop, and the smoke test refuses to run against real Windows tools.

Usage:
  python scripts/docker_llm.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_smoke  # noqa: E402
import prewarm_runtime  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434"
COMPOSE_PROFILE = "llm"
COMPOSE_UP_OLLAMA = ["docker", "compose", "--profile", COMPOSE_PROFILE, "up", "-d", "ollama"]

GPU_HELP = """\
No NVIDIA GPU is visible inside the Ollama container, and CPU inference is
disabled by default. To fix GPU access (Windows / Docker Desktop):
  1. Install the latest NVIDIA driver on Windows (the WSL2 CUDA driver is included).
  2. Update WSL2:           wsl --update
  3. Docker Desktop -> Settings -> General: use the WSL 2 based engine.
  4. Verify GPU passthrough: docker run --rm --gpus all ubuntu nvidia-smi
  5. Recreate the service:   docker compose --profile llm up -d --force-recreate ollama

If you really want slow CPU-only inference instead, opt in explicitly:
  set ALLOW_CPU_OLLAMA=true in .env (or the environment) and re-run."""


def docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "--version"], capture_output=True, timeout=20
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def start_ollama() -> bool:
    print("Starting Dockerized Ollama:", " ".join(COMPOSE_UP_OLLAMA))
    return subprocess.run(COMPOSE_UP_OLLAMA, cwd=PROJECT_ROOT).returncode == 0


def wait_for_ollama(base_url: str, timeout_seconds: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if llm_smoke.ollama_available(base_url):
            return True
        time.sleep(2.0)
    return False


def get_ollama_container_id() -> str | None:
    try:
        result = subprocess.run(
            ["docker", "compose", "--profile", COMPOSE_PROFILE, "ps", "-q", "ollama"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if result.returncode == 0 and lines else None


def detect_gpu(container_id: str) -> tuple[bool, str]:
    """(gpu_visible, gpu_name_or_reason) — checked INSIDE the container."""
    try:
        result = subprocess.run(
            [
                "docker", "exec", container_id,
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return True, result.stdout.strip().splitlines()[0].strip()

    # Fallback: ollama logs the compute library it found at startup,
    # e.g. 'inference compute ... library=cuda ... name="NVIDIA ..."'.
    try:
        logs = subprocess.run(
            ["docker", "logs", container_id], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "nvidia-smi failed and container logs are unavailable"
    text = (logs.stdout or "") + (logs.stderr or "")
    for line in text.splitlines():
        if "inference compute" in line and (
            "cuda" in line.lower() or "nvidia" in line.lower()
        ):
            match = re.search(r'name="?([^"\s]+(?:\s+[^"\s]+)*)"?', line)
            return True, match.group(1) if match else "CUDA device (from ollama logs)"
    return False, "no NVIDIA GPU visible inside the container"


def cpu_allowed() -> bool:
    """CPU inference is opt-in only (ALLOW_CPU_OLLAMA=true in env or .env)."""
    return llm_smoke.config_value("ALLOW_CPU_OLLAMA", "false").strip().lower() == "true"


def main() -> int:
    model = llm_smoke.config_value("LLM_PLANNER_MODEL", llm_smoke.DEFAULT_MODEL)
    print("=== Fifi Dockerized Ollama (GPU-first) ===")
    print(f"Compose profile : {COMPOSE_PROFILE}")
    print(f"Model           : {model}")

    if not docker_available():
        print(
            "Docker is not available. Install/start Docker Desktop first, or run\n"
            "Ollama natively and use:  python scripts/llm_smoke.py"
        )
        return 1
    if not start_ollama():
        print(
            "docker compose failed to start the ollama service (see output above).\n"
            "A 'could not select device driver' error means GPU passthrough is\n"
            "not set up — see the checklist below.\n\n" + GPU_HELP
        )
        return 1
    if not wait_for_ollama(OLLAMA_URL):
        print(
            f"Ollama did not answer at {OLLAMA_URL} in time. Check the container:\n"
            "  docker compose --profile llm logs ollama"
        )
        return 1
    print("Ollama is up.")

    container_id = get_ollama_container_id()
    print(f"Container       : {container_id or '(could not resolve id)'}")
    if container_id:
        gpu_detected, gpu_detail = detect_gpu(container_id)
    else:
        gpu_detected, gpu_detail = False, "could not resolve the ollama container id"
    print(f"GPU detected    : {'yes' if gpu_detected else 'NO'}")
    print(f"GPU info        : {gpu_detail}")

    if not gpu_detected:
        if cpu_allowed():
            print(
                "WARNING: ALLOW_CPU_OLLAMA=true — continuing with CPU-only inference. "
                "Expect slow planner/response calls."
            )
        else:
            print()
            print(GPU_HELP)
            return 1

    # Warm the model into VRAM (keep_alive=-1) and verify GPU placement before
    # smoking — the first real command must never pay a cold load (Phase 3D.1).
    ok, detail = llm_smoke.ensure_model(OLLAMA_URL, model)
    if not ok:
        print(f"Model not ready : {detail}")
        return 1
    if prewarm_runtime.prewarm_llm_enabled():
        if not prewarm_runtime.warm_llm(OLLAMA_URL, model):
            return 1
    else:
        print("LLM prewarm     : skipped (RUNTIME_PREWARM_LLM=false)")

    exit_code = llm_smoke.main(["--ollama-url", OLLAMA_URL, "--model", model])
    print(f"Smoke test      : {'PASS' if exit_code == 0 else 'FAIL'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
