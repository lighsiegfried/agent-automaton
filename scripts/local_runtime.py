"""Local persistent runtime manager for Fifi (agent-automaton).

One command brings Fifi's whole local stack up: Dockerized Ollama (GPU-first,
a support service only) plus the FastAPI server on the **Windows host** (never
in a container — the desktop automation runner must run on the host).

    python scripts/local_runtime.py start        # bring everything up (prewarmed)
    python scripts/local_runtime.py status       # what is / isn't ready (incl. warm state)
    python scripts/local_runtime.py stop         # stop the host API (keep Ollama + model)
    python scripts/local_runtime.py restart
    python scripts/local_runtime.py smoke        # run the unattended LLM smoke test
    python scripts/local_runtime.py warm         # (re)load the LLM into VRAM + prewarm STT
    python scripts/local_runtime.py model-status # is the model loaded, and where?
    python scripts/local_runtime.py unload       # release VRAM (keep_alive=0; deletes nothing)
    python scripts/local_runtime.py wake         # launch the wake-word listener (explicit)

Design / safety:
- Docker only ever runs support services. `start` brings up the `ollama`
  service behind the `llm` profile; it never starts the API in Docker and never
  enables real Windows tools anywhere.
- The host API inherits its configuration from `.env` — this manager does not
  weaken safety, does not force real tools on, and cannot unblock destructive
  actions. Starting the runtime changes *where* things run, never *what* is
  permitted.
- GPU-first: with RUNTIME_REQUIRE_GPU=true (default) inference refuses to fall
  back to CPU unless ALLOW_CPU_OLLAMA=true is explicitly set.
- `stop` never deletes models or Docker volumes. Ollama keeps running unless
  --stop-ollama is passed (or RUNTIME_KEEP_OLLAMA_RUNNING=false).

Python (not PowerShell) on purpose: this machine's Group Policy enforces
AllSigned, which blocks unsigned .ps1 files entirely.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import docker_llm  # noqa: E402
import fifi_ptt  # noqa: E402
import fifi_wake  # noqa: E402
import llm_smoke  # noqa: E402
import prewarm_runtime  # noqa: E402

# Runtime state and logs live under storage/ (gitignored — never committed).
RUNTIME_DIR = PROJECT_ROOT / "storage" / "runtime"
LOG_DIR = PROJECT_ROOT / "storage" / "logs"
API_PID_FILE = RUNTIME_DIR / "api.pid"
API_LOG_FILE = LOG_DIR / "api.log"

# Voice Lab (Phase 3D.0) — isolated worker managed here, but its venv, models,
# and logs all live under voice_lab/ (only the PID is runtime state of ours).
VOICE_LAB_DIR = PROJECT_ROOT / "voice_lab"
VOICE_LAB_PID_FILE = RUNTIME_DIR / "voice_lab.pid"
VOICE_LAB_LOG_FILE = VOICE_LAB_DIR / "storage" / "logs" / "worker.log"
# CWD-independent launcher (Phase 3D.0.3a): pins voice_lab/ on sys.path itself.
VOICE_LAB_RUNNER = VOICE_LAB_DIR / "run_worker.py"
VOICE_LAB_BUILD_FILE = VOICE_LAB_DIR / "app" / "build.py"

# Docker: stop (not down) the ollama support service — this preserves the
# container and the ollama-models volume. We NEVER run `down -v`.
OLLAMA_STOP = ["docker", "compose", "--profile", docker_llm.COMPOSE_PROFILE, "stop", "ollama"]


# --- configuration (env / .env, same source as the other scripts) ----------------


def _config_bool(name: str, fallback: bool) -> bool:
    raw = llm_smoke.config_value(name, "true" if fallback else "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def api_host() -> str:
    return llm_smoke.config_value("API_HOST", "127.0.0.1")


def api_port() -> int:
    try:
        return int(llm_smoke.config_value("API_PORT", "8000"))
    except ValueError:
        return 8000


def api_url() -> str:
    return f"http://{api_host()}:{api_port()}"


def ollama_url() -> str:
    return llm_smoke.config_value("OLLAMA_BASE_URL", docker_llm.OLLAMA_URL)


def configured_model() -> str:
    return llm_smoke.config_value("LLM_PLANNER_MODEL", llm_smoke.DEFAULT_MODEL)


def auto_start_ollama() -> bool:
    return _config_bool("RUNTIME_AUTO_START_OLLAMA", True)


def require_gpu() -> bool:
    return _config_bool("RUNTIME_REQUIRE_GPU", True)


def keep_ollama_running() -> bool:
    return _config_bool("RUNTIME_KEEP_OLLAMA_RUNNING", True)


def prewarm_llm_enabled() -> bool:
    return _config_bool("RUNTIME_PREWARM_LLM", True)


def prewarm_stt_enabled() -> bool:
    return _config_bool("RUNTIME_PREWARM_STT", True)


def voice_enabled() -> bool:
    return _config_bool("ENABLE_VOICE", False)


def voice_lab_url() -> str:
    return llm_smoke.config_value("VOICE_LAB_URL", "http://127.0.0.1:8766")


def auto_start_voice_lab() -> bool:
    """Opt-in only (Phase 3D.0.1): the Voice Lab stays optional by default."""
    return _config_bool("RUNTIME_AUTO_START_VOICE_LAB", False)


def prewarm_tts() -> bool:
    return _config_bool("RUNTIME_PREWARM_TTS", True)


def voice_lab_python() -> str:
    """The worker's OWN interpreter (voice_lab/.venv) — isolation by default.

    Falls back to this interpreter only so `voice-start` can explain what is
    missing; setup creates the venv:  python voice_lab/scripts/setup.py
    """
    venv_python = VOICE_LAB_DIR / ".venv" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    return str(venv_python) if venv_python.exists() else sys.executable


# --- thin wrappers over docker_llm / llm_smoke (patch points for tests) -----------


def docker_available() -> bool:
    return docker_llm.docker_available()


def start_ollama() -> bool:
    return docker_llm.start_ollama()


def wait_for_ollama(base_url: str) -> bool:
    return docker_llm.wait_for_ollama(base_url)


def ollama_container_id() -> str | None:
    return docker_llm.get_ollama_container_id()


def detect_gpu(container_id: str) -> tuple[bool, str]:
    return docker_llm.detect_gpu(container_id)


def ensure_model(base_url: str, model: str) -> tuple[bool, str]:
    return llm_smoke.ensure_model(base_url, model)


def warm_llm(base_url: str, model: str) -> bool:
    return prewarm_runtime.warm_llm(base_url, model)


def prewarm_stt(base_url: str) -> bool:
    return prewarm_runtime.prewarm_stt(base_url)


def llm_model_status(base_url: str, model: str) -> bool:
    return prewarm_runtime.model_status(base_url, model)


def unload_llm(base_url: str, model: str) -> bool:
    return prewarm_runtime.unload_model(base_url, model)


def llm_loaded_info() -> dict | None:
    """Normalized /api/ps entry for the configured model, or None. Never raises."""
    try:
        entry = prewarm_runtime.find_loaded(
            prewarm_runtime.loaded_models(ollama_url()), configured_model()
        )
    except Exception:
        return None
    return prewarm_runtime.describe_loaded(entry) if entry else None


def voice_status(base_url: str) -> dict | None:
    """Lightweight STT warm state from the API — never triggers a model load."""
    return llm_smoke.get_json(f"{base_url}/voice/status", timeout=5.0)


def voice_lab_health() -> dict | None:
    # Generous timeout: heavy GPU inference can make the worker's event loop
    # sluggish for seconds — slow is not down (a false "down" here caused a
    # doomed double-start against the already-bound port).
    return llm_smoke.get_json(f"{voice_lab_url()}/health", timeout=15.0)


def voice_lab_worker_status() -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/status", timeout=3.0)


def wait_for_voice_lab(timeout_seconds: float = 60.0) -> dict | None:
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        health = voice_lab_health()
        if health:
            return health
        time.sleep(1.0)
    return None


def wait_for_voice_lab_or_exit(process, timeout_seconds: float = 60.0):
    """(health, exit_code): poll /health but abort EARLY if the process exits.

    A worker that dies during startup/model import must be reported with its
    exit code — not silently retried against a dead port."""
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll() if hasattr(process, "poll") else None
        if exit_code is not None:
            return None, exit_code
        health = voice_lab_health()
        if health:
            return health, None
        time.sleep(1.0)
    return None, None


def voice_lab_warm(profile: str | None = None) -> dict | None:
    """POST /warm — ASYNC since 3D.0.4: returns a 202 job or an immediate
    ready result. Callers poll /jobs/{id} for progress."""
    import httpx

    try:
        response = httpx.post(
            f"{voice_lab_url()}/warm", json={"profile": profile or ""}, timeout=30.0
        )
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def voice_lab_job(job_id: str) -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/jobs/{job_id}", timeout=10.0)


def voice_lab_resources() -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/resources/status", timeout=20.0)


def warm_timeout_seconds() -> float:
    try:
        return float(llm_smoke.config_value("VOICE_LAB_WARM_TIMEOUT_SECONDS", "300"))
    except ValueError:
        return 300.0


def poll_voice_lab_job(job_id: str, timeout_seconds: float, out=print) -> dict | None:
    """Poll one job, printing each REAL stage change.

    Returns the terminal job, or {"_client_timeout": True, **last_seen} when
    the client deadline expires (callers report the real last status), or
    None when the job was never observable at all."""
    import time

    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    last_stage = None
    last_seen: dict | None = None
    while time.monotonic() < deadline:
        job = voice_lab_job(job_id)
        if job is None:
            time.sleep(1.0)
            continue
        last_seen = job
        stage = f"{job.get('status')} — {job.get('message') or ''}".strip(" —")
        if stage != last_stage:
            out(f"  [{job.get('elapsed_seconds', '?')}s] {stage}")
            last_stage = stage
        if job.get("status") in ("completed", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(1.0)
    if last_seen is not None:
        return {"_client_timeout": True, "_waited_seconds": round(time.monotonic() - started, 1),
                **last_seen}
    return None


def read_voice_lab_pid() -> int | None:
    record = read_voice_lab_pid_record()
    return record.get("pid") if record else None


def read_voice_lab_pid_record() -> dict | None:
    """The owned-worker PID record: {"pid", "created_utc", "build"} (or legacy int)."""
    if not VOICE_LAB_PID_FILE.exists():
        return None
    try:
        raw = VOICE_LAB_PID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        record = None
    if isinstance(record, dict) and isinstance(record.get("pid"), int):
        return record
    if isinstance(record, int):
        return {"pid": record}  # legacy plain-int PID file
    return None


def write_voice_lab_pid(pid: int) -> None:
    """Atomic PID-file write: temp file in the same directory + os.replace."""
    import tempfile
    from datetime import datetime, timezone

    ensure_dirs()
    payload = json.dumps({
        "pid": pid,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build": voice_lab_expected_build(),
    })
    fd, temp_name = tempfile.mkstemp(
        prefix=".voice_lab-", suffix=".pid.tmp", dir=VOICE_LAB_PID_FILE.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temp_name, VOICE_LAB_PID_FILE)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def clear_voice_lab_pid() -> None:
    VOICE_LAB_PID_FILE.unlink(missing_ok=True)


def voice_lab_disk_fingerprint() -> str:
    """Hash of the Voice Lab source ON DISK — must mirror app.build.source_fingerprint
    (same files, same order) so a running worker's startup fingerprint can be
    compared against the current code without importing voice_lab's app."""
    import hashlib

    app_dir = VOICE_LAB_DIR / "app"
    digest = hashlib.sha256()
    files = sorted(app_dir.rglob("*.py")) + [app_dir / "static" / "designer.html"]
    for path in files:
        if "__pycache__" in path.parts or not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def voice_lab_expected_build() -> str:
    """The build id of the Voice Lab code ON DISK (parsed, not imported —
    voice_lab's `app` package must never be imported into this process)."""
    import re

    try:
        text = VOICE_LAB_BUILD_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"^API_BUILD\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return match.group(1) if match else ""


def voice_lab_worker_info() -> dict | None:
    """Identity of whatever currently answers on the Voice Lab port."""
    return llm_smoke.get_json(f"{voice_lab_url()}/worker/info", timeout=10.0)


def voice_lab_ready() -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/ready", timeout=15.0)


def voice_lab_profiles() -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/profiles", timeout=10.0)


def voice_lab_jobs_active() -> dict | None:
    return llm_smoke.get_json(f"{voice_lab_url()}/jobs/active", timeout=10.0)


def voice_lab_port() -> int:
    try:
        return int(voice_lab_url().rsplit(":", 1)[-1].strip("/"))
    except ValueError:
        return 8766


def port_owner_pid(port: int) -> int | None:
    """PID of the process LISTENING on 127.0.0.1:<port>, or None. Best-effort."""
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
            try:
                return int(parts[4])
            except ValueError:
                continue
    return None


def process_parent_pid(pid: int) -> int | None:
    """Parent PID of a process, or None. Windows venv launchers re-exec the
    base interpreter as a CHILD, so the port owner can legitimately be a child
    of the PID we recorded."""
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').ParentProcessId"],
            capture_output=True, text=True, timeout=15,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def port_owned_by_us(owner: int | None, recorded: int | None) -> bool:
    """Does our PID file prove ownership of the port owner (self or child)?"""
    if owner is None or recorded is None:
        return False
    if owner == recorded:
        return True
    return process_parent_pid(owner) == recorded


def stop_owned_worker(out=print) -> bool:
    """Terminate the recorded worker AND its proven child (the real listener).

    Returns True when something was stopped. Never touches unrelated PIDs.
    """
    pid = read_voice_lab_pid()
    if pid is None:
        return False
    stopped = False
    owner = port_owner_pid(voice_lab_port())
    if process_alive(pid):
        terminate_process(pid)
        out(f"Voice Lab       : stopped owned worker (pid {pid}).")
        stopped = True
    if owner is not None and owner != pid and process_parent_pid(owner) == pid:
        terminate_process(owner)  # the venv launcher's child holds the socket
        out(f"Voice Lab       : stopped owned worker child (pid {owner}).")
        stopped = True
    clear_voice_lab_pid()
    return stopped


def start_voice_lab_process() -> subprocess.Popen | None:
    """Launch the Voice Lab worker with ITS OWN interpreter, loopback only.

    Only voice_lab/.venv may run the worker (isolation rule) — a missing venv
    is a refusal, never a silent fallback to the main interpreter. The
    launcher (run_worker.py) pins voice_lab/ on sys.path itself, so the
    worker resolves the same code and profile store from ANY CWD.
    """
    python = voice_lab_python()
    if not python.startswith(str(VOICE_LAB_DIR)):
        print(
            "Voice Lab venv missing — refusing to start the worker with the main\n"
            "interpreter. Set it up once with:  python voice_lab/scripts/setup.py"
        )
        return None
    ensure_dirs()
    VOICE_LAB_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(VOICE_LAB_LOG_FILE, "ab")
    process = subprocess.Popen(
        [python, str(VOICE_LAB_RUNNER)],
        cwd=VOICE_LAB_DIR,
        stdout=log_handle,
        stderr=log_handle,
    )
    write_voice_lab_pid(process.pid)
    return process


def api_health(base_url: str) -> dict | None:
    return llm_smoke.get_json(f"{base_url}/health", timeout=2.0)


def get_identity(base_url: str) -> dict | None:
    return llm_smoke.get_json(f"{base_url}/identity", timeout=3.0)


def api_running(base_url: str) -> bool:
    return api_health(base_url) is not None


def wait_for_api(base_url: str) -> dict | None:
    return llm_smoke.wait_for_api(base_url)


def ollama_container_status() -> str:
    """Human-readable state of the ollama container: running / exited / not created."""
    container_id = ollama_container_id()
    if not container_id:
        return "not created"
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "unknown"


# --- PID / process handling ------------------------------------------------------


def ensure_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_pid(pid: int) -> None:
    ensure_dirs()
    API_PID_FILE.write_text(str(pid), encoding="utf-8")


def read_pid() -> int | None:
    if not API_PID_FILE.exists():
        return None
    try:
        return int(API_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def clear_pid() -> None:
    API_PID_FILE.unlink(missing_ok=True)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:  # never let a liveness probe raise
            return False
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process(pid: int) -> None:
    # os.kill on Windows maps a plain signal to TerminateProcess; never use
    # signal 0 there (it would still terminate). We only reach here for a live pid.
    try:
        os.kill(pid, 9 if sys.platform == "win32" else 15)
    except OSError:
        pass


def start_api_process(host: str, port: int) -> subprocess.Popen:
    """Launch uvicorn on the Windows HOST (not Docker), logging to storage/logs.

    Configuration (planner, response generator, real tools, etc.) is inherited
    from .env — this manager deliberately does not override safety-relevant
    flags. It never sets ENABLE_REAL_WINDOWS_TOOLS.
    """
    ensure_dirs()
    log_handle = open(API_LOG_FILE, "ab")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=log_handle,
    )
    write_pid(process.pid)
    return process


# --- commands --------------------------------------------------------------------


def _start_dockerized_ollama() -> tuple[bool, str]:
    """Bring up GPU-first Ollama. Returns (ok, message)."""
    if not docker_available():
        return False, (
            "Docker is not available. Install/start Docker Desktop, or run Ollama\n"
            "natively and start the API with:  python scripts/local_runtime.py start --no-ollama"
        )
    if not start_ollama():
        return False, "docker compose failed to start the ollama service.\n\n" + docker_llm.GPU_HELP
    if not wait_for_ollama(ollama_url()):
        return False, (
            f"Ollama did not answer at {ollama_url()} in time.\n"
            "  docker compose --profile llm logs ollama"
        )

    container_id = ollama_container_id()
    if container_id:
        gpu_detected, gpu_detail = detect_gpu(container_id)
    else:
        gpu_detected, gpu_detail = False, "could not resolve the ollama container id"
    print(f"GPU detected    : {'yes' if gpu_detected else 'NO'}")
    print(f"GPU info        : {gpu_detail}")

    cpu_allowed = docker_llm.cpu_allowed() or not require_gpu()
    if not gpu_detected:
        if cpu_allowed:
            print(
                "WARNING: continuing with CPU-only inference "
                "(ALLOW_CPU_OLLAMA/RUNTIME_REQUIRE_GPU opt-in). Expect slow LLM calls."
            )
        else:
            return False, docker_llm.GPU_HELP

    model = configured_model()
    ok, detail = ensure_model(ollama_url(), model)
    if not ok:
        return False, f"model not ready: {detail}"
    print(f"Model           : {detail}")

    # Persistent warm runtime (Phase 3D.1): preload the model into VRAM with
    # keep_alive=-1 and verify it is GPU-resident. A CPU-offloaded model fails
    # here unless explicitly permitted (warm_llm enforces that).
    if prewarm_llm_enabled():
        if not warm_llm(ollama_url(), model):
            return False, "LLM warmup failed (see above)."
    else:
        print("LLM prewarm     : skipped (RUNTIME_PREWARM_LLM=false)")
    return True, "ollama ready"


def cmd_start(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: start ===")
    print(f"API             : {api_url()} (Windows host)")
    print(f"Ollama          : {ollama_url()} (Docker, profile {docker_llm.COMPOSE_PROFILE})")

    # 1. Dockerized Ollama (support service).
    if auto_start_ollama() and not args.no_ollama:
        ok, message = _start_dockerized_ollama()
        if not ok:
            print(message)
            return 1
        print("Ollama is up.")
    else:
        print("Skipping Ollama startup (disabled by flag/config).")

    # 2. FastAPI server on the Windows host. Validate via /health, not just PID.
    if api_running(api_url()):
        print("API already running — leaving it in place.")
    else:
        pid = read_pid()
        # We only ever act on the PID WE recorded — never a process we don't own.
        if pid is not None and not process_alive(pid):
            print(f"Owned API process (pid {pid}) has died — restarting it once.")
            clear_pid()
        elif pid is not None and process_alive(pid):
            # Owned but /health is silent (hung/starting). Recycle it once.
            print(f"Owned API pid {pid} is alive but /health is silent — restarting once.")
            terminate_process(pid)
            clear_pid()
        print("Starting API on the Windows host (simulated tools unless .env enables them)...")
        start_api_process(api_host(), api_port())
        if wait_for_api(api_url()) is None:
            print("API did not come up — check storage/logs/api.log")
            return 1

    # 3. Verify /health.
    health = api_health(api_url())
    if health is None:
        print("API /health did not respond.")
        return 1
    print(f"Health          : ok (version={health.get('version')})")

    # 4. Verify /identity.
    identity = get_identity(api_url())
    if not identity or identity.get("agent_name") != "Fifi":
        print(f"Identity check failed: {json.dumps(identity)}")
        return 1
    print(f"Identity        : {identity.get('agent_name')} ready")

    # 5. Prewarm STT: load faster-whisper (cuda/float16) before Fifi accepts
    # voice commands. A failure is reported loudly but NEVER fails the start —
    # /health is unaffected by STT problems, by design.
    if voice_enabled() and prewarm_stt_enabled():
        if not prewarm_stt(api_url()):
            print(
                "WARNING: STT prewarm failed — voice commands may be cold or broken. "
                "The API stays up; /health is unaffected."
            )

    # 6. Voice Lab (Phase 3D.0.1) — OPTIONAL, opt-in via
    # RUNTIME_AUTO_START_VOICE_LAB=true. A Voice Lab problem is a warning,
    # never a start failure: Fifi falls back to Windows TTS.
    if auto_start_voice_lab():
        if _voice_lab_up():
            if prewarm_tts():
                warm = voice_lab_warm()
                if warm and warm.get("status") == "ready":
                    print(f"TTS warm        : ready ({warm.get('engine')}, {warm.get('profile')})")
                elif warm and warm.get("job_id"):
                    print(f"TTS warm        : warming in the background (job {warm['job_id']})")
                else:
                    print("WARNING: TTS warm failed — first spoken reply may be slow.")
        else:
            print("WARNING: Voice Lab did not start — Fifi will use Windows TTS fallback.")
    print("Runtime is up. Fifi is listening on the host API.")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: stop ===")
    pid = read_pid()
    if pid is None:
        print("Host API        : no PID recorded by this script — nothing to stop.")
    elif not process_alive(pid):
        print(f"Host API        : recorded pid {pid} is not running — clearing stale PID.")
        clear_pid()
    else:
        terminate_process(pid)
        clear_pid()
        print(f"Host API        : stopped (pid {pid}).")

    stop_ollama = args.stop_ollama or not keep_ollama_running()
    if stop_ollama:
        print("Ollama          : stopping the Docker service (models are preserved)...")
        if stop_ollama_service():
            print("Ollama          : stopped. Models remain in the ollama-models volume.")
        else:
            print("Ollama          : stop command failed (see output above).")
    else:
        print("Ollama          : left running (persistent). Use --stop-ollama to stop it.")
    print("Models and Docker volumes were not deleted.")
    return 0


def stop_ollama_service() -> bool:
    return subprocess.run(OLLAMA_STOP, cwd=PROJECT_ROOT).returncode == 0


def cmd_restart(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: restart ===")
    cmd_stop(args)
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: status ===")
    docker_ok = docker_available()
    print(f"Docker available: {docker_ok}")

    if docker_ok:
        print(f"Ollama container: {ollama_container_status()}")
        container_id = ollama_container_id()
        if container_id:
            gpu_detected, gpu_detail = detect_gpu(container_id)
        else:
            gpu_detected, gpu_detail = False, "container not running"
    else:
        gpu_detected, gpu_detail = False, "docker unavailable"
        print("Ollama container: docker unavailable")
    print(f"GPU detected    : {gpu_detected}")
    print(f"GPU name        : {gpu_detail}")

    model = configured_model()
    print(f"Configured model: {model}")
    model_ok = llm_smoke.model_present(ollama_url(), model) if docker_ok else False
    print(f"Model available : {model_ok}")

    # Warm state (Phase 3D.1) — read-only /api/ps lookup, never loads anything.
    info = llm_loaded_info() if docker_ok else None
    if info:
        print(f"LLM warm        : True ({info['processor']}, {info['vram_gb']} GB VRAM)")
    else:
        print("LLM warm        : False (model not loaded — run 'warm')")

    running = api_running(api_url())  # validated via /health, not just PID presence
    print(f"API running     : {running}")
    pid = read_pid()
    if pid is not None and not running:
        state = "died" if not process_alive(pid) else "alive but /health silent"
        print(f"Owned API PID   : {pid} ({state}) — run 'start' to recover")
    identity = get_identity(api_url()) if running else None
    print(f"Fifi identity   : {json.dumps(identity) if identity else '(API not running)'}")

    voice_on = voice_enabled()
    ptt_enabled = _config_bool("ENABLE_PUSH_TO_TALK", False)
    ptt_available, ptt_missing = fifi_ptt.check_desktop_deps()
    print(f"Voice enabled   : {voice_on}")

    # STT warm state via the API's lightweight /voice/status (no model load).
    if running and voice_on:
        stt = voice_status(api_url())
        if stt and stt.get("status") == "ok":
            print(
                f"STT warm        : {stt.get('loaded')} "
                f"(device={stt.get('device')} compute={stt.get('compute_type')} "
                f"model={stt.get('model')})"
            )
        else:
            print("STT warm        : unknown (/voice/status unavailable)")
    else:
        print("STT warm        : (voice disabled or API not running)")

    ptt_detail = "" if ptt_available else f" (missing: {', '.join(ptt_missing)})"
    print(f"Push-to-talk    : available={ptt_available} enabled={ptt_enabled}{ptt_detail}")

    # Wake word (Phase 3D.1) — never auto-started; `wake` launches it explicitly.
    wake_deps_ok, wake_missing = fifi_wake.check_wake_deps()
    wake_detail = "" if wake_deps_ok else f" (missing: {', '.join(wake_missing)})"
    print(
        f"Wake word       : enabled={_config_bool('ENABLE_WAKE_WORD', False)} "
        f"deps={wake_deps_ok}{wake_detail}"
    )
    print(
        f"Wake model      : present={fifi_wake.wake_model_present()} "
        f"({fifi_wake.wake_model_path()})"
    )
    print(f"Wake active     : {fifi_wake.wake_active()}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: smoke ===")
    return llm_smoke.main(["--ollama-url", ollama_url(), "--model", configured_model()])


def cmd_ptt(args: argparse.Namespace) -> int:
    """Launch the Windows-host push-to-talk client against the running API.

    Not started automatically by `start` — push-to-talk is always explicit.
    """
    print("=== Fifi local runtime: push-to-talk ===")
    return fifi_ptt.run(["--server", api_url()])


def cmd_warm(args: argparse.Namespace) -> int:
    """(Re)load the LLM into VRAM (keep_alive=-1) and prewarm STT."""
    print("=== Fifi local runtime: warm ===")
    ok = warm_llm(ollama_url(), configured_model())
    if voice_enabled() and prewarm_stt_enabled():
        ok = prewarm_stt(api_url()) and ok
    return 0 if ok else 1


def cmd_model_status(args: argparse.Namespace) -> int:
    """Report whether the configured model is loaded and where. Read-only."""
    print("=== Fifi local runtime: model status ===")
    return 0 if llm_model_status(ollama_url(), configured_model()) else 1


def cmd_unload(args: argparse.Namespace) -> int:
    """Release the model from VRAM (keep_alive=0). Deletes nothing, ever."""
    print("=== Fifi local runtime: unload ===")
    return 0 if unload_llm(ollama_url(), configured_model()) else 1


def _stop_owned_stale_worker(out=print) -> bool:
    """Stop the port owner ONLY when our PID file proves we own it.

    Returns True when the port should now be free. An unrelated process is
    NEVER terminated — we refuse and tell the user instead.
    """
    port = voice_lab_port()
    owner = port_owner_pid(port)
    pid = read_voice_lab_pid()
    if owner is None:
        # Nothing (visibly) listening; clear a dead recorded pid if any.
        if pid is not None and not process_alive(pid):
            out(f"Voice Lab       : clearing stale PID file (pid {pid} is gone).")
            clear_voice_lab_pid()
        return True
    if port_owned_by_us(owner, pid):
        out(f"Voice Lab       : stopping OUR stale worker (pid {pid}, proven by PID file).")
        terminate_process(pid)
        if owner != pid:
            terminate_process(owner)  # proven child — it holds the socket
        clear_voice_lab_pid()
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and port_owner_pid(port) == owner:
            time.sleep(0.5)
        return port_owner_pid(port) != owner
    out(
        f"Voice Lab       : port {port} is owned by pid {owner}, which this script "
        "did NOT start — refusing to touch it. Stop that process yourself, or "
        "change VOICE_LAB_URL."
    )
    return False


def _voice_lab_up(out=print) -> bool:
    """Ensure a CURRENT worker is running (voice-start / voice-warm / start).

    BUSY IS NOT DOWN: heavy GPU work can make /health slow. If the probe fails
    but the process WE own is still alive, we never start a duplicate (the
    port is bound; a second uvicorn would just die) — we report busy instead.
    Only a PID recorded by this script is ever considered ours.

    STALE IS NOT CURRENT (Phase 3D.0.3a): a healthy worker must also match the
    build on disk. An owned stale worker is recycled; an unowned one is
    refused, never killed.
    """
    health = voice_lab_health()
    if health:
        expected = voice_lab_expected_build()
        running = health.get("api_build") or ""
        if not running:  # pre-3D.0.3a /health has no build — ask /worker/info
            running = (voice_lab_worker_info() or {}).get("api_build") or ""
        if not expected or running == expected:
            out(f"Voice Lab       : already running at {voice_lab_url()} (build {running or '?'})")
            return True
        out(
            f"Voice Lab       : the running worker is STALE (build {running or 'unknown'}, "
            f"disk has {expected}) — the UI would not match its API."
        )
        if not _stop_owned_stale_worker(out):
            return False
    else:
        pid = read_voice_lab_pid()
        if pid is not None and process_alive(pid):
            out(
                f"Voice Lab       : worker (pid {pid}) is alive but slow to answer — "
                "probably busy with a GPU job. NOT starting a duplicate; try again shortly."
            )
            return True
        if not _stop_owned_stale_worker(out):
            return False
    if not (VOICE_LAB_DIR / ".venv").exists():
        out(
            "Voice Lab venv missing — its dependencies are ISOLATED from the main\n"
            "environment. Set it up once with:  python voice_lab/scripts/setup.py"
        )
        return False
    # Watchdog (Phase 3D.0.3a hotfix): verify the worker STAYS alive after
    # startup. If our own worker exits (e.g. a native crash during model
    # import), report the exit code and the last log lines — and restart it at
    # most ONCE, never in a loop. Unrelated processes are never touched.
    for attempt in (1, 2):
        out(f"Starting worker : {voice_lab_url()} (isolated venv, loopback only)")
        process = start_voice_lab_process()
        if process is None:
            return False
        health, exit_code = wait_for_voice_lab_or_exit(process)
        if health is not None:
            out(f"Engines         : {health.get('engines_available')}")
            out(f"Active profile  : {health.get('active_profile')}")
            return True
        if exit_code is not None:
            out(f"Voice Lab worker EXITED during startup (exit code {exit_code}).")
        else:
            out("Voice Lab did not answer /health in time.")
        for line in _tail_worker_log(8):
            out(f"  log: {line}")
        clear_voice_lab_pid()
        if attempt == 1 and exit_code is not None:
            out("Restarting the owned worker once (max one automatic restart)...")
            continue
        out("NOT restarting again — fix the cause and run voice-start manually.")
        return False
    return False


def cmd_voice_start(args: argparse.Namespace) -> int:
    """Start the isolated Voice Lab worker (not started by `start` unless
    RUNTIME_AUTO_START_VOICE_LAB=true — the Voice Lab is optional)."""
    print("=== Fifi local runtime: voice lab start ===")
    if not _voice_lab_up():
        return 1
    print("Voice Lab is up. Try:  python voice_lab/scripts/voice_lab.py compare")
    return 0


def cmd_voice_warm(args: argparse.Namespace) -> int:
    """Asynchronous warmup (Phase 3D.0.4): submit, poll real stages, exit 0
    ONLY when ready. FAILED output always pairs with a nonzero exit code."""
    print("=== Fifi local runtime: voice lab warm ===")
    if not _voice_lab_up():
        return 1
    submitted = voice_lab_warm(getattr(args, "profile", None))
    if submitted is None:
        print("Warm FAILED     : the worker did not answer /warm.")
        return 1
    if submitted.get("status") == "error":
        print(f"Warm FAILED     : {submitted.get('user_message') or submitted.get('message')}")
        return 1
    if submitted.get("status") == "ready":
        # Idempotent path — possibly a load that finished after a timeout: the
        # engine is loaded NOW even though the recorded warm job ended in a
        # (client or no-progress) timeout.
        last = submitted.get("last_warm_job") or {}
        if last.get("status") == "completed" or last.get("error_code") == "model_load_timeout":
            print("La solicitud agotó el tiempo de espera, pero el motor terminó de cargar.")
        print(
            f"Warm READY      : profile={submitted.get('profile')} "
            f"engine={submitted.get('engine')} (ya cargado)"
        )
        return 0
    job_id = submitted.get("job_id")
    if not job_id:
        print(f"Warm FAILED     : unexpected response: {json.dumps(submitted)[:200]}")
        return 1
    print(f"Warm job        : {job_id} (model_cache={submitted.get('model_cache')})")
    final = poll_voice_lab_job(job_id, warm_timeout_seconds())
    if final is None or final.get("_client_timeout"):
        # Client-side timeout — reconcile ONCE with actual engine state before
        # deciding: the engine may have finished loading after the deadline.
        health = voice_lab_health()
        late = voice_lab_job(job_id)
        if late and late.get("status") == "completed":
            print("La solicitud agotó el tiempo de espera, pero el motor terminó de cargar.")
            return 0
        print(
            f"Warm TIMED OUT  : job {job_id} still {latest_status(late)} after "
            f"{(final or {}).get('_waited_seconds', warm_timeout_seconds())}s "
            f"(worker {'alive' if health else 'SILENT'}). "
            "No se inicia una carga duplicada — vuelve a ejecutar voice-warm."
        )
        return 2
    if final.get("status") == "completed":
        result = final.get("result") or {}
        print(
            f"Warm READY      : profile={result.get('profile')} "
            f"engine={result.get('engine')} "
            f"synthesis={result.get('seconds')}s "
            f"fallback_used={result.get('fallback_used')}"
        )
        return 0
    print(
        f"Warm FAILED     : status={final.get('status')} "
        f"[{final.get('error_code')}] {final.get('error')}"
    )
    return 1


def latest_status(job: dict | None) -> str:
    return (job or {}).get("status") or "unknown"


def cmd_voice_preview(args: argparse.Namespace) -> int:
    """Submit the SAME preview job the UI uses and report the playable result."""
    print("=== Fifi local runtime: voice preview ===")
    if not _voice_lab_up():
        return 1
    import httpx

    try:
        response = httpx.post(
            f"{voice_lab_url()}/jobs/profile-preview",
            json={"profile": args.profile, "force": bool(getattr(args, "force", False))},
            timeout=30.0,
        )
        submitted = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Preview FAILED  : no se pudo contactar con el worker ({type(exc).__name__})")
        return 1
    if not submitted.get("job_id"):
        print(
            f"Preview FAILED  : [{submitted.get('error_code')}] "
            f"{submitted.get('user_message') or submitted.get('message')}"
        )
        return 1
    final = poll_voice_lab_job(submitted["job_id"], warm_timeout_seconds())
    if final is not None and final.get("_client_timeout"):
        print(
            f"Preview TIMED OUT after {final.get('_waited_seconds')}s — "
            f"job {submitted['job_id']} sigue en estado {final.get('status')} en el worker."
        )
        return 2
    if final is None or final.get("status") != "completed":
        print(f"Preview FAILED  : status={latest_status(final)} "
              f"[{(final or {}).get('error_code')}] {(final or {}).get('error')}")
        return 1
    result = final.get("result") or {}
    print(f"Profile         : {result.get('requested_profile')}")
    print(f"Engine          : requested={result.get('requested_engine')} "
          f"actual={result.get('actual_engine')} fallback={result.get('fallback_used')}")
    print(f"Audio           : {result.get('audio_url')} "
          f"({result.get('audio_duration_seconds')}s, cache_hit={result.get('cache_hit')})")
    if getattr(args, "play", False):
        try:
            import winsound

            wav = VOICE_LAB_DIR / result.get("file", "")
            if wav.is_file():
                winsound.PlaySound(str(wav), winsound.SND_FILENAME)
                print("Playback        : done")
        except Exception as exc:
            print(f"Playback        : failed ({type(exc).__name__}) — el archivo sigue disponible")
    return 0


def cmd_voice_status(args: argparse.Namespace) -> int:
    print("=== Fifi local runtime: voice lab status ===")
    health = voice_lab_health()
    print(f"Worker          : {'running' if health else 'not running'} ({voice_lab_url()})")
    pid = read_voice_lab_pid()
    if pid is not None and not health:
        state = "died" if not process_alive(pid) else "alive but /health silent"
        print(f"Owned worker PID: {pid} ({state}) — run 'voice-start' to recover")
    if health:
        print(f"Engines         : {health.get('engines_available')}")
        status = voice_lab_worker_status() or {}
        print(f"Engines loaded  : {status.get('engines_loaded') or {}}")
        print(f"Active profile  : {health.get('active_profile')}")
        active = llm_smoke.get_json(f"{voice_lab_url()}/profile/active", timeout=5.0) or {}
        print(f"Active engine   : {active.get('engine') or '(sin seleccionar)'}")
        resources_view = voice_lab_resources() or {}
        if resources_view:
            print(
                f"Resources       : RAM {resources_view.get('system_ram_available_gb')} GB · "
                f"VRAM {resources_view.get('dedicated_vram_free_gb')} GB · "
                f"disk {resources_view.get('disk_free_gb')} GB"
            )
        jobs_view = voice_lab_jobs_active() or {}
        print(f"Active jobs     : {len(jobs_view.get('jobs') or [])}")
    else:
        active = llm_smoke.config_value("VOICE_PROFILE", "fifi_warm")
        print(f"Active profile  : (worker down; main app would use {active!r} + fallback TTS)")
    print(f"Main TTS engine : {llm_smoke.config_value('TTS_ENGINE', 'windows')}")
    return 0


def cmd_voice_stop(args: argparse.Namespace) -> int:
    """Stop the owned Voice Lab worker. Models/caches are never touched."""
    print("=== Fifi local runtime: voice lab stop ===")
    pid = read_voice_lab_pid()
    if pid is None:
        print("Voice Lab       : no PID recorded by this script — nothing to stop.")
    elif not stop_owned_worker():
        print(f"Voice Lab       : recorded pid {pid} is not running — cleared stale PID.")
    print("Models, caches, and profiles were not touched.")
    return 0


def _system_ram_gb() -> dict:
    """{total_gb, free_gb} via stdlib ctypes (Windows) — best-effort, no psutil."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return {}
            return {
                "total_gb": round(stat.ullTotalPhys / 2**30, 1),
                "free_gb": round(stat.ullAvailPhys / 2**30, 1),
            }
        page = os.sysconf("SC_PAGE_SIZE")
        return {
            "total_gb": round(os.sysconf("SC_PHYS_PAGES") * page / 2**30, 1),
            "free_gb": round(os.sysconf("SC_AVPHYS_PAGES") * page / 2**30, 1),
        }
    except Exception:
        return {}


def _vram_gb() -> dict:
    """{total_gb, free_gb} via nvidia-smi — best-effort."""
    import shutil

    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        total, free = result.stdout.strip().splitlines()[0].split(",")
        return {"total_gb": round(float(total) / 1024, 1), "free_gb": round(float(free) / 1024, 1)}
    except Exception:
        return {}


def _voice_model_cache_state() -> dict:
    """Which Voice Lab models are already in the LOCAL HF cache (no network)."""
    hub = VOICE_LAB_DIR / "models" / "hf" / "hub"
    state = {}
    for label, model_id in (
        ("voice_design", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        ("clone", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        ("kokoro", "hexgrad/Kokoro-82M"),
    ):
        slug = "models--" + model_id.replace("/", "--")
        snapshots = hub / slug / "snapshots"
        cached = snapshots.is_dir() and any(
            child.is_dir() and any(child.iterdir()) for child in snapshots.iterdir()
        )
        state[label] = "cached" if cached else "not_installed"
    return state


def _profile_store_state() -> dict:
    """Direct (worker-independent) look at the shared profile store."""
    profiles_dir = PROJECT_ROOT / "config" / "voices" / "profiles"
    active_path = PROJECT_ROOT / "config" / "voices" / "active.json"
    names = sorted(p.stem for p in profiles_dir.glob("*.json")) if profiles_dir.is_dir() else []
    state = {"profile_count": len(names), "active_profile": None, "valid": bool(names), "error": ""}
    if not names:
        state["error"] = "no profiles found in config/voices/profiles"
        return state
    if not active_path.exists():
        return state  # no selection yet — default applies
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        name = active.get("profile")
    except (OSError, ValueError):
        state.update(valid=False, error="active.json is unreadable")
        return state
    state["active_profile"] = name
    if name not in names:
        state.update(valid=False, error=f"active.json points to missing profile {name!r}")
    return state


def _tail_worker_log(lines: int = 12) -> list[str]:
    """Last worker log lines — access/uvicorn logs only; no env, no secrets."""
    try:
        content = VOICE_LAB_LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line.rstrip() for line in content.splitlines()[-lines:]]


def cmd_voice_doctor(args: argparse.Namespace) -> int:
    """One-shot diagnosis of everything a Voice Lab job needs to start.

    Exit code 0 = the system can start a job right now; nonzero otherwise.
    Never loads a model, never exposes secrets or machine-specific paths.
    """
    print("=== Fifi local runtime: voice doctor ===")
    problems: list[str] = []

    port = voice_lab_port()
    owner = port_owner_pid(port)
    pid = read_voice_lab_pid()
    ours = port_owned_by_us(owner, pid)  # owner may be the launcher's child
    ownership = "ours" if ours else ("unowned" if owner is not None else "free")
    print(f"Port {port}       : {'pid ' + str(owner) if owner else 'nothing listening'} ({ownership})")
    print(f"Owned PID file  : {pid if pid is not None else '(none)'}")
    if owner is not None and not ours:
        problems.append(f"port {port} is owned by pid {owner}, which our PID file does not prove ours")

    expected = voice_lab_expected_build()
    health = voice_lab_health()
    info = voice_lab_worker_info() or {}
    running = info.get("api_build") or (health or {}).get("api_build") or ""
    if health is None:
        print("Worker          : NOT RUNNING (or /health silent)")
        problems.append("worker is not answering /health")
    else:
        match = "MATCH" if running == expected else "MISMATCH"
        print(f"Worker build    : running={running or '?'} disk={expected or '?'} ({match})")
        if running != expected:
            problems.append(f"running worker build {running!r} != code on disk {expected!r}")
        run_fp = info.get("source_fingerprint")
        if run_fp:
            disk_fp = voice_lab_disk_fingerprint()
            fp_match = "MATCH" if run_fp == disk_fp else "MISMATCH"
            print(f"Worker source   : running={run_fp} disk={disk_fp} ({fp_match})")
            if run_fp != disk_fp:
                problems.append(
                    "the running worker was started from OLDER source than what is "
                    "on disk (stale worker) — run voice-restart"
                )
        print(f"Worker status   : {health.get('worker')}")
        print(f"Isolated venv   : {info.get('venv_isolated')}")
        if info and info.get("venv_isolated") is False:
            problems.append("worker is NOT running on voice_lab/.venv")

    store = _profile_store_state()
    print(f"Profile store   : valid={store['valid']} profiles={store['profile_count']} "
          f"active={store['active_profile']!r}"
          + (f" — {store['error']}" if store["error"] else ""))
    if not store["valid"]:
        problems.append(f"profile store invalid: {store['error']}")

    ready = voice_lab_ready() if health else None
    if ready is not None:
        print(f"Readiness       : ready={ready.get('ready')} store_ok={ready.get('profile_store', {}).get('valid')} "
              f"env_ok={ready.get('environment_ok')}")
        if not ready.get("ready"):
            problems.append("worker reports not ready")
    jobs_active = voice_lab_jobs_active() if health else None
    if jobs_active is not None:
        print(f"Job queue       : {len(jobs_active.get('jobs') or [])} active "
              f"(worker {jobs_active.get('worker')})")

    cache = _voice_model_cache_state()
    print(f"Model cache     : {cache}")
    ram = _system_ram_gb()
    vram = _vram_gb()
    print(f"System RAM      : {ram.get('free_gb', '?')} GB free of {ram.get('total_gb', '?')} GB")
    print(f"GPU VRAM        : {vram.get('free_gb', '?')} GB free of {vram.get('total_gb', '?')} GB")

    # Phase 3D.0.4: the worker's own canonical resource view + thresholds.
    resources_view = voice_lab_resources() if health else None
    if resources_view:
        print(
            f"Worker resources: RAM {resources_view.get('system_ram_available_gb')} GB · "
            f"VRAM {resources_view.get('dedicated_vram_free_gb')} GB "
            f"({resources_view.get('vram_source')}) · "
            f"disk {resources_view.get('disk_free_gb')} GB"
        )
        thresholds = resources_view.get("thresholds") or {}
        print(
            f"Thresholds      : RAM>={thresholds.get('min_free_system_ram_gb')} GB · "
            f"VRAM>={thresholds.get('min_free_vram_gb')} GB · "
            f"disk>={thresholds.get('min_free_disk_gb')} GB"
        )
        if resources_view.get("blocking"):
            problems.append(f"resource below threshold: {resources_view['blocking']}")
        print(f"Loaded engines  : {resources_view.get('engines_loaded')}")
    previews_dir = VOICE_LAB_DIR / "storage" / "previews" / "auditions"
    preview_count = len(list(previews_dir.glob("*.wav"))) if previews_dir.is_dir() else 0
    print(f"Playable preview: {preview_count} cached audition(s)")
    try:
        previews_dir.mkdir(parents=True, exist_ok=True)
        probe = previews_dir / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print("Preview storage : writable")
    except OSError:
        print("Preview storage : NOT WRITABLE")
        problems.append("preview storage is not writable")

    log_tail = _tail_worker_log()
    if log_tail:
        print("Last log lines  :")
        for line in log_tail:
            print(f"  {line}")

    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print("A Voice Lab job CANNOT start right now.")
        return 1
    print("All checks passed — a Voice Lab job can start.")
    return 0


def cmd_voice_restart(args: argparse.Namespace) -> int:
    """Stop ONLY the owned worker, start the isolated one, verify readiness.

    Never loads a model just to pass health; verifies the active profile and
    the saved-profile count survive the restart.
    """
    print("=== Fifi local runtime: voice lab restart ===")
    stop_owned_worker()
    if not _voice_lab_up():
        return 1
    ready = voice_lab_ready()
    if not ready or not ready.get("ready"):
        print(f"Readiness FAILED: {json.dumps(ready) if ready else 'no response'}")
        return 1
    profiles = voice_lab_profiles() or {}
    count = len(profiles.get("profiles") or [])
    active = profiles.get("active")
    print(f"Readiness       : ready (worker {ready.get('worker')})")
    print(f"Active profile  : {active}")
    print(f"Saved profiles  : {count}")
    if count == 0:
        print("PROBLEM: the shared profile store came back EMPTY after restart.")
        return 1
    return 0


def cmd_voice_ui(args: argparse.Namespace) -> int:
    """Start the Voice Lab worker (if needed) and open the Designer UI."""
    import webbrowser

    print("=== Fifi local runtime: voice designer UI ===")
    if not _voice_lab_up():
        return 1
    url = f"{voice_lab_url()}/ui"
    print(f"Opening         : {url} (local only — the worker binds to loopback)")
    webbrowser.open(url)
    return 0


def cmd_wake(args: argparse.Namespace) -> int:
    """Launch the wake-word listener in the FOREGROUND (Windows host).

    Never started automatically — wake listening is always explicit, and
    push-to-talk remains available independently.
    """
    print("=== Fifi local runtime: wake word ===")
    extra = ["--debug"] if getattr(args, "debug", False) else []
    return fifi_wake.run(["--server", api_url(), *extra])


# Wake listener log (background mode). PID ownership lives in fifi_wake's own
# PID file (storage/runtime/wake.pid), written by the listener when READY.
WAKE_LOG_FILE = LOG_DIR / "wake.log"


def cmd_wake_start(args: argparse.Namespace) -> int:
    """Start the wake listener in the BACKGROUND (owned, logged, single)."""
    print("=== Fifi local runtime: wake start ===")
    if fifi_wake.wake_active():
        print("Wake listener   : already running — not starting a duplicate.")
        return 0
    deps_ok, missing = fifi_wake.check_wake_deps()
    if not deps_ok:
        print(f"Missing deps    : {', '.join(missing)}")
        print("Install them    : pip install -r requirements-wakeword.txt")
        return 1
    if not fifi_wake.wake_model_present():
        print(f"Wake model      : MISSING at {fifi_wake.wake_model_path()}")
        print("It is never downloaded automatically — see models/wake_words/README.md")
        return 1
    ensure_dirs()
    log_handle = open(WAKE_LOG_FILE, "ab")
    process = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "fifi_wake.py"), "--server", api_url()],
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=log_handle,
    )
    import time

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if fifi_wake.wake_active():
            print(f"Wake listener   : running (say 'Fifi'; log: storage/logs/wake.log)")
            return 0
        if process.poll() is not None:
            print(f"Wake listener   : exited during startup (exit code {process.poll()}).")
            try:
                for line in WAKE_LOG_FILE.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-8:]:
                    print(f"  log: {line}")
            except OSError:
                pass
            return 1
        time.sleep(0.5)
    print("Wake listener   : did not become ready in time — check storage/logs/wake.log")
    return 1


def cmd_wake_stop(args: argparse.Namespace) -> int:
    """Stop ONLY the wake listener recorded in its own PID file."""
    print("=== Fifi local runtime: wake stop ===")
    if not fifi_wake.WAKE_PID_FILE.exists():
        print("Wake listener   : no PID recorded — nothing to stop.")
        return 0
    try:
        pid = int(fifi_wake.WAKE_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid = None
    if pid is None or not process_alive(pid):
        print("Wake listener   : recorded process is not running — clearing stale PID.")
    else:
        terminate_process(pid)
        print(f"Wake listener   : stopped (pid {pid}).")
    fifi_wake.WAKE_PID_FILE.unlink(missing_ok=True)
    return 0


def cmd_wake_status(args: argparse.Namespace) -> int:
    """Report the full wake-word picture. Read-only; loads nothing."""
    print("=== Fifi local runtime: wake status ===")
    deps_ok, missing = fifi_wake.check_wake_deps()
    detail = "" if deps_ok else f" (missing: {', '.join(missing)})"
    print(f"Dependencies    : {deps_ok}{detail}")
    print(
        f"Model present   : {fifi_wake.wake_model_present()} "
        f"({fifi_wake.wake_model_path()})"
    )
    print(f"Enabled         : {fifi_wake.wake_word_enabled()} (ENABLE_WAKE_WORD)")
    active = fifi_wake.wake_active()
    print(f"Active          : {active}")
    status = fifi_wake.read_wake_status()
    if active:
        print(f"Muted           : {status.get('muted', False)}")
    print(f"Microphone      : {status.get('input_device') or fifi_wake.WakeConfig.load().input_device}")
    detection = status.get("last_detection")
    if detection:
        score = detection.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        print(f"Last detection  : {detection.get('utc')}{score_text}")
    else:
        print("Last detection  : (none recorded)")
    last = status.get("last_command")
    if last:
        print(
            f"Last command    : {last.get('result')} at {last.get('utc')} "
            f"(stop: {last.get('stop_reason')}, capture {last.get('capture_seconds')}s)"
        )
        if last.get("timings"):
            timings = last["timings"]
            print(
                f"  durations     : stt={timings.get('transcribe_seconds')}s "
                f"cmd={timings.get('command_seconds')}s tts={timings.get('tts_seconds')}s"
            )
    else:
        print("Last command    : (none recorded)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local persistent runtime manager for Fifi.")
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="start Ollama (Docker) + API (host)")
    p_start.add_argument(
        "--no-ollama", action="store_true", help="start only the host API"
    )
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="stop the host API (Ollama stays up by default)")
    p_stop.add_argument(
        "--stop-ollama", action="store_true", help="also stop the Docker Ollama service"
    )
    p_stop.set_defaults(func=cmd_stop)

    p_restart = sub.add_parser("restart", help="stop then start")
    p_restart.add_argument("--no-ollama", action="store_true")
    p_restart.add_argument("--stop-ollama", action="store_true")
    p_restart.set_defaults(func=cmd_restart)

    p_status = sub.add_parser("status", help="print runtime status")
    p_status.set_defaults(func=cmd_status)

    p_smoke = sub.add_parser("smoke", help="run the unattended LLM smoke test")
    p_smoke.set_defaults(func=cmd_smoke)

    p_ptt = sub.add_parser("ptt", help="launch the push-to-talk client (Windows host)")
    p_ptt.set_defaults(func=cmd_ptt)

    p_warm = sub.add_parser("warm", help="load the LLM into VRAM (keep_alive=-1) + prewarm STT")
    p_warm.set_defaults(func=cmd_warm)

    p_model_status = sub.add_parser("model-status", help="report the model's load/VRAM state")
    p_model_status.set_defaults(func=cmd_model_status)

    p_unload = sub.add_parser("unload", help="release VRAM (keep_alive=0; deletes nothing)")
    p_unload.set_defaults(func=cmd_unload)

    p_wake = sub.add_parser("wake", help="launch the wake-word listener (foreground)")
    p_wake.add_argument("--debug", action="store_true", help="print live wake scores")
    p_wake.set_defaults(func=cmd_wake)

    p_wake_start = sub.add_parser("wake-start", help="start the wake listener (background)")
    p_wake_start.set_defaults(func=cmd_wake_start)

    p_wake_stop = sub.add_parser("wake-stop", help="stop the owned wake listener")
    p_wake_stop.set_defaults(func=cmd_wake_stop)

    p_wake_status = sub.add_parser("wake-status", help="wake-word runtime status")
    p_wake_status.set_defaults(func=cmd_wake_status)

    p_voice_start = sub.add_parser(
        "voice-start", help="start the isolated Voice Lab worker (127.0.0.1:8766)"
    )
    p_voice_start.set_defaults(func=cmd_voice_start)

    p_voice_status = sub.add_parser("voice-status", help="Voice Lab worker status")
    p_voice_status.set_defaults(func=cmd_voice_status)

    p_voice_stop = sub.add_parser("voice-stop", help="stop the Voice Lab worker")
    p_voice_stop.set_defaults(func=cmd_voice_stop)

    p_voice_warm = sub.add_parser(
        "voice-warm", help="asynchronously warm the active voice profile"
    )
    p_voice_warm.add_argument("--profile", default=None, help="profile to warm (default: active)")
    p_voice_warm.set_defaults(func=cmd_voice_warm)

    p_voice_preview = sub.add_parser(
        "voice-preview", help="generate + report a playable preview for a profile"
    )
    p_voice_preview.add_argument("profile", help="saved profile name")
    p_voice_preview.add_argument("--force", action="store_true", help="bypass the preview cache")
    p_voice_preview.add_argument("--play", action="store_true", help="play the WAV locally")
    p_voice_preview.set_defaults(func=cmd_voice_preview)

    p_voice_ui = sub.add_parser(
        "voice-ui", help="start the Voice Lab and open the Designer UI in the browser"
    )
    p_voice_ui.set_defaults(func=cmd_voice_ui)

    p_voice_doctor = sub.add_parser(
        "voice-doctor", help="diagnose the Voice Lab (exit 1 if a job cannot start)"
    )
    p_voice_doctor.set_defaults(func=cmd_voice_doctor)

    p_voice_restart = sub.add_parser(
        "voice-restart", help="restart the OWNED Voice Lab worker and verify readiness"
    )
    p_voice_restart.set_defaults(func=cmd_voice_restart)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
