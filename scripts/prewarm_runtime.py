"""Prewarm Fifi's local runtime (Phase 3D.1) — LLM into VRAM, STT onto the GPU.

LLM warmup:
  1. POST /api/generate with an EMPTY prompt and keep_alive=-1 — Ollama loads
     the configured model and keeps it resident until an explicit unload.
  2. Verify through /api/ps that the model actually stayed loaded, and that it
     is FULLY GPU-loaded. Reports model, processor, VRAM usage and load time.
  3. A (partially) CPU-offloaded model FAILS the warmup unless explicitly
     permitted (ALLOW_CPU_OLLAMA=true or OLLAMA_REQUIRE_FULL_GPU=false).

STT prewarm:
  Calls the host API's existing /voice/preflight, which loads faster-whisper
  (cuda/float16 on this machine) and runs a harmless silence probe. An STT
  failure is reported but never stops the API — /health stays healthy.

Unload:
  keep_alive=0 releases the model from VRAM. This script only ever talks to
  the Ollama HTTP API — it can never delete models, volumes, or containers.

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.

Usage:
  python scripts/prewarm_runtime.py                 # warm LLM + STT
  python scripts/prewarm_runtime.py --skip-stt      # warm the LLM only
  python scripts/prewarm_runtime.py --status        # report model load state
  python scripts/prewarm_runtime.py --unload        # release VRAM (keep_alive=0)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import llm_smoke  # noqa: E402  (shared env/.env config reader)

# The first load pulls ~5 GB of weights into VRAM; generous but finite.
LOAD_TIMEOUT_SECONDS = 300.0
STT_PREFLIGHT_TIMEOUT_SECONDS = 300.0


# --- configuration (env / .env, same source as the other scripts) ----------------


def _config_bool(name: str, fallback: bool) -> bool:
    raw = llm_smoke.config_value(name, "true" if fallback else "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def prewarm_llm_enabled() -> bool:
    return _config_bool("RUNTIME_PREWARM_LLM", True)


def prewarm_stt_enabled() -> bool:
    return _config_bool("RUNTIME_PREWARM_STT", True)


def require_full_gpu() -> bool:
    return _config_bool("OLLAMA_REQUIRE_FULL_GPU", True)


def keep_model_loaded() -> bool:
    return _config_bool("OLLAMA_KEEP_MODEL_LOADED", True)


def cpu_offload_allowed() -> bool:
    """CPU-offloaded inference is opt-in only, mirroring docker_llm.cpu_allowed."""
    return _config_bool("ALLOW_CPU_OLLAMA", False) or not require_full_gpu()


def warm_keep_alive() -> int | str:
    """-1 keeps the model resident forever; the fallback mirrors Ollama's default."""
    return -1 if keep_model_loaded() else "5m"


# --- Ollama HTTP helpers (the ONLY way this script touches Ollama) ---------------


def load_model(base_url: str, model: str, keep_alive: int | str) -> dict[str, Any]:
    """Empty-prompt /api/generate: loads the model without generating anything.

    Returns {"seconds": float}. Raises httpx.HTTPError on failure.
    """
    started = time.monotonic()
    response = httpx.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": "", "keep_alive": keep_alive},
        timeout=LOAD_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return {"seconds": time.monotonic() - started}


def loaded_models(base_url: str) -> list[dict[str, Any]]:
    """The models Ollama currently has in memory (/api/ps). Raises on failure."""
    response = httpx.get(f"{base_url}/api/ps", timeout=10.0)
    response.raise_for_status()
    return response.json().get("models", [])


def find_loaded(models: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    for entry in models:
        name = entry.get("name", "")
        if name == model or name.split(":")[0] == model:
            return entry
    return None


def describe_loaded(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize an /api/ps entry: processor placement + VRAM usage."""
    size = int(entry.get("size") or 0)
    vram = int(entry.get("size_vram") or 0)
    if vram <= 0:
        processor = "cpu"
    elif vram >= size:
        processor = "gpu"
    else:
        processor = f"gpu+cpu ({vram * 100 // size}% in VRAM)"
    return {
        "name": entry.get("name"),
        "processor": processor,
        "fully_gpu": size > 0 and vram >= size,
        "size_gb": round(size / 2**30, 2),
        "vram_gb": round(vram / 2**30, 2),
        "expires_at": entry.get("expires_at"),
    }


def unload_model(base_url: str, model: str, out=print) -> bool:
    """Release the model from VRAM with keep_alive=0.

    HTTP-only: the model files, ollama-models volume, and container are never
    touched — a later warm reloads instantly from the same volume.
    """
    out(f"Unloading {model!r} from memory (keep_alive=0; model files are kept)...")
    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        out(f"Unload failed: {exc}")
        return False
    try:
        still = find_loaded(loaded_models(base_url), model)
    except httpx.HTTPError as exc:
        out(f"Unload sent, but /api/ps verification failed: {exc}")
        return False
    if still is not None:
        out(
            "NOTE: the model is still listed by /api/ps — the model-keeper "
            "service re-warms it. Stop Ollama entirely to keep it unloaded:\n"
            "  python scripts/local_runtime.py stop --stop-ollama"
        )
    out(f"Model {model!r} unloaded — VRAM released. Nothing was deleted.")
    return True


# --- LLM warmup -------------------------------------------------------------------


def warm_llm(
    base_url: str,
    model: str,
    keep_alive: int | str | None = None,
    allow_cpu: bool | None = None,
    out=print,
) -> bool:
    """Load the model, pin it in memory, and verify it is GPU-resident."""
    keep_alive = warm_keep_alive() if keep_alive is None else keep_alive
    allow_cpu = cpu_offload_allowed() if allow_cpu is None else allow_cpu

    out(f"Warming LLM     : {model} (keep_alive={keep_alive})")
    try:
        load = load_model(base_url, model, keep_alive)
    except httpx.HTTPError as exc:
        out(f"LLM warmup FAILED: /api/generate error: {exc}")
        return False

    try:
        entry = find_loaded(loaded_models(base_url), model)
    except httpx.HTTPError as exc:
        out(f"LLM warmup FAILED: could not verify via /api/ps: {exc}")
        return False
    if entry is None:
        out(
            f"LLM warmup FAILED: {model!r} is not listed by /api/ps after loading "
            "— it did not stay in memory."
        )
        return False

    info = describe_loaded(entry)
    out(f"Model loaded    : {info['name']}")
    out(f"Processor       : {info['processor']}")
    out(f"VRAM usage      : {info['vram_gb']} GB (model size {info['size_gb']} GB)")
    out(f"Load duration   : {load['seconds']:.1f}s")
    if keep_alive == -1:
        out("Keep-alive      : -1 (stays loaded until an explicit unload)")

    if not info["fully_gpu"]:
        if allow_cpu:
            out(
                "WARNING: the model is not fully GPU-loaded "
                f"({info['processor']}) — continuing because CPU offload is "
                "explicitly permitted. Expect slow inference."
            )
            return True
        out(
            f"LLM warmup FAILED: the model is CPU-offloaded ({info['processor']}).\n"
            "Free VRAM (close other GPU apps) or explicitly permit CPU offload with\n"
            "ALLOW_CPU_OLLAMA=true / OLLAMA_REQUIRE_FULL_GPU=false in .env."
        )
        return False
    return True


def model_status(base_url: str, model: str, out=print) -> bool:
    """Report whether the configured model is loaded, and where. Never loads it."""
    try:
        models = loaded_models(base_url)
    except httpx.HTTPError as exc:
        out(f"Ollama not reachable at {base_url}: {exc}")
        return False
    entry = find_loaded(models, model)
    if entry is None:
        out(f"Model {model!r} is NOT loaded (VRAM free). Warm it with:")
        out("  python scripts/local_runtime.py warm")
        return False
    info = describe_loaded(entry)
    out(f"Model loaded    : {info['name']}")
    out(f"Processor       : {info['processor']}")
    out(f"VRAM usage      : {info['vram_gb']} GB (model size {info['size_gb']} GB)")
    out(f"Expires         : {info['expires_at'] or 'never (keep_alive=-1)'}")
    return True


# --- STT prewarm ------------------------------------------------------------------


def prewarm_stt(api_url: str, out=print) -> bool:
    """Load faster-whisper via the existing /voice/preflight endpoint.

    Returns whether the preflight PASSED. A failure here must never stop the
    API — callers report it and move on (/health is unaffected by design).
    """
    out("Prewarming STT  : GET /voice/preflight (loads faster-whisper)...")
    try:
        response = httpx.get(
            f"{api_url}/voice/preflight", timeout=STT_PREFLIGHT_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        report = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        out(f"STT prewarm FAILED: {exc} (the API stays up; /health is unaffected)")
        return False

    if report.get("status") == "disabled":
        out("STT prewarm     : voice is disabled (ENABLE_VOICE=false) — skipped.")
        return True

    out(
        "STT preflight   : device={device} compute={compute} model={model} "
        "gpu={gpu} passed={passed}".format(
            device=report.get("device"),
            compute=report.get("compute_type"),
            model=report.get("model"),
            gpu=report.get("gpu"),
            passed=report.get("passed"),
        )
    )
    if report.get("cpu_fallback_active"):
        out("STT preflight   : WARNING — CPU fallback is ACTIVE (slow inference).")
    if not report.get("passed"):
        out(
            f"STT prewarm FAILED: {report.get('message')} "
            "(the API stays up; /health is unaffected)"
        )
        return False
    return True


# --- CLI --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm Fifi's LLM (GPU, keep_alive=-1) and STT."
    )
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-stt", action="store_true", help="warm the LLM only")
    parser.add_argument(
        "--status", action="store_true", help="report model load state, change nothing"
    )
    parser.add_argument(
        "--unload", action="store_true", help="release VRAM (keep_alive=0)"
    )
    args = parser.parse_args(argv)

    ollama_url = args.ollama_url or llm_smoke.config_value(
        "OLLAMA_BASE_URL", llm_smoke.DEFAULT_OLLAMA_URL
    )
    api_host = llm_smoke.config_value("API_HOST", "127.0.0.1")
    api_port = llm_smoke.config_value("API_PORT", "8000")
    api_url = args.api_url or f"http://{api_host}:{api_port}"
    model = args.model or llm_smoke.config_value("LLM_PLANNER_MODEL", llm_smoke.DEFAULT_MODEL)

    if args.status:
        return 0 if model_status(ollama_url, model) else 1
    if args.unload:
        return 0 if unload_model(ollama_url, model) else 1

    ok = warm_llm(ollama_url, model)
    if not args.skip_stt and prewarm_stt_enabled():
        ok = prewarm_stt(api_url) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
