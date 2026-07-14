"""GPU resource coordinator — one 16 GB card, many tenants, no surprises.

Tenants on Wilson's RTX 5060 Ti: Ollama (qwen2.5:7b, ~4.4 GB, keep_alive=-1),
faster-whisper in the main API (~1 GB), and the Voice Lab engines (Kokoro
~0.5 GB, Qwen VoiceDesign ~5 GB, Qwen clone/Base ~2 GB).

This module:
- reports the full picture (GPU, VRAM, who has what loaded) with no absolute
  paths and no secrets;
- estimates headroom BEFORE a heavy load and returns a clear, actionable
  message instead of letting CUDA OOM;
- optionally (VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD=true + explicit per-job
  consent) frees VRAM by asking Ollama to unload its model — NEVER by killing
  a process. Note: the model-keeper compose service re-warms Ollama within
  ~60 s, so the freed window is temporary by design.

Every probe is best-effort and never raises.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from app.config import MODELS_DIR

OLLAMA_URL = os.environ.get("VOICE_LAB_OLLAMA_URL", "http://127.0.0.1:11434")
MAIN_API_URL = os.environ.get("VOICE_LAB_MAIN_API_URL", "http://127.0.0.1:8000")

# Conservative planning figures (bf16 weights + activations), in GB.
VRAM_ESTIMATES_GB = {
    "voice_design": 5.0,
    "clone": 2.0,
    "custom_voice": 5.0,
    "kokoro": 0.7,
}
# Keep this much VRAM untouched as a safety margin.
HEADROOM_MARGIN_GB = 1.0


def allow_temporary_ollama_unload() -> bool:
    """Opt-in only (default false): may a job ask Ollama to unload temporarily?"""
    raw = os.environ.get("VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- probes (best-effort, never raise) ------------------------------------------------


def system_ram_snapshot() -> dict[str, Any]:
    """{total_gb, used_gb, free_gb} for SYSTEM RAM, or {} when unprobeable.

    Low system RAM is as dangerous as low VRAM on this machine (16 GB each):
    a model load that swaps freezes the whole desktop, not just the worker.
    Windows: stdlib ctypes (GlobalMemoryStatusEx) — no psutil dependency.
    """
    try:
        import sys

        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return {}
            total = stat.ullTotalPhys / 2**30
            free = stat.ullAvailPhys / 2**30
        else:
            import os as _os

            page = _os.sysconf("SC_PAGE_SIZE")
            total = _os.sysconf("SC_PHYS_PAGES") * page / 2**30
            free = _os.sysconf("SC_AVPHYS_PAGES") * page / 2**30
        return {
            "total_gb": round(total, 1),
            "used_gb": round(total - free, 1),
            "free_gb": round(free, 1),
        }
    except Exception:
        return {}


def memory_pressure() -> dict[str, Any]:
    """Both memories against the configured thresholds — the heavy-job gate.

    {"ram": {...}, "vram": {...}, "ram_ok": bool|None, "vram_ok": bool|None,
     "constrained": "ram"|"vram"|None, "message": <Spanish, actionable>}
    An unprobeable resource is None (not a refusal) — the engine's own OOM
    guard remains the last line of defense.
    """
    from app.config import get_settings

    settings = get_settings()
    ram = system_ram_snapshot()
    vram = gpu_snapshot()
    ram_ok = (ram["free_gb"] >= settings.min_free_system_ram_gb) if ram else None
    vram_ok = (vram.get("free_gb", 0) >= settings.min_free_vram_gb) if vram.get("free_gb") is not None else None
    constrained = "ram" if ram_ok is False else ("vram" if vram_ok is False else None)
    message = ""
    if constrained == "ram":
        message = (
            f"RAM del sistema insuficiente: hay {ram['free_gb']:g} GB libres y el mínimo "
            f"configurado es {settings.min_free_system_ram_gb:g} GB "
            "(VOICE_LAB_MIN_FREE_SYSTEM_RAM_GB). Cierra aplicaciones que usen mucha "
            "memoria e inténtalo de nuevo."
        )
    elif constrained == "vram":
        message = (
            f"VRAM insuficiente: hay {vram['free_gb']:g} GB libres y el mínimo configurado "
            f"es {settings.min_free_vram_gb:g} GB (VOICE_LAB_MIN_FREE_VRAM_GB). Libera "
            "modelos (por ejemplo 'python scripts/local_runtime.py unload') e inténtalo "
            "de nuevo."
        )
    return {
        "ram": ram, "vram": vram, "ram_ok": ram_ok, "vram_ok": vram_ok,
        "constrained": constrained, "message": message,
    }


def pre_job_snapshot() -> dict[str, Any]:
    """Resource picture recorded on every heavy job BEFORE it starts (req. 13)."""
    from app.engines.base import loaded_engines

    qwen_loaded: list[str] = []
    try:
        from app.engines.base import _instances

        qwen = _instances.get("qwen3_tts")
        if qwen is not None:
            qwen_loaded = list(getattr(qwen, "_models", {}))
    except Exception:
        pass
    engines = loaded_engines()
    return {
        "ram": system_ram_snapshot(),
        "gpu": gpu_snapshot(),
        "ollama_loaded": ollama_loaded_models(),
        "whisper": whisper_state(),
        "voice_design_loaded": any("VoiceDesign" in m for m in qwen_loaded),
        "qwen_models_loaded": qwen_loaded,
        "kokoro_loaded": bool(engines.get("kokoro")),
        "engines_loaded": engines,
    }


def gpu_snapshot() -> dict[str, Any]:
    """{name, total_gb, used_gb, free_gb} via nvidia-smi, or {} when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    try:
        return {
            "name": parts[0],
            "total_gb": round(float(parts[1]) / 1024, 1),
            "used_gb": round(float(parts[2]) / 1024, 1),
            "free_gb": round(float(parts[3]) / 1024, 1),
        }
    except (ValueError, IndexError):
        return {"name": parts[0] if parts else "unknown"}


def ollama_loaded_models() -> list[dict[str, Any]]:
    """Models Ollama holds in memory: [{name, vram_gb}] — [] when unreachable."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=3.0)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return []
    return [
        {"name": m.get("name"), "vram_gb": round((m.get("size_vram") or 0) / 2**30, 1)}
        for m in models
    ]


def whisper_state() -> dict[str, Any]:
    """STT warm state from the main API's lightweight /voice/status. Best-effort."""
    try:
        response = httpx.get(f"{MAIN_API_URL}/voice/status", timeout=3.0)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return {"reachable": False, "loaded": False}
    return {
        "reachable": True,
        "loaded": bool(data.get("loaded")),
        "device": data.get("device"),
        "model": data.get("model"),
    }


def _hf_cache_root() -> Path:
    """HF cache root — HF_HOME when set (the Docker volume mount at /models/hf),
    else the isolated voice_lab/models/hf. Lets the containerized worker see the
    seeded volume cache while the host worker keeps its local path."""
    env = os.environ.get("HF_HOME")
    return Path(env) if env else (MODELS_DIR / "hf")


def model_cache_status(model_id: str) -> str:
    """'cached' | 'not_installed' — does the HF cache already hold this model?

    Zero network: a cached model means "Modelo encontrado en caché local; no se
    requiere descarga."
    """
    if not model_id or "/" not in model_id:
        return "cached"  # engine-local ids (kokoro voices, sapi) need no download
    slug = "models--" + model_id.replace("/", "--")
    root = _hf_cache_root()
    for hub in (root / "hub", root):
        snapshot_root = hub / slug / "snapshots"
        if snapshot_root.is_dir():
            for snapshot in snapshot_root.iterdir():
                if snapshot.is_dir() and any(snapshot.iterdir()):
                    return "cached"
    return "not_installed"


def cache_dir_size_bytes() -> int:
    """Total bytes under the HF cache — used to show real download movement."""
    total = 0
    root = _hf_cache_root()
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# --- coordination policy ---------------------------------------------------------------


def headroom_check(required_gb: float, allow_ollama_unload: bool = False) -> dict[str, Any]:
    """Can a load of `required_gb` fit? Returns a decision, never raises.

    {"ok": bool, "free_gb", "required_gb", "message", "can_free_ollama": bool}
    When the GPU cannot be probed the load is allowed (the engine still has its
    own OOM handling as the last line of defense).
    """
    gpu = gpu_snapshot()
    if not gpu or "free_gb" not in gpu:
        return {"ok": True, "message": "GPU not probeable — proceeding with engine-level OOM guard"}
    free = gpu["free_gb"]
    needed = required_gb + HEADROOM_MARGIN_GB
    if free >= needed:
        return {"ok": True, "free_gb": free, "required_gb": required_gb, "message": ""}

    ollama = ollama_loaded_models()
    ollama_gb = sum(m["vram_gb"] for m in ollama)
    decision: dict[str, Any] = {
        "ok": False,
        "free_gb": free,
        "required_gb": required_gb,
        "can_free_ollama": bool(ollama) and allow_temporary_ollama_unload(),
    }
    if ollama and allow_temporary_ollama_unload() and allow_ollama_unload:
        return {**decision, "ok": True, "unload_ollama_first": True, "message": ""}
    hints = ["cierra otras aplicaciones que usen la GPU"]
    if ollama:
        hints.insert(0, (
            f"libera el modelo de Ollama ({ollama_gb:g} GB) con "
            "'python scripts/local_runtime.py unload'"
        ))
        if not allow_temporary_ollama_unload():
            hints.append(
                "o permite la liberación temporal con "
                "VOICE_LAB_ALLOW_TEMPORARY_OLLAMA_UNLOAD=true en voice_lab/.env"
            )
    decision["message"] = (
        f"VRAM insuficiente: se necesitan ~{required_gb:g} GB (+{HEADROOM_MARGIN_GB:g} de margen) "
        f"y hay {free:g} GB libres. Opciones: " + "; ".join(hints) + "."
    )
    return decision


def unload_ollama_temporarily() -> bool:
    """Ask Ollama (HTTP only, never a process kill) to release its model.

    The model-keeper re-warms it within ~60 s, so this is a temporary window —
    callers should start their load immediately after.
    """
    models = ollama_loaded_models()
    ok = True
    for model in models:
        try:
            httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model["name"], "prompt": "", "keep_alive": 0},
                timeout=30.0,
            ).raise_for_status()
        except httpx.HTTPError:
            ok = False
    return ok


def resource_report() -> dict[str, Any]:
    """The full coordination picture for /models/status — no paths, no secrets."""
    from app.engines.base import loaded_engines
    from app.engines.qwen3_tts import CLONE_MODEL_ID, DEFAULT_MODEL_ID

    engines = loaded_engines()
    qwen_models: dict[str, Any] = {}
    try:
        from app.engines.base import _instances  # read-only peek

        qwen = _instances.get("qwen3_tts")
        if qwen is not None:
            qwen_models = {mid: True for mid in getattr(qwen, "_models", {})}
    except Exception:
        pass

    return {
        "gpu": gpu_snapshot(),
        "ram": system_ram_snapshot(),
        "ollama": {"loaded": ollama_loaded_models()},
        "whisper": whisper_state(),
        "voice_lab": {
            "engines_loaded": engines,
            "voice_design": {
                "model": DEFAULT_MODEL_ID,
                "cache": model_cache_status(DEFAULT_MODEL_ID),
                "loaded": DEFAULT_MODEL_ID in qwen_models,
            },
            "clone": {
                "model": CLONE_MODEL_ID,
                "cache": model_cache_status(CLONE_MODEL_ID),
                "loaded": CLONE_MODEL_ID in qwen_models,
            },
            "kokoro": {
                "model": "hexgrad/Kokoro-82M",
                "cache": model_cache_status("hexgrad/Kokoro-82M"),
                "loaded": bool(engines.get("kokoro")),
            },
        },
        "estimates_gb": VRAM_ESTIMATES_GB,
        "allow_temporary_ollama_unload": allow_temporary_ollama_unload(),
    }
