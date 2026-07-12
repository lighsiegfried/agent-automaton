"""Canonical resource inspection for the Voice Lab (Phase 3D.0.4).

ONE service answers every "is there enough X?" question, and it never
confuses resource types:

- DISK is not RAM. SYSTEM RAM is not VRAM. SHARED GPU memory is not
  dedicated VRAM. Task Manager's "GPU Memory" (dedicated+shared) is never
  treated as dedicated VRAM.
- Dedicated VRAM comes from NVML when available (pynvml, or torch's NVML
  binding when torch is already loaded), falling back to `nvidia-smi`
  GLOBAL memory values. Per-process GPU memory is never used — WDDM
  reports N/A for it.
- System RAM uses psutil.virtual_memory().available (ctypes fallback).
- Disk uses shutil.disk_usage() resolved against the volume that actually
  holds the Voice Lab models/storage.

ALL internal comparisons happen in BYTES; GB values exist only for display.
The measured GPU global used/free values are the source of truth — loaded
models are never double-counted on top of them (Ollama's usage is already
inside `nvidia-smi`'s used figure).

Every probe is best-effort and never raises.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.config import MODELS_DIR, get_settings

GiB = 1024**3
MiB = 1024**2


# --- unit conversions (bytes are canonical) --------------------------------------------


def bytes_to_gb(value: int | float | None) -> float | None:
    """Bytes -> GB for DISPLAY only. Comparisons stay in bytes."""
    if value is None:
        return None
    return round(value / GiB, 2)


def mib_to_bytes(value: int | float) -> int:
    """nvidia-smi reports MiB; convert exactly, never compare MiB against GB."""
    return int(value * MiB)


def gb_to_bytes(value: int | float) -> int:
    return int(value * GiB)


# --- probes -----------------------------------------------------------------------------


def system_ram() -> dict[str, Any]:
    """System RAM in bytes: psutil (present in the worker venv) or ctypes."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "used_bytes": int(memory.total - memory.available),
        }
    except Exception:
        pass
    try:
        if sys.platform != "win32":
            return {}
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
            "total_bytes": int(stat.ullTotalPhys),
            "available_bytes": int(stat.ullAvailPhys),
            "used_bytes": int(stat.ullTotalPhys - stat.ullAvailPhys),
        }
    except Exception:
        return {}


def disk() -> dict[str, Any]:
    """Free space on the volume that holds the Voice Lab models/storage.

    shutil.disk_usage resolves the mount/volume of the given path, so this
    measures the drive the model cache actually writes to."""
    try:
        target = MODELS_DIR if MODELS_DIR.exists() else Path(__file__).resolve().parent
        usage = shutil.disk_usage(target)
        return {
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "used_bytes": int(usage.total - usage.free),
        }
    except Exception:
        return {}


def _vram_via_nvml() -> dict[str, Any]:
    """Dedicated VRAM via NVML (pynvml). {} when unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8", "replace")
            return {
                "total_bytes": int(info.total),
                "used_bytes": int(info.used),
                "free_bytes": int(info.free),
                "name": name,
                "driver": driver,
                "source": "nvml",
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return {}


def _vram_via_torch() -> dict[str, Any]:
    """LAST-resort probe only. torch.cuda.mem_get_info goes through the CUDA
    runtime, which under WDDM virtualizes VRAM per process — it reported
    14.8 GB "free" while nvidia-smi showed 7.7 GB physically free on this
    machine. It is therefore ordered BELOW nvidia-smi and used only when
    neither NVML nor nvidia-smi is available."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info(0)
        return {
            "total_bytes": int(total),
            "used_bytes": int(total - free),
            "free_bytes": int(free),
            "name": torch.cuda.get_device_name(0),
            "driver": None,
            "source": "torch_cuda_runtime",
        }
    except Exception:
        return {}


def _vram_via_nvidia_smi() -> dict[str, Any]:
    """GLOBAL dedicated memory from nvidia-smi (MiB -> bytes, exact).

    Global totals only — never per-process figures, which WDDM reports as
    N/A. These values already include every tenant (Ollama, Whisper, us)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
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
            "driver": parts[1],
            "total_bytes": mib_to_bytes(float(parts[2])),
            "used_bytes": mib_to_bytes(float(parts[3])),
            "free_bytes": mib_to_bytes(float(parts[4])),
            "source": "nvidia-smi",
        }
    except (ValueError, IndexError):
        return {}


def dedicated_vram() -> dict[str, Any]:
    """PHYSICAL dedicated VRAM (bytes): NVML, else nvidia-smi global values.

    The WDDM-virtualized CUDA-runtime figure is the very last resort — it is
    NOT physical free VRAM (see _vram_via_torch)."""
    return _vram_via_nvml() or _vram_via_nvidia_smi() or _vram_via_torch()


def shared_gpu_memory() -> dict[str, Any] | None:
    """Shared GPU memory (system RAM windowed to the GPU under WDDM).

    There is no reliable public probe without WMI performance counters; it is
    reported only when torch exposes it. It is NEVER mixed into dedicated
    VRAM figures — Task Manager's combined "GPU memory" is not VRAM."""
    return None  # not reliably measurable here; deliberately separate & absent


def cuda_available() -> bool:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            pass
    return bool(dedicated_vram())


# --- the canonical snapshot --------------------------------------------------------------


def snapshot() -> dict[str, Any]:
    """Everything, in bytes + display GB, clearly separated by resource type."""
    ram = system_ram()
    storage = disk()
    vram = dedicated_vram()

    from app.engines.base import loaded_engines

    try:
        from app import gpu as gpu_module

        ollama = gpu_module.ollama_loaded_models()
        whisper = gpu_module.whisper_state()
    except Exception:
        ollama, whisper = [], {"reachable": False, "loaded": False}

    return {
        "system_ram_total_bytes": ram.get("total_bytes"),
        "system_ram_available_bytes": ram.get("available_bytes"),
        "system_ram_used_bytes": ram.get("used_bytes"),
        "system_ram_available_gb": bytes_to_gb(ram.get("available_bytes")),
        "disk_total_bytes": storage.get("total_bytes"),
        "disk_free_bytes": storage.get("free_bytes"),
        "disk_free_gb": bytes_to_gb(storage.get("free_bytes")),
        "dedicated_vram_total_bytes": vram.get("total_bytes"),
        "dedicated_vram_used_bytes": vram.get("used_bytes"),
        "dedicated_vram_free_bytes": vram.get("free_bytes"),
        "dedicated_vram_free_gb": bytes_to_gb(vram.get("free_bytes")),
        "shared_gpu_memory": shared_gpu_memory(),
        "gpu_name": vram.get("name"),
        "gpu_driver": vram.get("driver"),
        "vram_source": vram.get("source"),
        "cuda_available": cuda_available(),
        "engines_loaded": loaded_engines(),
        "ollama": {"loaded": ollama},
        "whisper": whisper,
    }


# --- policy checks (bytes vs bytes, per-resource Spanish messages) -------------------------


def check_resources() -> dict[str, Any]:
    """Per-resource checks against the configured thresholds — in BYTES.

    A resource that cannot be probed is ok=None (never a refusal; the
    engine-level OOM guard stays as the last line of defense). Messages are
    resource-SPECIFIC — a generic "no hay espacio" does not exist here."""
    settings = get_settings()
    ram = system_ram()
    storage = disk()
    vram = dedicated_vram()

    def entry(measured: int | None, required_gb: float, message: str, unit_note: str):
        required_bytes = gb_to_bytes(required_gb)
        ok = None if measured is None else (measured >= required_bytes)
        return {
            "ok": ok,
            "free_bytes": measured,
            "free_gb": bytes_to_gb(measured),
            "required_bytes": required_bytes,
            "required_gb": required_gb,
            "resource": unit_note,
            "message": "" if ok in (True, None) else message,
        }

    checks = {
        "system_ram": entry(
            ram.get("available_bytes"), settings.min_free_system_ram_gb,
            "No hay suficiente memoria RAM disponible.", "system_ram",
        ),
        "vram": entry(
            vram.get("free_bytes"), settings.min_free_vram_gb,
            "No hay suficiente memoria VRAM disponible.", "dedicated_vram",
        ),
        "disk": entry(
            storage.get("free_bytes"), settings.min_free_disk_gb,
            "No hay suficiente espacio de almacenamiento.", "disk",
        ),
    }
    blocking = next((k for k, c in checks.items() if c["ok"] is False), None)
    return {
        "checks": checks,
        "blocking": blocking,
        "message": checks[blocking]["message"] if blocking else "",
    }
