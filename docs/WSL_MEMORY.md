# Docker / WSL2 memory for the Fifi runtime (Phase 3D.0.5)

Fifi's heavy local tenants — **Ollama** (the planner/response LLM) and the
**Voice Lab** neural TTS worker — run in Docker Desktop, which on Windows is
backed by a **single WSL2 virtual machine**. Every container shares that one
VM's RAM. The Windows host having 16 GB does **not** mean Docker sees 16 GB:
WSL2 caps the VM's memory (by default to a fraction of the host), and both heavy
tenants draw from that same pool.

If the VM is too small, loading Qwen (VoiceDesign ~5 GB, clone ~2 GB) while
Ollama's 7B model is resident makes the VM swap. Synthesis then crawls or the
worker is killed natively (no traceback). The coordinator's daily-use policy
(load only the active voice's engine, unload idle heavy models) is designed to
fit this budget, but it cannot conjure RAM that the VM does not have.

## Recommended `.wslconfig`

The runtime **never edits `.wslconfig` for you** — it only *detects* the RAM
Docker/WSL sees and warns when it is below **12 GB**. Configure it yourself.

Create or edit `%UserProfile%\.wslconfig` (i.e. `C:\Users\<you>\.wslconfig`):

```ini
[wsl2]
# Give the WSL2 VM enough RAM for Ollama + the Voice Lab at the same time.
# 14 GB leaves ~2 GB for Windows on a 16 GB machine; lower it if the desktop
# starts swapping, raise it on a machine with more RAM.
memory=14GB
# A little swap absorbs transient spikes during a model load without thrashing.
swap=8GB
# Optional: cap CPUs if you want to reserve host cores.
# processors=8
```

Notes:
- **≥ 12 GB** is the floor the runtime warns below. 14 GB is comfortable on a
  16 GB host; on 32 GB you can give 20–24 GB.
- `.wslconfig` uses `GB`/`MB` suffixes (binary units). `memory=14GB` = 14 GiB.
- This file lives on the **Windows** side (`%UserProfile%`), not inside WSL.

## Apply the change (restart required)

`.wslconfig` is read only when the WSL2 VM starts, so you must restart it:

```powershell
# 1. Shut the WSL2 VM down completely (closes all distros + the Docker VM).
wsl --shutdown

# 2. Restart Docker Desktop (Quit from the tray icon, then reopen it), or:
#    Restart-Service com.docker.service   # if you run Docker as a service
```

Then confirm what Docker now sees:

```powershell
docker info --format "{{.MemTotal}}"        # bytes the Docker VM has
python scripts/local_runtime.py model-status   # prints "Docker/WSL RAM : NN GB"
python scripts/local_runtime.py status         # also prints the warning if < 12 GB
```

## Dedicated VRAM is not shared GPU memory

The runtime reports **dedicated** VRAM only (from NVML / `nvidia-smi`), never
Windows Task Manager's combined "GPU memory" (dedicated **+** shared). Shared
GPU memory is system RAM windowed to the GPU under WDDM; treating it as VRAM
would badly over-report headroom and invite OOM. `model-status` labels the VRAM
figure with its source (`nvml` / `nvidia-smi`) and reports shared memory
separately (and, on this platform, as absent) so the two are never conflated.

## Related

- `docs/ARCHITECTURE.md` — Windows host vs Docker (why automation stays on the
  host and only support services run in containers).
- `voice_lab/.env.example` — `VOICE_LAB_MODE`, `VOICE_LAB_IDLE_UNLOAD_SECONDS`,
  `VOICE_LAB_MAX_LOADED_HEAVY_MODELS`, and the RAM/VRAM thresholds.
- `python scripts/local_runtime.py voice-mode low-memory` — force Kokoro/Windows
  only when the VM is memory-starved and you cannot restart right now.
