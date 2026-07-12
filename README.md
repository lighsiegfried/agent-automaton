# agent-automaton

**Fifi** — a local-first voice assistant for Windows 10. She will eventually
listen to you, reason with a local LLM (via Ollama), speak back, and execute
safe PC actions through controlled tools. No cloud APIs required.

*Fifi* is the assistant's persona (configurable via `AGENT_NAME`); the project
and repository remain `agent-automaton`. The name appears in replies,
`GET /identity`, and CLI output — it is presentation only and never changes
safety behavior. The planned wake word ("fifi", `WAKE_WORD`) is configured as
metadata for a future phase; **no wake word listening exists yet**.

**Current status (Phase 3A):** text or voice in, structured response out, in
English and Spanish. Everything is **simulated by default**. With
`ENABLE_REAL_WINDOWS_TOOLS=true`, exactly three safe actions run for real:
opening validated non-system folders, launching allowlisted apps, and web
searches in the default browser. With `ENABLE_LLM_PLANNER=true`, a local LLM
(Ollama) interprets commands — its plans are strictly validated and still pass
the same safety layer. With `ENABLE_VOICE=true`, WAV audio can be transcribed
locally (faster-whisper) and fed into the **same** command pipeline — voice
never bypasses safety. Everything else stays simulated or blocked.

## Project conventions

- Any Docker Compose file in this repo must be named **`compose.yml`**, never
  `docker-compose.yml`.
- The desktop automation runner always runs **directly on the Windows host** —
  it is never containerized (details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- Full rules: [docs/DEVELOPMENT_STANDARDS.md](docs/DEVELOPMENT_STANDARDS.md).

## Architecture at a glance

```
text command --> /command --> router (intent detection)
                                 |
                                 v
                          safety layer  --(blocked / needs confirmation)--> structured refusal
                                 |
                                 v
                          tool registry --> tool (simulated) --> structured result
                                 |
                                 v
                          memory (SQLite log)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md),
and [docs/SAFETY_RULES.md](docs/SAFETY_RULES.md) for details.

## Requirements

- Windows 10
- Python 3.11+
- (Later, not needed for the MVP) [Ollama](https://ollama.com) for local LLM inference

## Setup (Windows / PowerShell)

From the project root:

```powershell
# One-time setup: creates .venv, installs dependencies, copies .env.example to .env
.\scripts\setup_windows.ps1
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

> If script execution is blocked, run:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> If Group Policy enforces `AllSigned` machine-wide (check with
> `Get-ExecutionPolicy -List`), unsigned `.ps1` files cannot run at all.
> Pipe any script through the shell instead, e.g.:
> `Get-Content -Raw .\scripts\run_dev.ps1 | Invoke-Expression`
> — or run the underlying command directly:
> `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`

## Run the server

```powershell
.\scripts\run_dev.ps1
```

The API starts at `http://127.0.0.1:8000` (interactive docs at `/docs`).

## Local persistent runtime

For the full local stack in one command, use the runtime manager. It starts
**Dockerized Ollama** (GPU-first, a persistent support service) and the
**FastAPI server on the Windows host** (never in a container), then checks that
both are healthy:

```powershell
.venv\Scripts\python.exe scripts\local_runtime.py start    # Ollama (Docker) + API (host)
.venv\Scripts\python.exe scripts\local_runtime.py status   # what is / isn't ready
.venv\Scripts\python.exe scripts\local_runtime.py stop      # stop the host API; Ollama stays up
```

`status` shows Docker availability, the Ollama container state, GPU detection
and name, the configured model and whether it is present, whether the API is
running, and Fifi's identity:

```
=== Fifi local runtime: status ===
Docker available: True
Ollama container: running
GPU detected    : True
GPU name        : NVIDIA GeForce RTX 5060 Ti
Configured model: qwen2.5:7b
Model available : True
API running     : False
Fifi identity   : (API not running)
```

`start` requires the GPU by default (CPU inference only with an explicit
`ALLOW_CPU_OLLAMA=true`), and `--no-ollama` starts just the host API. `stop`
terminates only the API process this script started and **never deletes models
or Docker volumes**; Ollama keeps running unless you pass `--stop-ollama` (or
set `RUNTIME_KEEP_OLLAMA_RUNNING=false`). `restart` does both; `smoke` runs the
unattended LLM smoke test. Starting the runtime changes only *where* things run,
never the safety layer or tool permissions (see
[docs/SAFETY_RULES.md](docs/SAFETY_RULES.md)). It is a Python script on purpose —
machine-wide `AllSigned` policy blocks unsigned `.ps1` files.

## Try it

Health check and identity:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/identity   # who Fifi is + active layers
```

Send a command:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/command `
  -ContentType "application/json" `
  -Body '{"text": "search the web for local llm benchmarks"}'
```

A sensitive command (requires confirmation):

```powershell
# First call returns status "needs_confirmation"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/command `
  -ContentType "application/json" `
  -Body '{"text": "open app notepad"}'

# Re-send with confirm: true to proceed (still simulated)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/command `
  -ContentType "application/json" `
  -Body '{"text": "open app notepad", "confirm": true}'
```

Spanish works too:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/command `
  -ContentType "application/json" `
  -Body '{"text": "abre la carpeta descargas"}'
```

List registered tools:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/tools
```

## Run the tests

```powershell
.\scripts\test.ps1
```

## Real mode (optional)

Set `ENABLE_REAL_WINDOWS_TOOLS=true` in `.env` and restart. Only these actions
become real; each has its own validation:

| Action | Guard |
| --- | --- |
| `open_folder` | must exist, must not be Windows/System32/Program Files/hidden/system |
| `open_app` | only apps in `ALLOWED_APPS` (still requires `confirm: true`) |
| `search_web` | opens the default browser with a search URL |

Typing, file writes, deletion, shutdown, shell execution and clipboard access
remain simulated or blocked regardless of this flag.

## LLM planner (optional)

By default, commands are routed by bilingual keyword rules. To let a local LLM
interpret them instead:

1. Install [Ollama](https://ollama.com) and pull a model:
   `ollama pull qwen2.5:7b`
2. In `.env`, set `ENABLE_LLM_PLANNER=true` (and adjust `LLM_PLANNER_MODEL` /
   `LLM_PLANNER_TIMEOUT_SECONDS` if needed).
3. Restart the server. `/health` now reports `"llm_planner": true`.

Each `/command` response includes a `planner` field — `llm_planner` when a
validated LLM plan was used, `fallback_router` when the LLM failed and the
keyword rules took over, `rule_router` when the flag is off — plus a short
`plan` object (confidence, one-line reasoning summary, detected language).

The LLM never executes anything. It proposes a plan that is rejected unless it
names a known non-destructive tool with expected arguments and sufficient
confidence; accepted plans still pass the normal safety layer (see
[docs/SAFETY_RULES.md](docs/SAFETY_RULES.md)). If Ollama is down, malformed, or
uncertain, the rule-based router answers instead — the endpoint never breaks.

## Voice (optional)

Voice is fully optional: the base install has no audio dependencies, and all
`/voice` endpoints answer `{"status": "disabled"}` until you opt in.

1. Install the optional voice dependencies (faster-whisper, sounddevice,
   soundfile, numpy, pyttsx3):

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements-voice.txt
   ```

2. Set `ENABLE_VOICE=true` in `.env` and restart. `/health` now shows
   `"voice": true`. Tune `STT_MODEL` (`small` default; `medium`/`large-v3`
   fit the RTX 5060 Ti), `STT_DEVICE`, `VOICE_LANGUAGE` (auto-detects es/en).

3. Send a WAV file (`curl.exe` ships with Windows 10; `Invoke-RestMethod -Form`
   needs PowerShell 7+):

   ```powershell
   # transcribe only
   curl.exe -s -X POST http://127.0.0.1:8000/voice/transcribe -F "file=@clip.wav"

   # transcribe AND run through the normal command pipeline
   curl.exe -s -X POST "http://127.0.0.1:8000/voice/command" -F "file=@clip.wav"

   # approve a sensitive action
   curl.exe -s -X POST "http://127.0.0.1:8000/voice/command?confirm=true" -F "file=@clip.wav"
   ```

4. Or record straight from the microphone with the CLI (server must be running):

   ```powershell
   .venv\Scripts\python.exe scripts\voice_command.py              # records 5s
   .venv\Scripts\python.exe scripts\voice_command.py --seconds 8
   .venv\Scripts\python.exe scripts\voice_command.py --file clip.wav
   .venv\Scripts\python.exe scripts\voice_command.py --confirm    # approve a sensitive action
   ```

Voice commands are treated **exactly** like typed commands: same
router/planner, same safety layer, same confirmation rules, same command log.
Text-to-speech (`POST /voice/speak`, and the `speak` tool) uses Windows SAPI
via pyttsx3 when installed, and simulates otherwise — a missing TTS engine
never breaks the API. No wake word yet; recording is explicit.

## Push-to-talk (Phase 3C, Windows host)

Talk to Fifi by holding a key. `scripts/fifi_ptt.py` is a host-only client:
hold the hotkey, speak, release — it records a 16 kHz mono clip, sends it to
`/voice/command`, and prints the transcription, intent, safety status, planner
path and Fifi's reply (the API speaks it when `VOICE_SPEAK_COMMAND_RESPONSE=true`).

```powershell
# install the desktop deps (reuses the voice deps + a global-hotkey library)
.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt

# set ENABLE_VOICE=true in .env, start the runtime, then launch push-to-talk
.venv\Scripts\python.exe scripts\local_runtime.py start
.venv\Scripts\python.exe scripts\local_runtime.py ptt
```

Hold **`Ctrl+Alt+Space`** (configurable via `PTT_HOTKEY`) to talk; release to
send; **Esc** cancels the current recording. Recordings are capped at
`PTT_MAX_SECONDS` and clips shorter than `PTT_MIN_SECONDS` are ignored.

**Sensitive actions need a spoken confirmation.** When a command comes back as
`needs_confirmation`, the client remembers exactly one pending command and waits
`PTT_CONFIRM_WINDOW_SECONDS`. Say **`confirm`** (or `confirmar` / `yes confirm` /
`sí confirmar`) to resend that exact command approved; say **`cancel`**
(`cancelar` / `no`), issue a different command, or let it time out to drop it.
Matching is exact (never fuzzy), the pending command lives only in memory (a
restart clears it), and the client never changes any safety setting. There is
still no always-on microphone and no wake word — the mic is live only while the
key is held.

`local_runtime.py status` shows whether the desktop dependencies are installed
and whether the API has voice enabled. Push-to-talk is never auto-started by
`start` — it is always an explicit `ptt` launch.

## Assistant replies (Phase 3B)

Every `/command` and `/voice/command` response now includes an
`assistant_message` — a short natural reply matched to the command's language
(Spanish command → Spanish reply, English → English, unknown → Spanish):

> `"open app notepad"` → *"I need confirmation before doing that. Please
> repeat the command with confirmation to proceed."*
> `"abre descargas"` → *"Listo, simulé la carpeta — no se realizó ninguna acción real."*
> unknown command → *"Soy Fifi. No entendí el comando — ¿qué necesitas que haga? …"*

By default these are deterministic templates. With
`ENABLE_RESPONSE_GENERATOR=true`, Ollama phrases the reply instead
(`RESPONSE_MODEL`, `RESPONSE_TIMEOUT_SECONDS`); the model only rephrases the
already-decided outcome and templates remain the fallback on any failure.

With `VOICE_SPEAK_COMMAND_RESPONSE=true` (and TTS installed), `/voice/command`
also speaks the reply out loud and reports a `speech` status
(`spoken` / `simulated` / `error`) in the response.

## Docker (optional, simulated only)

```powershell
docker compose up --build
```

Runs the API in a container **in simulated mode only** — a container cannot
control the Windows desktop (no user session, windows, keyboard, or apps), and
the code additionally requires a Windows host for real execution. Use this only
as a support service; real automation always runs via `.\scripts\run_dev.ps1`.

## Optional Ollama via Docker

`compose.yml` includes an `ollama` service behind the `llm` profile (so plain
`docker compose up` never starts it). It binds to `127.0.0.1:11434` only and
stores models in the named volume `ollama-models`.

```powershell
# start Ollama, wait for it, pull the configured model, run the LLM smoke test:
.venv\Scripts\python.exe scripts\docker_llm.py

# or manage it manually:
docker compose --profile llm up -d ollama
docker compose --profile llm logs ollama
docker compose --profile llm down
```

The unattended smoke test also works against a natively installed Ollama:

```powershell
.venv\Scripts\python.exe scripts\llm_smoke.py
```

It checks `/health` and `/identity`, verifies Ollama, pulls
`LLM_PLANNER_MODEL` if missing, starts the API with the planner and response
generator enabled (simulated tools only — it forces
`ENABLE_REAL_WINDOWS_TOOLS=false` and *refuses* to test an API that reports
real tools enabled), sends bilingual commands including a destructive attempt
that must not execute, and prints a pass/fail report. These are Python scripts
on purpose — machine-wide `AllSigned` policy blocks unsigned `.ps1` files.

### GPU-first (CPU is opt-in only)

Dockerized Ollama **requires the NVIDIA GPU by default**: the service carries
a `deploy.resources.reservations.devices` NVIDIA reservation, and
`scripts/docker_llm.py` verifies GPU visibility *inside* the container
(`docker exec … nvidia-smi`, with an Ollama-log fallback) before running any
inference. If no GPU is visible it exits with a checklist instead of silently
crawling on CPU. To deliberately allow slow CPU inference, set
`ALLOW_CPU_OLLAMA=true` in `.env` — the script then continues with a warning.

Troubleshooting GPU passthrough (Windows / Docker Desktop / WSL2):

1. Install the latest NVIDIA Windows driver (includes the WSL2 CUDA driver).
2. Update WSL2: `wsl --update`
3. Docker Desktop → Settings → General → "Use the WSL 2 based engine".
4. Verify passthrough works at all: `docker run --rm --gpus all ubuntu nvidia-smi`
5. Recreate the service: `docker compose --profile llm up -d --force-recreate ollama`
6. A `could not select device driver "nvidia"` error from compose means steps
   1–3 are not complete.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable               | Default                  | Purpose                              |
| ---------------------- | ------------------------ | ------------------------------------ |
| `AGENT_NAME`           | `Fifi`                   | Assistant persona name               |
| `WAKE_WORD`            | `fifi`                   | Future wake word (metadata only)     |
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | Local Ollama server (future use)     |
| `DEFAULT_MODEL`        | `qwen2.5:7b`             | Default local LLM (future use)       |
| `STT_MODEL`            | `small`                  | faster-whisper model size (future)   |
| `TTS_ENGINE`           | `piper`                  | TTS backend (future)                 |
| `REQUIRE_CONFIRMATION` | `true`                   | Gate sensitive actions on `confirm`  |
| `ENABLE_REAL_WINDOWS_TOOLS` | `false`             | Enable the three safe real actions   |
| `ALLOWED_APPS`         | `notepad,calculator,chrome,edge,explorer` | Apps `open_app` may launch |
| `ENABLE_LLM_PLANNER`   | `false`                  | Route commands through Ollama first  |
| `LLM_PLANNER_MODEL`    | `qwen2.5:7b`             | Model used by the command planner    |
| `LLM_PLANNER_TIMEOUT_SECONDS` | `20`              | Planner request timeout              |
| `ENABLE_VOICE`         | `false`                  | Enable the /voice endpoints          |
| `STT_MODEL`            | `small`                  | faster-whisper model size            |
| `STT_DEVICE`           | `auto`                   | cuda / cpu / auto                    |
| `STT_COMPUTE_TYPE`     | `auto`                   | e.g. float16, int8, auto             |
| `VOICE_LANGUAGE`       | `auto`                   | force `es`/`en`, or auto-detect      |
| `TTS_ENGINE`           | `windows`                | `windows` (SAPI) or `simulated`      |
| `VOICE_RECORD_SECONDS` | `5`                      | CLI default recording length         |
| `VOICE_SAMPLE_RATE`    | `16000`                  | CLI recording sample rate            |
| `ENABLE_RESPONSE_GENERATOR` | `false`             | Ollama phrases assistant replies     |
| `RESPONSE_MODEL`       | `qwen2.5:7b`             | Model for reply phrasing             |
| `RESPONSE_TIMEOUT_SECONDS` | `15`                 | Reply generation timeout             |
| `VOICE_SPEAK_COMMAND_RESPONSE` | `false`          | Speak replies after /voice/command   |
| `API_HOST`             | `127.0.0.1`              | Host the runtime binds the API to     |
| `API_PORT`             | `8000`                   | Port the runtime binds the API to     |
| `RUNTIME_AUTO_START_OLLAMA` | `true`              | `start` also brings up Docker Ollama  |
| `RUNTIME_REQUIRE_GPU`  | `true`                   | Require GPU (CPU needs `ALLOW_CPU_OLLAMA`) |
| `RUNTIME_KEEP_OLLAMA_RUNNING` | `true`            | `stop` leaves Ollama running          |
| `ENABLE_PUSH_TO_TALK`  | `false`                  | Push-to-talk client opt-in (metadata) |
| `PTT_HOTKEY`           | `ctrl+alt+space`         | Global hold-to-talk hotkey            |
| `PTT_MAX_SECONDS`      | `20`                     | Max recording length (hard cap)       |
| `PTT_MIN_SECONDS`      | `0.4`                    | Clips shorter than this are ignored   |
| `PTT_CONFIRM_WINDOW_SECONDS` | `30`               | Spoken-confirmation validity window   |
| `PTT_AUDIO_FEEDBACK`   | `true`                   | Beep on record start/stop             |
| `PTT_INPUT_DEVICE`     | `default`                | Microphone device for push-to-talk    |

## Project layout

```
app/            FastAPI app, config
app/core/       logger, safety layer, memory (SQLite), command router
app/voice/      STT / TTS / wake word placeholders
app/llm/        Ollama client placeholder, prompt templates
app/tools/      tool registry + simulated Windows/browser/file/text tools
app/schemas/    Pydantic request/response models
storage/        local data (SQLite DB) — gitignored
scripts/        PowerShell setup, dev-run and test scripts
tests/          pytest suite (safety, routing, allowlists, real-vs-simulated)
docs/           architecture, roadmap, safety model, development standards
compose.yml     optional: API in simulated mode (never the desktop runner)
```
