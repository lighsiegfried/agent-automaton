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
