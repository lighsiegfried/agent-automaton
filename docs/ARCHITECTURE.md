# Architecture

agent-automaton is a local-first assistant for Windows 10, presented to the
user as **Fifi**. Everything runs on the user's machine: API server, intent
routing, memory, and (later) speech and LLM inference. No cloud APIs are
required.

Fifi is a **persona layer** above the command pipeline, not part of it: the
name (`AGENT_NAME`) appears in assistant replies, `GET /identity`, prompts and
CLI output, while the pipeline underneath — router/planner → safety layer →
tool registry → memory — is identical regardless of the persona. Changing the
name changes zero safety decisions. The configured wake word (`WAKE_WORD`,
"fifi") is future metadata for Phase 3C; nothing listens for it yet.

## Design principles

1. **Local-first** — no external services needed; Ollama provides LLM inference locally.
2. **Modular** — each capability (voice, LLM, tools) lives in its own package with a small interface, so implementations can be swapped without touching the rest.
3. **Safety by construction** — every action goes through a central registry and safety layer; there is no code path that executes an action without a safety decision.
4. **Simulate first** — new tools start as simulations that describe what they *would* do, and only become real once their safety story is settled.
5. **Windows host for automation** — anything that touches the desktop runs directly on the Windows host, never in a container (see below).

## Windows host vs Docker

The desktop automation runner **must run directly on Windows** and is never
containerized. Real automation needs things a container fundamentally does not
have:

- the interactive **user session** (a container has no logged-in desktop),
- the **window manager** — enumerating, focusing and manipulating windows,
- **keyboard/mouse input injection** into that session,
- the user's **installed applications**, file associations and default browser.

This is enforced in code, not just by convention: `real_windows_tools_enabled()`
(`app/config.py`) requires both `ENABLE_REAL_WINDOWS_TOOLS=true` **and**
`sys.platform == "win32"`, so a Linux container can never execute real actions
even if misconfigured.

Docker is allowed only for **optional support services**. The provided
`compose.yml` (always this name, never `docker-compose.yml`) defines:

- `api` — the FastAPI app in simulated mode, useful for trying the
  routing/safety pipeline from another machine.
- `ollama` — local LLM inference for the planner and response generator,
  gated behind the `llm` profile (`docker compose --profile llm up -d ollama`),
  bound to `127.0.0.1` only, with models in a named volume. **GPU-first:** the
  service reserves the NVIDIA GPU by default and `scripts/docker_llm.py`
  verifies GPU visibility inside the container before any inference; CPU-only
  operation requires an explicit `ALLOW_CPU_OLLAMA=true` opt-in.

The split is deliberate: **the host runner acts, support services only think.**
Ollama receives text and returns text — its output enters the pipeline as an
untrusted plan or as a reply to phrase, both of which are validated and pass
the same safety layer. Nothing that runs in Docker can touch the desktop, and
`scripts/llm_smoke.py` refuses to run against an API with real Windows tools
enabled.

## Local persistent runtime (Phase 3B.7)

`scripts/local_runtime.py` is the one-command manager for Fifi's local stack.
It embodies the host/Docker split directly:

- **Dockerized Ollama** is a persistent *support service*. `compose.yml` marks
  the `ollama` service `restart: unless-stopped` (still `llm`-profile-gated,
  bound to `127.0.0.1`, GPU-first, models in the `ollama-models` volume), so the
  LLM backend survives reboots. `start` brings it up, verifies the GPU inside
  the container, and verifies/pulls the model.
- **The FastAPI server runs on the Windows host**, never in Docker — it is the
  process that can (when `.env` allows) touch the desktop. `local_runtime.py`
  launches `uvicorn app.main:app` on the host, records its PID under
  `storage/runtime/`, and streams its output to `storage/logs/` (both
  gitignored). It inherits configuration from `.env`; it does **not** override
  safety-relevant flags and never enables real Windows tools.

```
python scripts/local_runtime.py start
        |
        |-- Docker:  compose --profile llm up -d ollama   (GPU-first support svc)
        |            wait -> verify GPU in container -> verify/pull model
        |
        `-- Host:    uvicorn app.main:app  (PID -> storage/runtime/api.pid)
                     wait -> GET /health -> GET /identity (agent_name == "Fifi")
```

`stop` terminates only the host API process this script started and clears its
PID; it leaves Ollama running by default (persistent) and **never removes models
or Docker volumes** — `--stop-ollama` stops the service with `compose stop`
(not `down`, never `-v`). `status` reports Docker availability, the Ollama
container state, GPU detection and name, the configured model and whether it is
present, whether the host API is up, and Fifi's `/identity`. Startup is purely
about *where* things run — it changes no tool permission (see
`docs/SAFETY_RULES.md`). Config: `API_HOST`, `API_PORT`,
`RUNTIME_AUTO_START_OLLAMA`, `RUNTIME_REQUIRE_GPU`, `RUNTIME_KEEP_OLLAMA_RUNNING`.

## Execution modes

- **Simulated (default)** — every tool returns a `would_do` description.
  Input validation still runs, so simulation is an honest preview: a command
  that would be rejected for real is rejected in simulation too.
- **Real (`ENABLE_REAL_WINDOWS_TOOLS=true`, Windows host only)** — exactly
  three safe actions execute: `open_folder` (existing, non-system folders),
  `open_app` (strict `ALLOWED_APPS` allowlist, still confirmation-gated), and
  `search_web` (default browser). `type_text` and every destructive action are
  unaffected by the flag.

A response's `status` field reports what actually happened: `simulated`,
`executed`, `rejected` (tool refused its input), `needs_confirmation`,
`blocked`, or `not_handled`.

## Layers

```
+--------------------------------------------------------------+
|                     FastAPI (app/main.py)                    |
|          /health   /command   /tools   /history              |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|                 Router (app/core/router.py)                  |
|                                                              |
|  ENABLE_LLM_PLANNER=false          ENABLE_LLM_PLANNER=true   |
|  rule-based bilingual rules        LLM planner (app/llm/     |
|  (also the fallback path)          command_planner.py) with  |
|                                    strict plan validation;   |
|                                    falls back to rules on    |
|                                    any failure               |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|              Safety layer (app/core/safety.py)               |
|   safe -> proceed | sensitive -> confirm | destructive -> block |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|            Tool registry (app/tools/registry.py)             |
|   windows_tools | browser_tools | file_tools | text_tools    |
|   (all simulated in the MVP)                                 |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|             Memory (app/core/memory.py, SQLite)              |
|   command log; later: preferences, conversation context      |
+--------------------------------------------------------------+
```

## Voice pipeline (Phase 3A, `ENABLE_VOICE=true`)

Voice sits *above* the router and never bypasses it — a transcription enters
the pipeline at exactly the same point as typed text:

```
microphone (scripts/voice_command.py, explicit recording — no wake word yet)
     |  or any .wav upload
     v
POST /voice/command  (app/voice/api.py)
     |
     v
SpeechToTextService (app/voice/stt.py, faster-whisper, es/en auto-detect)
     |  transcription text
     v
handle_command()  <-- IDENTICAL entry point to POST /command:
     |                router/planner -> safety layer -> tool -> memory log
     |                -> response generator (assistant_message)
     v
{"transcription", "language", "assistant_message",
 "command": <normal CommandResponse>, "speech": <optional TTS status>}
     |
     v  (only if VOICE_SPEAK_COMMAND_RESPONSE=true)
TextToSpeechService speaks assistant_message out loud

POST /voice/speak --> TextToSpeechService (app/voice/tts.py, Windows SAPI
via pyttsx3; simulates when unavailable — never blocks the API)
```

The voice dependencies (`requirements-voice.txt`) are optional. Missing
packages degrade gracefully: STT answers `"unavailable"` with an install hint,
TTS falls back to simulation. Wake word detection (`app/voice/wake_word.py`)
remains a placeholder for Phase 3B.

## Push-to-talk client (Phase 3C, Windows host)

`scripts/fifi_ptt.py` is a **host-only client** — like the desktop automation
runner, it never runs in Docker. It holds no authority: it only records audio
while a global hotkey is held and posts it to the same `/voice/command`
endpoint a `.wav` upload would use. It cannot enable real tools, cannot unblock
anything, and adds no new execution path.

```
hold PTT_HOTKEY (default ctrl+alt+space) on the Windows host
     |  record while held -> 16 kHz mono WAV (Esc cancels; PTT_MAX_SECONDS caps)
     v
POST /voice/command  -> transcription -> handle_command() (same safety layer)
     |
     v
print transcription / intent / safety status / planner / Fifi reply
     |  (API speaks it when VOICE_SPEAK_COMMAND_RESPONSE=true)
     v
temp WAV deleted immediately after processing
```

Sensitive actions use a **one-shot, short-lived confirmation** held only in the
client's memory: a `needs_confirmation` result stores exactly one pending
command; the next *exact* phrase `confirm` / `confirmar` / `yes confirm` /
`sí confirmar` resends the original command with `confirm=true` (via
`POST /command`), while `cancel` / `cancelar` / `no`, a different command, a
`PTT_CONFIRM_WINDOW_SECONDS` timeout, or a process restart all clear it. Matching
is exact — never fuzzy — so nothing an LLM or a mumble produces can approve an
action by accident.

Launch it with `python scripts/local_runtime.py ptt` (never auto-started by
`start`). `local_runtime.py status` reports whether the desktop dependencies are
present and whether the API has voice enabled. There is still **no always-on
microphone and no wake word** — the mic is live only while the key is held.

## Key components

| Component | File | Role |
| --- | --- | --- |
| Settings | `app/config.py` | pydantic-settings; reads `.env`; no hardcoded paths |
| Router | `app/core/router.py` | intent detection + orchestration of safety/tools/memory |
| Safety | `app/core/safety.py` | single decision point for every tool call |
| Registry | `app/tools/registry.py` | declarative tool catalog with safety levels |
| Memory | `app/core/memory.py` | SQLite command log in `storage/` |
| Schemas | `app/schemas/commands.py` | request/response models, intent + safety enums |
| STT | `app/voice/stt.py` | faster-whisper (optional dep, lazy-loaded, es/en) |
| TTS | `app/voice/tts.py` | Windows SAPI via pyttsx3 (optional, simulates if absent) |
| Voice API | `app/voice/api.py` | /voice/transcribe, /voice/speak, /voice/command |
| Wake word | `app/voice/wake_word.py` | placeholder for openWakeWord (Phase 3C) |
| LLM planner | `app/llm/command_planner.py` | validated plans from Ollama (optional) |
| Response generator | `app/llm/response_generator.py` | bilingual assistant_message; templates by default, Ollama optional |

## Rule router vs LLM planner

Two planners feed the **same** execution path (safety layer → tool → memory):

| | Rule router (default) | LLM planner (`ENABLE_LLM_PLANNER=true`) |
| --- | --- | --- |
| How | Bilingual keyword rules in `app/core/router.py` | Ollama proposes a JSON `CommandPlan` (`app/llm/command_planner.py`) |
| Strengths | Deterministic, instant, works offline-of-Ollama | Understands phrasing variations ("abre mi carpeta de descargas") |
| Trust model | Code | **Untrusted** — plan is validated before use |
| On failure | Returns `not_handled` | Falls back to the rule router |

A validated plan may only *tighten* safety: if the LLM sets
`requires_confirmation` on a safe tool, confirmation is required; it can never
remove the confirmation a sensitive tool already needs, and destructive tools
are rejected outright (they are not even listed in the planner's tool catalog).
The response's `planner` field reports which path answered:
`rule_router`, `llm_planner`, or `fallback_router`.

## Request flow for `POST /command`

1. Body is validated into `CommandRequest` (`text`, optional `confirm`).
2. Intent + arguments are produced by the rule router (keyword rules, English
   and Spanish) — or, when enabled, by the LLM planner after strict validation
   (known tool, matching intent, expected arguments, confidence ≥ 0.6).
3. `safety.evaluate()` checks the tool's declared safety level against the
   request's `confirm` flag and `REQUIRE_CONFIRMATION`.
4. If allowed, the tool handler runs (simulated or real per
   `ENABLE_REAL_WINDOWS_TOOLS`).
5. The response generator produces `assistant_message` — a short bilingual
   reply describing the outcome (templates by default; Ollama when
   `ENABLE_RESPONSE_GENERATOR=true`, with templates as fallback). It runs
   *after* all decisions are made and can only describe them, never change them.
6. The outcome is written to the SQLite command log.
7. A `CommandResponse` is returned with intent, tool, safety level, status,
   message, result, planner source, plan summary (LLM plans) and
   `assistant_message`.

## Extending with a new tool

```python
# app/tools/my_tools.py
from app.schemas.commands import SafetyLevel
from app.tools.registry import registry

@registry.register(
    name="my_tool",
    description="What it does",
    safety_level=SafetyLevel.SENSITIVE,
)
def my_tool(param: str = "") -> dict:
    return {"simulated": True, "would_do": f"..."}
```

Then import the module in `load_tools()` (`app/tools/registry.py`) and, if the
tool should be reachable from text commands, add an intent rule and a mapping
in `app/core/router.py`.

## Hardware notes

Target machine: Ryzen 7 2700X, 16 GB RAM, RTX 5060 Ti 16 GB VRAM.

- 16 GB VRAM comfortably fits a 7B–14B model at Q4/Q5 via Ollama plus a
  faster-whisper `medium`/`large-v3` STT model.
- Piper TTS runs fine on CPU, leaving the GPU for LLM + STT.
