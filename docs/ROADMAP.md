# Roadmap

## Phase 0 — MVP scaffold (done)

- [x] FastAPI server with `/health`, `/command`, `/tools`, `/history`
- [x] Rule-based intent router (open_folder, open_app, type_text, search_web, speak)
- [x] Safety layer: safe / sensitive (confirm) / destructive (blocked)
- [x] Tool registry with simulated tools
- [x] SQLite command log
- [x] Placeholder modules for STT, TTS, wake word, Ollama
- [x] Windows setup + run scripts, docs

## Phase 1.5 — Standardization + real safe actions (done)

- [x] Development standards doc; `compose.yml` naming rule; Windows-host
      automation policy (enforced in code via `real_windows_tools_enabled()`)
- [x] Optional `compose.yml`: API in simulated mode only
- [x] `ENABLE_REAL_WINDOWS_TOOLS` flag: real `open_folder` (validated paths),
      `open_app` (`ALLOWED_APPS` allowlist), `search_web` (default browser)
- [x] Spanish + English routing
- [x] pytest suite + `scripts/test.ps1`

## Phase 2 — LLM command planner (done)

- [x] `app/llm/command_planner.py`: Ollama proposes a strict JSON `CommandPlan`
      (intent, tool, arguments, confidence, one-line reasoning summary,
      requires_confirmation, language)
- [x] Plan validation: known non-destructive tools only, intent/tool match,
      declared arguments only, confidence ≥ 0.6; plans can tighten safety,
      never loosen it
- [x] `ENABLE_LLM_PLANNER` / `LLM_PLANNER_MODEL` / `LLM_PLANNER_TIMEOUT_SECONDS`;
      default off — behavior unchanged
- [x] Fallback to the rule router on any planner failure; `planner` field in
      responses (`rule_router` / `llm_planner` / `fallback_router`)
- [ ] `/chat` endpoint for free-form conversation with the local model

## Phase 2.5 — More real (but safe) tool execution

- [x] `open_folder` for real (validated, non-system directories only) — Phase 1.5
- [x] `search_web` via the default browser (`webbrowser.open`) — Phase 1.5
- [x] `open_app` against an explicit allowlist — Phase 1.5
- [ ] Real `list_folder` restricted to allowlisted roots
- [ ] Per-tool dry-run flag so any tool can be forced back into simulation

## Phase 3A — Voice pipeline, no wake word (done)

- [x] Optional dependencies split into `requirements-voice.txt`
- [x] faster-whisper STT (`app/voice/stt.py`): lazy-loaded, configurable model/
      device/compute type, es/en auto-detect, graceful "unavailable" when missing
- [x] TTS via Windows SAPI/pyttsx3 with simulated fallback (`app/voice/tts.py`)
- [x] `/voice/transcribe`, `/voice/speak`, `/voice/command` — all answer
      "disabled" until `ENABLE_VOICE=true`; voice feeds the same /command flow
- [x] `scripts/voice_command.py`: record from the mic, send, print the outcome

## Phase 3B — Spoken response layer (done)

- [x] `app/llm/response_generator.py`: short bilingual `assistant_message` on
      every /command and /voice/command response; deterministic templates by
      default, Ollama phrasing behind `ENABLE_RESPONSE_GENERATOR` with
      template fallback
- [x] Language follows the command: detected voice language > planner language
      > keyword heuristic > Spanish
- [x] `VOICE_SPEAK_COMMAND_RESPONSE=true` speaks the reply after
      /voice/command and reports a `speech` status (spoken/simulated/error)
- [x] Distinct replies per outcome: executed, simulated, needs-confirmation,
      rejected, blocked (refusal), unknown (asks for a clearer command)

## Phase 3B.5 — Fifi identity/persona layer (done)

- [x] `AGENT_NAME=Fifi` and `WAKE_WORD=fifi` config (wake word is inert
      metadata until Phase 3C — nothing listens)
- [x] Reply templates use the name where natural (unknown-command intro),
      never in routine confirmations/results
- [x] `GET /identity`: project name, agent name, wake word (+ inactive flag),
      version, active optional layers — no paths or secrets
- [x] CLI prints "Fifi heard / Fifi response" (name fetched from /identity)
- [x] Persona is provably safety-inert (tested: renaming the agent or
      addressing it by name changes no safety decision)

## Phase 3B.6 — Dockerized Ollama + unattended LLM smoke tests (done)

- [x] `ollama` service in `compose.yml` behind the `llm` profile
      (localhost-only port, named model volume, optional GPU block)
- [x] `scripts/llm_smoke.py`: unattended end-to-end check — Ollama, model
      pull, /health, /identity, bilingual /command cases incl. a destructive
      attempt that must not execute; forces simulated tools, refuses APIs
      with real tools enabled
- [x] `scripts/docker_llm.py`: compose up → wait → pull → smoke test
      (Python, because AllSigned policy blocks unsigned .ps1)
- [x] Containment tests: tool surface locked; docker/shell/compose not
      launchable through commands; Docker mode never reports real tools
- [x] **GPU-first hotfix:** active NVIDIA reservation on the ollama service;
      docker_llm.py verifies GPU inside the container and refuses CPU
      inference unless ALLOW_CPU_OLLAMA=true; reports profile, container,
      GPU name, model, and smoke result

## Phase 3B.7 — Local persistent runtime (done)

- [x] `scripts/local_runtime.py`: one-command `start` / `stop` / `restart` /
      `status` / `smoke` for Fifi's local stack — Dockerized Ollama (GPU-first,
      a support service) plus the FastAPI server on the Windows host
- [x] `start`: brings up Ollama behind the `llm` profile, verifies GPU
      visibility (CPU only with explicit `ALLOW_CPU_OLLAMA=true`), verifies/pulls
      the model, launches the host API if not already running, checks `/health`
      and `/identity`
- [x] `status`: Docker availability, Ollama container state, GPU detected + name,
      configured model, model availability, API running, Fifi identity
- [x] `stop`: terminates the host API started by the script and clears its PID;
      **never deletes models or Docker volumes**; Ollama stays running unless
      `--stop-ollama` (or `RUNTIME_KEEP_OLLAMA_RUNNING=false`)
- [x] `compose.yml` `ollama` gains `restart: unless-stopped` for persistence;
      still `llm`-profile-gated, localhost-only, GPU-first, model volume kept
- [x] PID under `storage/runtime/`, logs under `storage/logs/` (gitignored);
      config `API_HOST` / `API_PORT` / `RUNTIME_AUTO_START_OLLAMA` /
      `RUNTIME_REQUIRE_GPU` / `RUNTIME_KEEP_OLLAMA_RUNNING` (safe local defaults)
- [x] Tests: start/status/stop flows, GPU-required failure, CPU-only opt-in,
      never enables real Windows tools, API never runs in Docker, compose naming,
      runtime state/logs gitignored — all mocked (no Docker/GPU needed)

## Phase 3C — Push-to-talk client (done)

- [x] `scripts/fifi_ptt.py`: Windows-host push-to-talk client — hold a global
      hotkey (`PTT_HOTKEY`, default `ctrl+alt+space`), speak, release; the clip
      is captured as 16 kHz mono WAV, sent to `/voice/command`, and the
      transcription, intent, safety status, planner path and Fifi's reply are
      printed (API speaks it when `VOICE_SPEAK_COMMAND_RESPONSE=true`)
- [x] Short-lived, exact-phrase voice confirmation: one pending sensitive
      command at a time; `confirm`/`confirmar`/`yes confirm`/`sí confirmar`
      resends the *exact* original with `confirm=true`; `cancel`/`cancelar`/`no`
      or a `PTT_CONFIRM_WINDOW_SECONDS` timeout clears it; a different command
      clears it; never fuzzy-matched; never persisted (gone on restart)
- [x] Startup validation: refuses unless the API is agent-automaton with voice
      enabled and the desktop deps present; warns (does not fail) when the LLM
      planner or spoken responses are off; never changes safety config
- [x] Audio/hotkey safety: no concurrent recordings, key-repeat debounced,
      `PTT_MAX_SECONDS` cap, Esc-to-cancel, safe mic-disconnect handling, no
      leftover WAV files or background threads on exit
- [x] `requirements-desktop.txt` (reuses `requirements-voice.txt` + `keyboard`);
      config `ENABLE_PUSH_TO_TALK` / `PTT_HOTKEY` / `PTT_MAX_SECONDS` /
      `PTT_MIN_SECONDS` / `PTT_CONFIRM_WINDOW_SECONDS` / `PTT_AUDIO_FEEDBACK` /
      `PTT_INPUT_DEVICE`
- [x] `python scripts/local_runtime.py ptt` launches the client (never auto-started
      by `start`); `status` reports push-to-talk availability and voice-enabled
- [x] Tests with mic/hotkey/HTTP/TTS fully mocked (record→submit, too-short,
      max-duration, API down, voice disabled, confirm/cancel/expire, different
      command clears pending)
- [ ] Wake word (openWakeWord) and always-on listening — still not started
- [ ] Piper/Kokoro as higher-quality TTS engines

## Phase 3C.1 — Blackwell-safe STT + API protection (done)

- [x] `ctranslate2>=4.8.1` pin; GPU-arch detection; `resolve_stt_config()`
      resolves `auto`→`float16` on RTX 50 (Blackwell) and never uses INT8 there
      (fixes `CUBLAS_STATUS_NOT_SUPPORTED`); logs device/compute/CT2/GPU
- [x] `GET /voice/preflight` loads the model and runs a harmless silence probe
      (no tool runs); STT runs in a bounded worker thread (never the event loop)
      with a GPU lock — a CUDA failure returns a structured error and leaves
      `/health` + `/identity` reachable; never a fake successful voice result
- [x] `STT_ALLOW_CPU_FALLBACK=false` (opt-in cpu/int8 fallback, clearly reported)
- [x] Runtime: `status` validates `/health` (not just PID); `start` restarts an
      owned dead API once and never touches a process it doesn't own
- [x] PTT: connection vs processing timeouts separated (a slow STT is not
      "API unreachable")

## Phase 3C.2 — PTT release detection + deterministic folders (done)

- [x] Push-to-talk hotkey modelled as individual keys (`HotkeyListener`): starts
      when the full combo is down, stops the instant ANY key is released,
      ignores key-repeat, and re-arms only after a full release (max-duration
      stops one recording, never immediately restarts); supports combos and
      single keys (F8); Esc still cancels
- [x] Leading invocation prefixes ("Fifi", "Fifi,", "hey Fifi", "oye Fifi")
      stripped for routing only — never mid-text; original transcription
      preserved in responses and history
- [x] Deterministic known-folder normalization (ES/EN) maps
      descargas/documentos/escritorio · downloads/documents/desktop (and filler
      like de/la/mi/carpeta/folder) to `Path.home()` folders, overriding a messy
      LLM/fallback argument; arbitrary text is refused with a clarification and
      never resolved under the CWD; system-folder rejection preserved

## Phase 4 — Deeper automation (carefully)

- [ ] pywinauto window management (focus, move, close — with confirmation)
- [ ] Playwright browser automation for whitelisted sites
- [ ] Typed text injection with an on-screen preview before sending

## Phase 5 — Memory and personality

- [ ] Preference storage (name, favorite apps, default folders)
- [ ] Conversation context window persisted in SQLite
- [ ] Daily-summary / reminders via scheduled local jobs

## Explicit non-goals for now

- Cloud APIs of any kind
- WhatsApp or other messaging automation
- Unattended destructive actions (these stay blocked, not just confirmed)
- Multi-user / SaaS features
