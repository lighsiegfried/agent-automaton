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

## Phase 3C — Voice, continued

- [ ] Push-to-talk loop; wake word (openWakeWord) after
- [ ] Voice confirmation flow for sensitive actions ("say yes to confirm")
- [ ] Piper/Kokoro as higher-quality TTS engines

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
