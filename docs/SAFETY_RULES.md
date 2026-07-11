# Safety model

Every action the assistant can take is a **tool** declared in the registry
(`app/tools/registry.py`) with an explicit safety level. The router never calls
a function directly; every call passes through `app/core/safety.py`. There is
deliberately no way to execute an action that skips this path.

## Safety levels

| Level | Meaning | Behavior | Examples |
| --- | --- | --- | --- |
| `safe` | No lasting effect on the system or data | Executed / simulated immediately | `speak`, `search_web`, `open_folder`, `list_folder` |
| `sensitive` | Interacts with apps or injects input; reversible but intrusive | Requires `confirm: true` on the request (when `REQUIRE_CONFIRMATION=true`) | `open_app`, `type_text` |
| `destructive` | Can lose data or change system state | **Blocked unconditionally** | `delete_file`, `shutdown_pc` |

## Rules

1. **Destructive means blocked, not confirmed.** A confirmation click is too
   cheap a barrier for data loss. Enabling a destructive tool requires a
   deliberate code change and its own review of scope and reversibility —
   it can never be toggled on via config or API.
2. **Confirmation is per-request.** A `confirm: true` approves exactly one
   command. There is no "always allow" state.
3. **Safety level is declared where the tool is defined**, next to its code,
   so a review of any tool file shows both what it does and how it is gated.
4. **Default to the stricter level.** If a tool is arguably sensitive,
   it is sensitive. If it is arguably destructive, it is destructive.
5. **Simulate first.** New tools ship as simulations returning a `would_do`
   description. They become real only after their input validation,
   allowlists, and failure modes are worked out.
6. **Everything is logged.** Every command — including blocked and
   unconfirmed ones — is written to the SQLite command log (`/history`).

## Current status (Phase 1.5)

Everything is simulated by default. With `ENABLE_REAL_WINDOWS_TOOLS=true`
(Windows host only — the flag is inert inside Docker), exactly three safe
actions run for real, each with its own validation:

| Real action | Validation |
| --- | --- |
| `open_folder` | must exist and be a directory; Windows, System32, Program Files, ProgramData and hidden/system folders are refused |
| `open_app` | strict `ALLOWED_APPS` allowlist (Spanish aliases map to canonical names); launch targets come from a code-defined map, never raw user text; still confirmation-gated |
| `search_web` | opens the default browser with an encoded search URL |

Validation runs in **both** modes, so simulation is an honest preview.
Still fully simulated or blocked regardless of the flag: `type_text`
(real input injection is Phase 4), file writes, deletion, shutdown, shell
execution, clipboard access, and WhatsApp/messaging automation.

## LLM output is never trusted directly

With `ENABLE_LLM_PLANNER=true`, a local LLM proposes command plans — and that
is all it does. The plan is treated as untrusted input:

1. It must be a single JSON object matching the `CommandPlan` schema exactly
   (unknown keys rejected).
2. `tool_name` must exist in the registry. Invented tools are rejected.
3. Destructive tools are rejected even if the LLM suggests them — they are
   not even shown in the planner's tool catalog.
4. The intent must map to that exact tool; mismatches are rejected.
5. Arguments are restricted to the tool handler's declared parameters.
6. Plans below the confidence threshold (0.6) are rejected.
7. An accepted plan still passes `safety.evaluate()` like any rule-routed
   command: sensitive tools require `confirm=true` no matter what the plan's
   `requires_confirmation` says. The plan can only **tighten** safety
   (request confirmation for a safe tool), never loosen it.
8. Any rejection falls back to the deterministic rule-based router.

The model's chain-of-thought is never requested or exposed; responses carry
only a one-sentence `reasoning_summary`.

## Voice commands are typed commands

A transcribed voice command is handed to `handle_command()` — the **exact**
function behind `POST /command`. There is no voice-specific execution path:

- same router / LLM planner, same intent detection
- same safety levels and the same `confirm` requirement for sensitive actions
  (`/voice/command?confirm=true` is the spoken-world equivalent of the typed
  `confirm: true`)
- destructive actions stay blocked no matter how they were requested
- every voice command lands in the same SQLite command log

Speech synthesis is output-only and safe by construction: `/voice/speak` and
the `speak` tool can only produce audio, and they simulate when TTS is not
installed. Voice as a whole is disabled by default (`ENABLE_VOICE=false`);
there is no always-on microphone and no wake word yet — recording is an
explicit, user-initiated CLI action.

## Spoken responses are summaries, not decisions

The `assistant_message` (and its spoken form when
`VOICE_SPEAK_COMMAND_RESPONSE=true`) is generated **after** the router, safety
layer and tool have fully decided and recorded the outcome. It is a
description of a safe, already-final command result — it cannot execute,
confirm, retry, or modify anything. When Ollama phrases it
(`ENABLE_RESPONSE_GENERATOR=true`), the model receives only the final status
and message, its output is length-capped, chain-of-thought is never requested
or exposed, and deterministic templates take over on any failure. A blocked
destructive action is always answered with a clear refusal, and an
unconfirmed sensitive action always asks for confirmation.

## Persona never changes safety

The assistant's name ("Fifi", `AGENT_NAME`) is presentation only. It is
interpolated into reply templates, the `/identity` endpoint, LLM prompts and
CLI output **after** every routing and safety decision is already made.
Nothing in the safety layer, tool registry, allowlists, or confirmation logic
reads the persona: renaming the agent, or addressing it by name in a command,
cannot unlock, soften, or bypass anything. The configured wake word is inert
metadata — no microphone listens for it.

## When implementing more real tools (Phase 2+)

- Validate and canonicalize all paths; restrict to allowlisted roots.
- Launch apps only from an explicit allowlist, never from raw user text.
- Never pass user text to a shell; use argument lists.
- Keep a per-tool `dry_run` escape hatch to force simulation.
- Voice confirmations must be as explicit as API confirmations.
