# Development standards

Conventions for anyone (human or AI) extending agent-automaton.

## Naming

- Compose files are named **`compose.yml`** — never `docker-compose.yml`.
- Python modules, functions, variables: `snake_case`. Classes: `PascalCase`.
- Config values: `UPPER_SNAKE_CASE` in `.env`, mirrored as `lower_snake_case`
  fields in `app/config.py::Settings`. Every new value gets a commented entry
  in `.env.example`.
- Tool names are `snake_case` verbs (`open_folder`, `search_web`) and match
  their `Intent` value when router-reachable.

## Docker policy

- **The desktop automation runner is never containerized.** Real automation
  needs the interactive Windows session (windows, keyboard, mouse, installed
  apps); containers don't have it. `real_windows_tools_enabled()` additionally
  requires `sys.platform == "win32"`, so a container can never flip it on.
- Docker is only for optional support services. The provided `compose.yml`
  runs the API in simulated mode and nothing else. Don't add databases or
  extra services until something actually needs them.

## Safety rules (summary — full text in SAFETY_RULES.md)

- Every capability is a registered tool with an explicit `SafetyLevel`.
- New tools ship **simulated first**; they return a `would_do` description.
- Real execution is opt-in via `ENABLE_REAL_WINDOWS_TOOLS` and is only ever
  granted to safe, validated, allowlisted actions.
- Destructive actions are blocked in code, never merely gated by config.
- Input validation runs in both simulated and real mode, so both modes reject
  the same inputs and simulation is an honest preview.
- Never pass user text to a shell or interpolate it into a command string.
  Launch targets come from code-defined maps behind allowlists.

## Adding a tool

1. Write the handler in the right `app/tools/*.py` module, decorated with
   `@registry.register(name=..., description=..., safety_level=...)`.
2. Return a `dict`: `{"error": ...}` to reject input,
   `{"simulated": True, "would_do": ...}` for simulation,
   `{"simulated": False, ...}` only after a real, guarded action.
3. If it's a new module, import it in `load_tools()` (`app/tools/registry.py`).
4. Router-reachable tools also need: an `Intent` value, a rule in
   `app/core/router.py` (English **and** Spanish), and an `INTENT_TOOL_MAP` entry.
5. Add tests: routing, rejection cases, and simulated-vs-real behavior with
   launch primitives stubbed (see `tests/conftest.py::launches`).

## Testing

- `.\scripts\test.ps1` (pytest). Tests must never launch real apps, open real
  windows, or touch the real command-log DB — `tests/conftest.py` stubs the
  launch primitives and swaps in a temporary database; keep it that way.
- Every new tool or router rule lands with tests for its rejection paths,
  not just its happy path.
