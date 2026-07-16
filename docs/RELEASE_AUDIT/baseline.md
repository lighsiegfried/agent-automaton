# Release audit — baseline report (Phase 7A)

_Date: 2026-07-15 · branch `main` · read-only inventory (no code changed to produce this)._

## 1. Repository shape

| Metric | Value |
| --- | --- |
| Tracked files | 302 |
| Untracked files (excl. git-ignored) | 91 |
| Modified (tracked, uncommitted) | 27 |
| Untracked `app/` + `desktop/` modules | 20 |
| Test suite | **1165 passed, 1 skipped** |
| `pip check` | No broken requirements |
| TODO/FIXME/XXX/HACK in `app/ desktop/ scripts/` | 0 |
| Tracked files containing credential shapes | **0** (rigorous regex scan) |

Top-level layout: `app/` (API + all subsystems), `desktop/` (Phase 6C PySide6 client),
`scripts/` (host runtime, tray, doctors), `voice_lab/` + `wake_training/` (isolated
workspaces with their own venvs), `config/`, `docs/`, `models/`, `storage/`,
`compose.yml` + `Dockerfile`.

## 2. Uncommitted work — the headline finding

The working tree contains a large amount of **complete, tested, but uncommitted** code.
Whole subsystems are untracked, including:

- `app/knowledge/`, `app/security/`, `app/uibridge/`, `app/integrations/`,
  `app/memory/`, `app/schedules/`, `app/tasks/`, `app/text/`, `app/browser/`,
  `app/activity/`, `app/core/{conversation,errors,eventbus,nl,pending,spoken}.py`
- `desktop/` (entire Phase 6C client)
- ~60 `tests/test_*.py` files
- new `scripts/` (tray, autostart, wake calibration, deploy_doctor, release_smoke,
  backup_restore) and `requirements-*.txt`

The last commits (`d8ab099 fix backend to files`, `137d3f0 backend with voice`, …) predate
Phases ~4B–6C. **This is the single release blocker** (see `final.md`): the code is green,
but a release cannot be cut from an uncommitted tree. No auto-commit was performed.

## 3. Dependencies

Six requirement sets, layered and optional-by-design:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Core API + Knowledge Vault extraction (fastapi, uvicorn, pydantic, httpx, numpy, pypdf, python-docx). |
| `requirements-voice.txt` | STT/TTS/audio (optional). |
| `requirements-wakeword.txt` | Wake-word runtime (optional). |
| `requirements-browser.txt` | Playwright browser automation (optional). |
| `requirements-desktop.txt` | Push-to-talk + tray (pystray/Pillow/plyer). |
| `requirements-desktop-ui.txt` | Phase 6C desktop app (PySide6, httpx). |

`pip check` is clean. Heavy/optional deps (torch, playwright, PySide6, pywinauto) are
lazy-imported, so the core API and its tests run without them.

## 4. Docker / deployment

Already well-hardened; no changes required:

- `Dockerfile` builds a **simulated-only** API (`ENABLE_REAL_WINDOWS_TOOLS=false` pinned);
  containers cannot touch the Windows desktop (enforced in code too).
- `compose.yml` is **profile-gated** (`sim`, `llm`) so an implicit `up` never shadows the
  real host API; Ollama binds **127.0.0.1 only** and persists to the named volume
  `ollama-models` (**protected** — never delete).
- `.dockerignore` excludes `.env`, `models/`, `storage/`, `.git/`, caches, and
  `graphify-out*` from every build context.

## 5. Secrets

- **No credential shapes in any tracked file** (regex scan: `sk-…`, `AKIA…`, `ghp_…`,
  `xox[baprs]-…`, PEM private-key headers, `AIza…`).
- `.env` exists locally, is **git-ignored** (`git check-ignore .env` ✓), and holds only
  local runtime config (Ollama URL, model names, voice/PTT settings, feature flags) —
  no cloud API keys. This is a local-first app; there are no third-party credentials to
  rotate. If any operator later adds a key to `.env`, it stays ignored and out of logs.

## 6. Protected local data (never delete — see `deletion-manifest.md`)

| Path | Contents |
| --- | --- |
| `storage/memory,tasks,schedules,activity,knowledge,security` | User SQLite DBs (memory, tasks, schedules, activity, Knowledge Vault metadata, salted security verifier). |
| `storage/knowledge/files/` | Ingested document originals. |
| `storage/browser/` | Isolated Chromium profile. |
| `models/wake_words/*.onnx`, `*.metadata.json`, `backups/` | Wake-word models. |
| `config/voices/`, `voice_lab/models/` | Voice profiles + downloaded weights. |
| Docker volume `ollama-models` | LLM weights. |

## 7. New release tooling (Phase 7A, tested)

- `scripts/deploy_doctor.py` — deterministic preflight (interpreter, imports, safety
  defaults, loopback binds, no tracked secrets, `.env` ignored, storage writable).
- `scripts/release_smoke.py` — boots the app via its lifespan and checks health/identity,
  tool registry, safety gating (destructive blocked, sensitive gated), no pending on boot.
- `scripts/backup_restore.py` — backs up `storage/` user data **excluding secrets and
  transient sidecars**; round-trip restore.

Live run on this dev host: `deploy_doctor` → **READY** (1 warn: `ENABLE_REAL_WINDOWS_TOOLS`
is on, which is intended on the trusted host, not the shipped default).
