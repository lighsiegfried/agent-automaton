# Cleanup / deletion manifest (Phase 7A)

Every item is classified **SAFE TO REMOVE**, **UNTRACK ONLY**, or **PROTECTED**. Nothing
here was deleted automatically — the destructive commands are listed for the operator to
run deliberately. No user data, model, profile, or backup is ever a removal candidate.

## PROTECTED — never delete (contains user data or model weights)

| Path | Why |
| --- | --- |
| `storage/memory/`, `storage/tasks/`, `storage/schedules/`, `storage/activity/` | Personal memory, tasks, schedules, activity DBs. |
| `storage/knowledge/` (DB + `files/`) | Knowledge Vault metadata **and ingested document originals**. |
| `storage/security/fifi_security.db` | Salted password verifier, lockout + elevation state. |
| `storage/browser/` | Isolated Chromium profile (logins, cookies). |
| `storage/assistant.db`, `storage/runtime/` | Command log + persistent runtime state. |
| `models/wake_words/*.onnx`, `*.metadata.json`, `backups/` | Wake-word models + versioned backups. |
| `config/voices/`, `voice_lab/models/` | Voice profiles + downloaded TTS/STT weights. |
| Docker volume `ollama-models` | LLM weights (tens of GB). **Do not** `docker volume rm`. |
| `backups/` | Local backups produced by `scripts/backup_restore.py`. |

These are all covered by `.gitignore` (`storage/*`, `models/wake_words/*.onnx`, …) and
`.dockerignore`, so they are already never committed and never enter a build context.

## SAFE TO REMOVE (regenerable, no history value)

| Path | What | Command |
| --- | --- | --- |
| `**/__pycache__/`, `*.pyc` | Byte-code caches (git-ignored). | `git clean -Xd -e '!storage/**' -n` then drop `-n` |
| `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` | Tool caches (git-ignored). | delete the folder |
| `_live*.py` | Ad-hoc live-validation scratch files. | already removed; now git-ignored |

`git clean -Xd` removes only **git-ignored** files (never tracked or untracked-but-wanted
work). Run with `-n` (dry-run) first; keep `storage/`, `models/`, `backups/` — they are
git-ignored and would be swept, so exclude them or run `git clean` from a clean subtree.
**Recommended instead:** delete caches manually and leave `git clean` alone here, because
so much valid work is currently untracked (see below).

## UNTRACK ONLY (tracked cruft — remove from git, keep on disk)

`graphify-out/` (6.6 MB) and `graphify-out.zip` (469 KB) are graphify skill output that was
committed **before** the ignore rule existed. They are the largest tracked files and carry
no source value. `.gitignore` now covers them, but git keeps tracking already-committed
paths. Untrack (do **not** delete the files) with:

```bash
git rm -r --cached graphify-out graphify-out.zip
# then commit the removal together with the rest of the working tree (see final.md)
```

## DO NOT auto-clean while work is uncommitted

The working tree has 91 untracked files that are **real, tested source** (Phases ~4B–6C).
Do **not** run a broad `git clean -xd` (lower-case `-x` removes untracked files too) — it
would delete unmerged work. Commit first (see `final.md`), then clean caches if desired.
