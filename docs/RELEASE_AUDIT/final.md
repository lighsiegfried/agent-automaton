# Release readiness — final verdict (Phase 7A)

_Date: 2026-07-15 · branch `main`._

## Verdict: **NOT READY — one blocker: the working tree is uncommitted**

The software itself is in excellent shape: the full suite is **1165 passed, 1 skipped**,
the safety controls are intact, dependencies are consistent, Docker is hardened, and no
secret is tracked. The **only** thing standing between this tree and a release is that
the bulk of the product (Phases ~4B–6C) is **not committed to git**. A release cannot be
cut from an uncommitted working tree. Once the commit + untrack steps below are done, the
verdict flips to **READY**.

## What is green (verified this audit)

| Area | Status |
| --- | --- |
| Test suite | ✅ 1165 passed, 1 skipped |
| `deploy_doctor` (defaults) | ✅ READY (0 fail) |
| `release_smoke` (lifecycle boot) | ✅ health, tools, safety gating, no pending |
| Safety defaults | ✅ real tools off, confirmation required, destructive blocked, sensitive gated |
| Central authorization (6B) | ✅ default-deny gate, no profile bypasses invariants |
| Desktop bridge (6C) | ✅ loopback-only, token-gated, sanitized, no generic confirm |
| Secrets | ✅ none tracked; `.env` git-ignored; no cloud creds to rotate |
| Dependencies | ✅ `pip check` clean; heavy deps optional/lazy |
| Docker | ✅ simulated-only image, profile-gated, loopback Ollama, `.env`/models/storage excluded |
| Backups | ✅ `backup_restore.py` copies user data, excludes secrets + sidecars, round-trips |

## Blocker — required before release (operator runs these; not auto-done)

1. **Commit the working tree.** 91 untracked files + 27 modified are real, tested source.
   From a branch (not directly on `main` if you gate releases):
   ```bash
   git add -A
   git status                     # review — confirm .env is NOT listed (it is git-ignored)
   git rm -r --cached graphify-out graphify-out.zip   # untrack cruft (kept on disk)
   git commit -m "Phases 4B–6C: browser, integrations, memory, tasks, schedules,
                  activity, knowledge vault, security profiles, desktop app + audit"
   ```
2. **Re-run the gates on the committed tree:**
   ```bash
   python scripts/deploy_doctor.py     # expect READY (0 fail)
   python scripts/release_smoke.py     # expect PASS
   python -m pytest                    # expect 1165 passed, 1 skipped
   ```

## Advisories (not blockers)

- **`ENABLE_REAL_WINDOWS_TOOLS`** is `true` in this host's `.env`. That is intended for the
  trusted Windows host, but a shipped/default build must keep it **off** (the code default
  and the test suite both enforce off). `deploy_doctor` reports this as a WARN, and
  `release_smoke` run against a real-tools-on host will flag `real_tools_off` — expected.
- **Take a backup before first release** so a bad deploy can't lose local data:
  `python scripts/backup_restore.py backup`.
- Optionally delete tool caches (`__pycache__`, `.pytest_cache`) — see `deletion-manifest.md`.
  Do **not** run `git clean -xd` until after the commit (it would remove untracked work).

## Sign-off checklist

- [ ] Working tree committed; `git status` clean.
- [ ] `graphify-out/` + `.zip` untracked.
- [ ] `deploy_doctor.py` → READY on the committed tree.
- [ ] `release_smoke.py` → PASS.
- [ ] `pytest` → 1165 passed / 1 skipped.
- [ ] `ENABLE_REAL_WINDOWS_TOOLS` set intentionally for the target host.
- [ ] Backup taken.

When every box is checked, this release is **READY**.
