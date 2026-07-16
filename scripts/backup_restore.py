"""Local backup / restore for Fifi user data (Phase 7A).

Backs up the LOCAL user data under ``storage/`` — the personal-memory, tasks, schedules,
activity, Knowledge Vault, and security databases plus ingested document files — into a
timestamped folder, and restores it. It NEVER copies secrets: any ``.env`` / ``*.env`` /
``*.key`` / ``*.pem`` and the transient logs/runtime are excluded, and the copy is
refused if a secret-looking file would be included. The security database holds only a
SALTED password verifier (no recoverable plaintext), so it is safe user data to back up.

Pure planning (:func:`plan_backup`) is unit-tested; the copy/restore are thin file ops.
CLI:  python scripts/backup_restore.py backup [dest]   |   restore <backup_dir>
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"

# Never copied into a backup — credentials and transient/regenerable data.
_SECRET_SUFFIXES = {".env", ".key", ".pem", ".pfx", ".p12"}
_EXCLUDED_DIRS = {"logs", "runtime"}          # transient; regenerated at runtime
_EXCLUDED_SUFFIXES = {".log", ".corrupt"}     # transient
# SQLite sidecars are named "<db>.db-wal" / "<db>.db-shm", so match by name ending.
_TRANSIENT_ENDINGS = ("-wal", "-shm", ".wal", ".shm")


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    return name == ".env" or name.endswith(".env") or path.suffix.lower() in _SECRET_SUFFIXES


def _excluded(rel: Path) -> bool:
    if any(part in _EXCLUDED_DIRS for part in rel.parts):
        return True
    if rel.suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    if rel.name.lower().endswith(_TRANSIENT_ENDINGS):
        return True
    return False


def plan_backup(storage_dir: Path = STORAGE_DIR):
    """(included, excluded) lists of storage-relative POSIX paths. Pure and testable."""
    included, excluded = [], []
    if not storage_dir.exists():
        return included, excluded
    for path in sorted(storage_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(storage_dir)
        if is_secret_file(path):
            excluded.append(rel.as_posix())         # a secret must never be backed up
        elif _excluded(rel):
            excluded.append(rel.as_posix())
        else:
            included.append(rel.as_posix())
    return included, excluded


def backup(storage_dir: Path = STORAGE_DIR, dest_dir: Path | None = None, *,
           now_fn=lambda: datetime.now(timezone.utc)) -> dict:
    """Copy the included files into ``dest_dir/fifi-backup-<stamp>/`` and write a
    manifest. Returns the manifest dict."""
    dest_dir = Path(dest_dir or (PROJECT_ROOT / "backups"))
    stamp = now_fn().strftime("%Y%m%dT%H%M%SZ")
    root = dest_dir / f"fifi-backup-{stamp}"
    included, excluded = plan_backup(storage_dir)

    for rel in included:
        src = storage_dir / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # a final guard: never let a secret slip through even if the plan changed
        if is_secret_file(src):
            continue
        shutil.copy2(src, dst)

    manifest = {"created_at": now_fn().isoformat(timespec="seconds"),
                "source": str(storage_dir), "backup_root": str(root),
                "included": included, "excluded": excluded,
                "included_count": len(included), "excluded_count": len(excluded)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def restore(backup_root: Path, storage_dir: Path = STORAGE_DIR) -> dict:
    """Restore a backup folder's files back into ``storage_dir`` (overwriting). The
    MANIFEST.json and any secret file are skipped defensively."""
    backup_root = Path(backup_root)
    restored = []
    for path in sorted(backup_root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        if is_secret_file(path):
            continue
        rel = path.relative_to(backup_root)
        dst = storage_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        restored.append(rel.as_posix())
    return {"restored": restored, "restored_count": len(restored), "target": str(storage_dir)}


def main(argv=None) -> int:  # pragma: no cover - CLI wrapper
    import sys

    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] not in ("backup", "restore"):
        print("usage: backup_restore.py backup [dest] | restore <backup_dir>")
        return 2
    if args[0] == "backup":
        m = backup(dest_dir=Path(args[1]) if len(args) > 1 else None)
        print(f"backed up {m['included_count']} file(s), excluded {m['excluded_count']} "
              f"(secrets/transient) → {m['backup_root']}")
        return 0
    if len(args) < 2:
        print("restore needs a backup directory")
        return 2
    res = restore(Path(args[1]))
    print(f"restored {res['restored_count']} file(s) → {res['target']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
