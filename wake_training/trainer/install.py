"""Install a trained candidate as the active runtime wake model — safely.

Guarantees (phase requirements):
- The exported ONNX is validated for runtime I/O compatibility BEFORE anything is
  touched. An invalid model is refused; the active model is never replaced by one
  that fails validation.
- Replacing an already-installed model is never silent: it requires ``force`` and
  always writes a versioned backup (model + its metadata) first.
- The copy into ``models/wake_words/fifi.onnx`` is atomic (temp file in the same
  directory + os.replace), and the metadata JSON is written atomically too.
- ``rollback`` restores the most recent (or a named) backup, so a bad install is
  reversible.

Every filesystem location is overridable so this is testable entirely under tmp.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import metadata as meta_mod
from . import onnx_validate
from . import paths


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_copy(source: str | Path, target: str | Path) -> None:
    """Copy bytes to a temp file in the target's dir, then os.replace into place."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fifi-onnx-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def list_backups(backups_dir: str | Path | None = None) -> list[Path]:
    backups_dir = Path(backups_dir or paths.BACKUPS_DIR)
    if not backups_dir.is_dir():
        return []
    return sorted(backups_dir.glob("fifi-*.onnx"), key=lambda p: p.name)


def _backup_current(target: Path, metadata_target: Path, backups_dir: Path, stamp: str) -> dict:
    """Copy the current active model + metadata into backups/. No-op if absent."""
    if not target.is_file():
        return {}
    digest8 = meta_mod.sha256_file(target)[:8]
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_model = backups_dir / f"fifi-{stamp}-{digest8}.onnx"
    _atomic_copy(target, backup_model)
    record = {"model": str(backup_model)}
    if metadata_target.is_file():
        backup_meta = backups_dir / f"fifi-{stamp}-{digest8}.metadata.json"
        _atomic_copy(metadata_target, backup_meta)
        record["metadata"] = str(backup_meta)
    return record


def install_model(
    source: str | Path,
    *,
    candidate: str,
    phrase: str,
    threshold: float,
    metrics: dict | None = None,
    provenance: dict | None = None,
    verifier: dict | None = None,
    force: bool = False,
    target: str | Path | None = None,
    metadata_target: str | Path | None = None,
    backups_dir: str | Path | None = None,
    validate_fn=onnx_validate.validate_onnx,
    stamp: str | None = None,
) -> dict:
    """Validate + atomically install ``source`` as the active wake model.

    Returns ``{"status": "ok"|"refused"|"error", ...}``. ``status == "refused"``
    means a policy gate stopped a *safe* no-op (model already installed without
    force, or validation failed) — the active model is untouched in every
    non-ok case.
    """
    source = Path(source)
    target = Path(target or paths.INSTALL_TARGET)
    metadata_target = Path(metadata_target or paths.METADATA_TARGET)
    backups_dir = Path(backups_dir or paths.BACKUPS_DIR)

    if not source.is_file():
        return {"status": "error", "message": f"source model not found: {source}"}

    validation = validate_fn(source)
    if not validation.get("ok"):
        return {
            "status": "refused",
            "reason": "validation_failed",
            "message": "the exported model failed ONNX I/O validation — the active "
                       "model was NOT replaced",
            "validation": validation,
        }

    already_installed = target.is_file()
    if already_installed and not force:
        return {
            "status": "refused",
            "reason": "already_installed",
            "message": f"a wake model is already installed at {target.name}; pass "
                       "force=True to replace it (a versioned backup is kept)",
            "active_hash": meta_mod.sha256_file(target)[:12],
        }

    stamp = stamp or _utc_stamp()
    backup = _backup_current(target, metadata_target, backups_dir, stamp)

    # Atomic replace only AFTER validation passed and a backup was taken.
    _atomic_copy(source, target)
    model_hash = meta_mod.sha256_file(target)

    document = meta_mod.build_metadata(
        model_path=target,
        candidate=candidate,
        phrase=phrase,
        threshold=threshold,
        metrics=metrics,
        provenance={**(provenance or {}), "source_path": str(source)},
        verifier=verifier,
        model_hash=model_hash,
    )
    meta_mod.write_metadata(metadata_target, document)

    return {
        "status": "ok",
        "candidate": candidate,
        "target": str(target),
        "metadata": str(metadata_target),
        "hash": model_hash,
        "size_bytes": target.stat().st_size,
        "replaced": already_installed,
        "backup": backup,
        "validation_summary": validation.get("summary", {}),
    }


def rollback(
    backup_name: str | None = None,
    *,
    target: str | Path | None = None,
    metadata_target: str | Path | None = None,
    backups_dir: str | Path | None = None,
) -> dict:
    """Restore the most recent (or a named) backup as the active model."""
    target = Path(target or paths.INSTALL_TARGET)
    metadata_target = Path(metadata_target or paths.METADATA_TARGET)
    backups_dir = Path(backups_dir or paths.BACKUPS_DIR)

    backups = list_backups(backups_dir)
    if not backups:
        return {"status": "error", "message": "no backups to roll back to"}
    if backup_name:
        chosen = next((b for b in backups if b.name == backup_name), None)
        if chosen is None:
            return {"status": "error", "message": f"backup not found: {backup_name}"}
    else:
        chosen = backups[-1]  # most recent by timestamped name

    _atomic_copy(chosen, target)
    backup_meta = chosen.parent / (chosen.stem + ".metadata.json")
    restored_meta = False
    if backup_meta.is_file():
        _atomic_copy(backup_meta, metadata_target)
        restored_meta = True
    return {
        "status": "ok",
        "restored_from": chosen.name,
        "target": str(target),
        "hash": meta_mod.sha256_file(target),
        "metadata_restored": restored_meta,
    }
