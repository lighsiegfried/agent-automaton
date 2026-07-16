"""Phase 7A — local backup/restore: user data in, secrets NEVER out, round-trip."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backup_restore as br  # noqa: E402


def _make_storage(tmp: Path) -> Path:
    storage = tmp / "storage"
    (storage / "memory").mkdir(parents=True)
    (storage / "memory" / "fifi_memory.db").write_bytes(b"MEMORY-DB")
    (storage / "security").mkdir()
    (storage / "security" / "fifi_security.db").write_bytes(b"SALTED-VERIFIER-ONLY")
    (storage / "knowledge" / "files" / "ab").mkdir(parents=True)
    (storage / "knowledge" / "files" / "ab" / "doc.pdf").write_bytes(b"PDF")
    (storage / "logs").mkdir()
    (storage / "logs" / "app.log").write_text("noisy log")
    (storage / "memory" / "fifi_memory.db-wal").write_bytes(b"WAL")   # transient sidecar
    # secrets that must NEVER be backed up
    (storage / ".env").write_text("SECRET=xyz")
    (storage / "creds.key").write_bytes(b"PRIVATE")
    return storage


def test_plan_includes_data_excludes_secrets_and_transient(tmp_path):
    storage = _make_storage(tmp_path)
    included, excluded = br.plan_backup(storage)
    assert "memory/fifi_memory.db" in included
    assert "security/fifi_security.db" in included        # salted verifier only — safe
    assert "knowledge/files/ab/doc.pdf" in included
    # secrets + transient excluded
    assert ".env" in excluded and "creds.key" in excluded
    assert "logs/app.log" in excluded
    assert "memory/fifi_memory.db-wal" in excluded
    assert not any("env" in p or ".key" in p for p in included)


def test_is_secret_file():
    assert br.is_secret_file(Path("x/.env")) is True
    assert br.is_secret_file(Path("x/prod.env")) is True
    assert br.is_secret_file(Path("x/tls.pem")) is True
    assert br.is_secret_file(Path("x/data.db")) is False


def test_backup_writes_manifest_and_no_secrets(tmp_path):
    storage = _make_storage(tmp_path)
    dest = tmp_path / "out"
    manifest = br.backup(storage, dest, now_fn=lambda: datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
    root = Path(manifest["backup_root"])
    assert (root / "MANIFEST.json").exists()
    assert (root / "memory" / "fifi_memory.db").read_bytes() == b"MEMORY-DB"
    # no secret file exists anywhere in the backup tree
    assert not any(br.is_secret_file(p) for p in root.rglob("*") if p.is_file())
    assert manifest["included_count"] >= 3


def test_restore_round_trip(tmp_path):
    storage = _make_storage(tmp_path)
    dest = tmp_path / "out"
    manifest = br.backup(storage, dest, now_fn=lambda: datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
    target = tmp_path / "restored"
    res = br.restore(Path(manifest["backup_root"]), target)
    assert (target / "memory" / "fifi_memory.db").read_bytes() == b"MEMORY-DB"
    assert (target / "knowledge" / "files" / "ab" / "doc.pdf").read_bytes() == b"PDF"
    # the restore never re-creates a secret or the manifest
    assert not (target / ".env").exists() and not (target / "MANIFEST.json").exists()
    assert res["restored_count"] == manifest["included_count"]


def test_backup_missing_storage_is_empty(tmp_path):
    included, excluded = br.plan_backup(tmp_path / "nope")
    assert included == [] and excluded == []
