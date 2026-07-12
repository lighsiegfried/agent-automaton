"""Canonical repository paths (Phase 3D.0.3a, Part A).

The worker, the profile store, and the models directory must resolve from the
code's own location — NEVER from the process CWD. These tests launch the real
resolver (run_worker.py --print-paths) from four different directories and
require identical answers, and they prove no second profile store appears
under a foreign CWD.
"""

import json
import subprocess
import sys
from pathlib import Path

from conftest import PROJECT_ROOT, VOICE_LAB_ROOT

RUNNER = VOICE_LAB_ROOT / "run_worker.py"


def _print_paths(cwd: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--print-paths"],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_worker_discovers_same_store_from_any_cwd(tmp_path):
    cwds = [PROJECT_ROOT, VOICE_LAB_ROOT, PROJECT_ROOT / "scripts", tmp_path]
    snapshots = [_print_paths(cwd) for cwd in cwds]
    reference = snapshots[0]
    assert reference["profiles_dir"] == "config/voices/profiles"
    assert reference["active_path"] == "config/voices/active.json"
    assert reference["profiles"], "the shared store must contain profiles"
    for snapshot in snapshots[1:]:
        assert snapshot == reference  # identical store, repo, build — everywhere


def test_no_second_store_created_under_foreign_cwd(tmp_path):
    _print_paths(tmp_path)
    assert not (tmp_path / "config").exists()
    assert not (tmp_path / "voice_lab").exists()
    assert not (tmp_path / "storage").exists()


def test_active_profile_visible_regardless_of_cwd(tmp_path):
    """The live active.json (if any) is discovered from a foreign CWD too."""
    snapshot = _print_paths(tmp_path)
    active_path = PROJECT_ROOT / "config" / "voices" / "active.json"
    if active_path.exists():
        expected = json.loads(active_path.read_text(encoding="utf-8")).get("profile")
        assert snapshot["active_profile"] == expected
        assert snapshot["store_valid"] is True
    else:
        assert snapshot["active_profile"] is None


def test_print_paths_output_is_path_free(tmp_path):
    """Diagnostics stay safe to share: relative paths and fingerprints only."""
    raw = subprocess.run(
        [sys.executable, str(RUNNER), "--print-paths"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    ).stdout
    assert str(PROJECT_ROOT) not in raw
    assert ":\\" not in raw
