"""Phase 7A — the release smoke test (against the real app via TestClient)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import release_smoke  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_smoke_passes_on_default_build():
    result = release_smoke.smoke(client=client)
    failed = [name for name, ok, _ in result.checks if not ok]
    assert result.passed, f"smoke failures: {failed}"


def test_smoke_checks_cover_safety_defaults():
    result = release_smoke.smoke(client=client)
    names = {name for name, _, _ in result.checks}
    assert {"health_ok", "real_tools_off", "tools_registered",
            "destructive_blocked_by_default", "sensitive_needs_confirmation",
            "nothing_pending_on_boot"} <= names


def test_smoke_result_fails_if_a_check_fails():
    r = release_smoke.SmokeResult()
    r.add("x", True)
    r.add("y", False, "boom")
    assert r.passed is False
