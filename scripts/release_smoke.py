"""Release smoke test (Phase 7A) — boot the API in-process and exercise the safe path.

Uses a FastAPI TestClient (no network, no real actions) to confirm a freshly-built
release actually serves: health/identity are green, a simulated command round-trips,
the tool registry is non-empty, and the safety-critical defaults hold (real Windows
tools OFF, nothing pending, confirmation required). Returns a structured result so it is
unit-tested (tests/test_release_smoke.py); the CLI exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:         # allow `python scripts/release_smoke.py`
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class SmokeResult:
    checks: list = field(default_factory=list)          # (name, ok, detail)

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def smoke(client=None) -> SmokeResult:
    """Run the smoke checks against ``client`` (a TestClient-like). Builds one against
    the real app when not injected, running the app lifespan (which loads tools and warms
    the stores) so the boot path is exercised exactly as in production."""
    if client is None:                       # pragma: no cover - real app wiring
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as booted:      # context manager runs startup/shutdown
            return _run_checks(booted)
    return _run_checks(client)


def _run_checks(client) -> SmokeResult:
    r = SmokeResult()

    health = client.get("/health").json()
    r.add("health_ok", health.get("status") == "ok", str(health.get("status")))
    r.add("real_tools_off", health.get("real_windows_tools") is False,
          f"real_windows_tools={health.get('real_windows_tools')}")

    identity = client.get("/identity").json()
    r.add("identity_has_agent", bool(identity.get("agent_name")), identity.get("agent_name", ""))

    tools = client.get("/tools").json()
    r.add("tools_registered", isinstance(tools, list) and len(tools) > 0, f"{len(tools)} tools")
    # destructive tools MAY exist in the registry (e.g. delete_file) — what matters is the
    # safety layer blocks them by default and gates sensitive ones behind confirmation.
    try:
        from app.core import safety
        from app.schemas.commands import SafetyLevel
        blocked = not safety.evaluate(SafetyLevel.DESTRUCTIVE, False).allowed
        gated = not safety.evaluate(SafetyLevel.SENSITIVE, False).allowed
        r.add("destructive_blocked_by_default", blocked)
        r.add("sensitive_needs_confirmation", gated)
    except Exception as exc:                                  # pragma: no cover
        r.add("destructive_blocked_by_default", False, str(exc))

    cmd = client.post("/command", json={"text": "open folder downloads"}).json()
    r.add("command_simulated", cmd.get("status") in ("simulated", "executed", "needs_confirmation"),
          str(cmd.get("status")))

    pending = client.get("/pending").json()
    r.add("nothing_pending_on_boot", pending.get("pending") is None, str(pending.get("pending")))

    return r


def main(argv=None) -> int:  # pragma: no cover - CLI wrapper
    result = smoke()
    for name, ok, detail in result.checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nrelease smoke: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
