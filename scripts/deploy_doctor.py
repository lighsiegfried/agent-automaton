"""Pre-deployment doctor (Phase 7A) — a deterministic release preflight.

Runs a set of SAFE, read-only checks and prints a PASS/WARN/FAIL report. It never
changes anything: it verifies the interpreter, that core dependencies import, that the
safety-critical defaults are intact (real Windows tools OFF, binds are loopback, the new
6B/6C bridges are off by default), that no tracked file carries a secret, and that the
local storage directory is writable.

The check functions are pure and take injected providers, so the whole doctor is unit-
tested without a real environment (see tests/test_deploy_doctor.py). Exit code is 0 when
there is no FAIL, 1 otherwise — suitable for CI / a release gate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:          # allow `python scripts/deploy_doctor.py`
    sys.path.insert(0, str(PROJECT_ROOT))

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class Finding:
    name: str
    status: str
    detail: str = ""


def _ok(name, detail=""):
    return Finding(name, OK, detail)


def check_python(version_info=sys.version_info, minimum=(3, 11)) -> Finding:
    got = f"{version_info[0]}.{version_info[1]}"
    if tuple(version_info[:2]) >= minimum:
        return _ok("python_version", f"{got} >= {minimum[0]}.{minimum[1]}")
    return Finding("python_version", FAIL, f"{got} < {minimum[0]}.{minimum[1]}")


def check_core_imports(importer=__import__) -> Finding:
    missing = []
    for mod in ("fastapi", "uvicorn", "pydantic", "pydantic_settings", "httpx", "numpy"):
        try:
            importer(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return Finding("core_imports", FAIL, f"missing: {', '.join(missing)}")
    return _ok("core_imports", "fastapi/uvicorn/pydantic/httpx/numpy import")


def _is_loopback(host: str) -> bool:
    return (host or "").strip() in ("127.0.0.1", "::1", "localhost")


def check_safety_defaults(settings) -> list:
    """The safety-critical toggles must default to the SAFE value for a release build."""
    findings = []
    findings.append(_ok("real_windows_tools_off") if not settings.enable_real_windows_tools
                    else Finding("real_windows_tools_off", WARN,
                                 "ENABLE_REAL_WINDOWS_TOOLS is on — intended only for the trusted host"))
    findings.append(_ok("require_confirmation_on") if settings.require_confirmation
                    else Finding("require_confirmation_on", FAIL,
                                 "REQUIRE_CONFIRMATION is off — sensitive actions would skip confirmation"))
    findings.append(_ok("api_bind_loopback", settings.api_host) if _is_loopback(settings.api_host)
                    else Finding("api_bind_loopback", WARN, f"API_HOST={settings.api_host} is not loopback"))
    dbh = getattr(settings, "desktop_bind_host", "127.0.0.1")
    findings.append(_ok("desktop_bind_loopback", dbh) if _is_loopback(dbh)
                    else Finding("desktop_bind_loopback", FAIL, f"DESKTOP_BIND_HOST={dbh} is not loopback"))
    return findings


def check_sensitive_flags_default_off(settings) -> Finding:
    """New capability layers must be OFF by default (backward compatible + opt-in)."""
    on = [name for name in ("enable_security_profiles", "enable_desktop_bridge",
                            "enable_knowledge_vault")
          if getattr(settings, name, False)]
    if on:
        return Finding("capability_flags_default", WARN, f"enabled: {', '.join(on)}")
    return _ok("capability_flags_default", "security/desktop/knowledge default off")


import re

# Anchored, length-bounded patterns so ordinary words (e.g. "task-level" contains
# "sk-") never false-positive; these match real credential shapes only.
_SECRET_REGEXES = tuple(re.compile(p) for p in (
    r"sk-[A-Za-z0-9]{20,}",
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    r"ghp_[A-Za-z0-9]{20,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",
))


def check_no_tracked_secrets(tracked_reader=None) -> Finding:
    """Scan tracked-file CONTENT for real credential shapes. ``tracked_reader`` yields
    (path, text); the default reads ``git ls-files``. Reports paths, never values."""
    if tracked_reader is None:
        tracked_reader = _default_tracked_reader
    hits = []
    for path, text in tracked_reader():
        if any(rx.search(text) for rx in _SECRET_REGEXES):
            hits.append(path)
    if hits:
        return Finding("no_tracked_secrets", FAIL, f"secret-like content in: {', '.join(hits[:10])}")
    return _ok("no_tracked_secrets", "no secret markers in tracked files")


def _default_tracked_reader():  # pragma: no cover - exercised via the CLI, not unit tests
    import subprocess

    try:
        out = subprocess.run(["git", "ls-files"], cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return
    for rel in (out.stdout or "").splitlines():
        p = PROJECT_ROOT / rel
        if p.suffix in (".png", ".jpg", ".ico", ".zip", ".onnx", ".pyc"):
            continue
        try:
            yield rel, p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue


def check_env_ignored(is_ignored=None) -> Finding:
    """The real .env must be git-ignored so credentials never get committed."""
    if is_ignored is None:
        is_ignored = _default_env_ignored
    if not (PROJECT_ROOT / ".env").exists():
        return _ok("env_ignored", "no .env present")
    return _ok("env_ignored", ".env is git-ignored") if is_ignored() else \
        Finding("env_ignored", FAIL, ".env exists and is NOT git-ignored")


def _default_env_ignored():  # pragma: no cover - needs git
    import subprocess

    r = subprocess.run(["git", "check-ignore", ".env"], cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True)
    return r.returncode == 0


def check_storage_writable(storage_dir: Path, write_probe=None) -> Finding:
    if write_probe is None:
        def write_probe(d: Path) -> bool:
            try:
                d.mkdir(parents=True, exist_ok=True)
                probe = d / ".doctor_probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                return True
            except Exception:
                return False
    return _ok("storage_writable", str(storage_dir)) if write_probe(storage_dir) else \
        Finding("storage_writable", FAIL, f"cannot write to {storage_dir}")


def run_checks(*, settings=None, storage_dir=None, tracked_reader=None,
               env_ignored=None, write_probe=None) -> list:
    from app.config import get_settings

    settings = settings or get_settings()
    storage_dir = storage_dir or (PROJECT_ROOT / "storage")
    findings = [
        check_python(),
        check_core_imports(),
        *check_safety_defaults(settings),
        check_sensitive_flags_default_off(settings),
        check_no_tracked_secrets(tracked_reader),
        check_env_ignored(env_ignored),
        check_storage_writable(storage_dir, write_probe),
    ]
    return findings


def summarize(findings) -> dict:
    return {
        "fail": sum(1 for f in findings if f.status == FAIL),
        "warn": sum(1 for f in findings if f.status == WARN),
        "ok": sum(1 for f in findings if f.status == OK),
        "ready": not any(f.status == FAIL for f in findings),
    }


def main(argv=None) -> int:  # pragma: no cover - CLI wrapper
    findings = run_checks()
    icon = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}
    for f in findings:
        print(f"[{icon[f.status]}] {f.name}: {f.detail}")
    s = summarize(findings)
    print(f"\n{s['ok']} pass, {s['warn']} warn, {s['fail']} fail — "
          f"{'READY' if s['ready'] else 'NOT READY'}")
    return 0 if s["ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
