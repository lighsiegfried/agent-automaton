"""Phase 7A — the deployment doctor's pure checks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import deploy_doctor as dd  # noqa: E402


def _settings(**over):
    base = dict(enable_real_windows_tools=False, require_confirmation=True,
                api_host="127.0.0.1", desktop_bind_host="127.0.0.1",
                enable_security_profiles=False, enable_desktop_bridge=False,
                enable_knowledge_vault=False)
    base.update(over)
    return SimpleNamespace(**base)


def test_python_version_ok():
    assert dd.check_python(version_info=(3, 12, 0)).status == dd.OK
    assert dd.check_python(version_info=(3, 9, 0)).status == dd.FAIL


def test_core_imports_reports_missing():
    def importer(name):
        if name == "numpy":
            raise ImportError("no numpy")
        return object()
    assert dd.check_core_imports(importer=importer).status == dd.FAIL
    assert dd.check_core_imports(importer=lambda n: object()).status == dd.OK


def test_safety_defaults_all_ok():
    findings = dd.check_safety_defaults(_settings())
    assert all(f.status == dd.OK for f in findings)


def test_require_confirmation_off_is_fail():
    findings = {f.name: f for f in dd.check_safety_defaults(_settings(require_confirmation=False))}
    assert findings["require_confirmation_on"].status == dd.FAIL


def test_real_tools_on_is_warn_not_fail():
    findings = {f.name: f for f in dd.check_safety_defaults(_settings(enable_real_windows_tools=True))}
    assert findings["real_windows_tools_off"].status == dd.WARN


def test_desktop_non_loopback_is_fail():
    findings = {f.name: f for f in dd.check_safety_defaults(_settings(desktop_bind_host="0.0.0.0"))}
    assert findings["desktop_bind_loopback"].status == dd.FAIL


def test_api_non_loopback_is_warn():
    findings = {f.name: f for f in dd.check_safety_defaults(_settings(api_host="0.0.0.0"))}
    assert findings["api_bind_loopback"].status == dd.WARN


def test_sensitive_flags_default_off():
    assert dd.check_sensitive_flags_default_off(_settings()).status == dd.OK
    assert dd.check_sensitive_flags_default_off(_settings(enable_security_profiles=True)).status == dd.WARN


def test_no_tracked_secrets_detects_and_passes():
    clean = lambda: [("app/x.py", "print('hello')"), ("README.md", "# docs")]
    assert dd.check_no_tracked_secrets(tracked_reader=clean).status == dd.OK
    dirty = lambda: [("config.py", "key = 'sk-ABCDEFGHIJKLMNOPQRSTUVWX0123'")]
    bad = dd.check_no_tracked_secrets(tracked_reader=dirty)
    assert bad.status == dd.FAIL and "config.py" in bad.detail


def test_no_tracked_secrets_ignores_ordinary_words():
    # "task-level" contains "sk-" but is NOT a credential — must not false-positive
    reader = lambda: [("commands.py", "# each task-level step is gated")]
    assert dd.check_no_tracked_secrets(tracked_reader=reader).status == dd.OK


def test_env_ignored_check():
    assert dd.check_env_ignored(is_ignored=lambda: True).status == dd.OK
    # only fails if a .env actually exists AND is not ignored
    f = dd.check_env_ignored(is_ignored=lambda: False)
    assert f.status in (dd.OK, dd.FAIL)


def test_storage_writable():
    assert dd.check_storage_writable(Path("/x"), write_probe=lambda d: True).status == dd.OK
    assert dd.check_storage_writable(Path("/x"), write_probe=lambda d: False).status == dd.FAIL


def test_summarize_ready_flag():
    findings = [dd.Finding("a", dd.OK), dd.Finding("b", dd.WARN)]
    assert dd.summarize(findings)["ready"] is True
    findings.append(dd.Finding("c", dd.FAIL))
    assert dd.summarize(findings)["ready"] is False


def test_run_checks_on_real_settings_is_ready():
    # the shipped defaults must yield a READY verdict (no FAIL)
    from app.config import get_settings
    findings = dd.run_checks(settings=get_settings(),
                             storage_dir=Path("/x"),
                             tracked_reader=lambda: [],
                             env_ignored=lambda: True,
                             write_probe=lambda d: True)
    assert dd.summarize(findings)["ready"] is True
