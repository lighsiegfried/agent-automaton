"""Worker ownership, build compatibility, and recovery (Phase 3D.0.3a).

Pins the runtime's contract around port 8766:
- a healthy CURRENT worker is reused, never duplicated;
- a stale worker is stopped ONLY when the PID file proves we own it;
- an unrelated process on the port is never terminated;
- PID files are JSON, written atomically, with legacy plain-int support;
- only voice_lab/.venv may launch the worker, via the CWD-independent runner;
- voice-doctor exits nonzero when a job could not start;
- voice-restart stops the owned worker, verifies readiness and the store.

All HTTP, process, and netstat probes are stubbed — nothing real is touched.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_runtime  # noqa: E402

EXPECTED_BUILD = local_runtime.voice_lab_expected_build()
# Captured before the autouse fixture stubs it — the parser test needs the real one.
_REAL_PORT_OWNER_PID = local_runtime.port_owner_pid


@pytest.fixture(autouse=True)
def _isolated_pid_file(monkeypatch, tmp_path):
    """Never read the real PID file, port state, or process tree from tests."""
    monkeypatch.setattr(local_runtime, "VOICE_LAB_PID_FILE", tmp_path / "voice_lab.pid")
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: None)
    monkeypatch.setattr(local_runtime, "process_parent_pid", lambda pid: None)


def _args():
    class Args:
        no_ollama = False
        stop_ollama = False

    return Args()


# --- PID file ---------------------------------------------------------------------


def test_pid_file_is_atomic_json_with_build(monkeypatch):
    local_runtime.write_voice_lab_pid(12345)
    record = local_runtime.read_voice_lab_pid_record()
    assert record["pid"] == 12345
    assert record["build"] == EXPECTED_BUILD
    assert record["created_utc"]
    assert local_runtime.read_voice_lab_pid() == 12345
    # No leftover temp files from the atomic write.
    siblings = list(local_runtime.VOICE_LAB_PID_FILE.parent.glob("*.tmp"))
    assert siblings == []


def test_pid_file_accepts_legacy_plain_int():
    local_runtime.VOICE_LAB_PID_FILE.write_text("456", encoding="utf-8")
    assert local_runtime.read_voice_lab_pid() == 456


def test_pid_file_garbage_is_none():
    local_runtime.VOICE_LAB_PID_FILE.write_text("not-a-pid", encoding="utf-8")
    assert local_runtime.read_voice_lab_pid() is None


# --- port ownership ---------------------------------------------------------------


def test_port_owned_by_us_accepts_launcher_child(monkeypatch):
    """Windows venv launchers re-exec python as a CHILD; the child owning the
    socket still counts as ours when its parent is the recorded PID."""
    monkeypatch.setattr(
        local_runtime, "process_parent_pid", lambda pid: 4242 if pid == 8760 else None
    )
    assert local_runtime.port_owned_by_us(4242, 4242) is True
    assert local_runtime.port_owned_by_us(8760, 4242) is True  # proven child
    assert local_runtime.port_owned_by_us(7777, 4242) is False
    assert local_runtime.port_owned_by_us(None, 4242) is False
    assert local_runtime.port_owned_by_us(8760, None) is False


def test_stop_owned_worker_kills_proven_child_too(monkeypatch):
    local_runtime.write_voice_lab_pid(4242)
    killed = []
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: True)
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: 8760)
    monkeypatch.setattr(
        local_runtime, "process_parent_pid", lambda pid: 4242 if pid == 8760 else None
    )
    assert local_runtime.stop_owned_worker(out=lambda *a: None) is True
    assert killed == [4242, 8760]  # launcher AND its listening child
    assert local_runtime.read_voice_lab_pid() is None  # PID file cleared


def test_stop_owned_worker_never_kills_unrelated_owner(monkeypatch):
    local_runtime.write_voice_lab_pid(4242)
    killed = []
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: 7777)  # not ours
    assert local_runtime.stop_owned_worker(out=lambda *a: None) is False
    assert killed == []


def test_port_owner_pid_parses_netstat(monkeypatch):
    output = (
        "  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       111\n"
        "  TCP    127.0.0.1:8766         0.0.0.0:0              LISTENING       4242\n"
        "  TCP    127.0.0.1:56462        127.0.0.1:8766         SYN_SENT        27052\n"
    )

    class Result:
        stdout = output

    monkeypatch.setattr(local_runtime.subprocess, "run", lambda *a, **k: Result())
    assert _REAL_PORT_OWNER_PID(8766) == 4242
    assert _REAL_PORT_OWNER_PID(9999) is None


# --- reuse / recycle / refuse -------------------------------------------------------


def test_current_worker_is_reused(monkeypatch, capsys):
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": EXPECTED_BUILD},
    )
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: (_ for _ in ()).throw(AssertionError("must not start a duplicate")),
    )
    assert local_runtime._voice_lab_up() is True
    assert "already running" in capsys.readouterr().out


def test_stale_owned_worker_is_recycled(monkeypatch, tmp_path, capsys):
    """Old build on the port + our PID file proves ownership → stop + restart."""
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", tmp_path)
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": "3d.0.2-old"},
    )
    state = {"killed": None}
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 4242)
    monkeypatch.setattr(
        local_runtime, "port_owner_pid",
        lambda port: None if state["killed"] else 4242,
    )
    monkeypatch.setattr(
        local_runtime, "terminate_process", lambda pid: state.update(killed=pid)
    )
    started = {}
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: started.update(done=True) or object(),
    )
    monkeypatch.setattr(
        local_runtime, "wait_for_voice_lab_or_exit",
        lambda process, **k: ({"engines_available": {}, "active_profile": "fifi_luna"}, None),
    )
    assert local_runtime._voice_lab_up() is True
    assert state["killed"] == 4242  # ONLY the proven-owned pid
    assert started == {"done": True}
    assert "STALE" in capsys.readouterr().out


def test_unowned_stale_worker_is_refused_never_terminated(monkeypatch, capsys):
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": "3d.0.2-old"},
    )
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: 7777)
    killed = []
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    assert local_runtime._voice_lab_up() is False
    assert killed == []  # an unrelated process is NEVER touched
    assert "refusing" in capsys.readouterr().out


def test_mismatched_pid_file_never_kills_port_owner(monkeypatch, capsys):
    """PID file says 4242 but the port belongs to 7777 → no ownership proof."""
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": "3d.0.2-old"},
    )
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 4242)
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: 7777)
    killed = []
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    assert local_runtime._voice_lab_up() is False
    assert killed == []


# --- launcher isolation --------------------------------------------------------------


def test_worker_launch_refuses_foreign_interpreter(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "voice_lab_python", lambda: sys.executable)
    monkeypatch.setattr(
        local_runtime.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    assert local_runtime.start_voice_lab_process() is None
    assert "refusing" in capsys.readouterr().out


def test_worker_launch_uses_cwd_independent_runner(monkeypatch):
    venv_python = str(local_runtime.VOICE_LAB_DIR / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(local_runtime, "voice_lab_python", lambda: venv_python)
    monkeypatch.setattr(local_runtime, "ensure_dirs", lambda: None)
    monkeypatch.setattr(local_runtime, "write_voice_lab_pid", lambda pid: None)
    monkeypatch.setattr(
        local_runtime, "open", lambda *a, **k: __import__("io").BytesIO(), raising=False
    )
    captured = {}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        captured.update(cmd=cmd, cwd=cwd)

        class P:
            pid = 999

        return P()

    monkeypatch.setattr(local_runtime.subprocess, "Popen", fake_popen)
    process = local_runtime.start_voice_lab_process()
    assert process is not None
    # The CWD-independent runner, not `-m uvicorn app.main:app` (which resolves
    # `app` from the CWD and can import the WRONG project).
    assert captured["cmd"] == [venv_python, str(local_runtime.VOICE_LAB_RUNNER)]
    assert "uvicorn" not in captured["cmd"]


def test_expected_build_parsed_from_disk():
    assert EXPECTED_BUILD  # build.py exists and declares API_BUILD
    assert EXPECTED_BUILD == "3d.0.4"


# --- voice-doctor ---------------------------------------------------------------------


def _healthy_doctor_stubs(monkeypatch):
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: 4242)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 4242)
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": EXPECTED_BUILD, "worker": "idle"},
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_worker_info",
        lambda: {"api_build": EXPECTED_BUILD, "venv_isolated": True, "pid": 4242},
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_ready",
        lambda: {"ready": True, "profile_store": {"valid": True}, "environment_ok": True},
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_jobs_active", lambda: {"jobs": [], "worker": "idle"}
    )


def test_voice_doctor_healthy_exits_zero(monkeypatch, capsys):
    _healthy_doctor_stubs(monkeypatch)
    assert local_runtime.cmd_voice_doctor(_args()) == 0
    out = capsys.readouterr().out
    assert "(ours)" in out
    assert "MATCH" in out
    assert "Profile store" in out
    assert "Model cache" in out
    assert "System RAM" in out
    assert "GPU VRAM" in out
    assert "can start" in out


def test_voice_doctor_worker_down_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "port_owner_pid", lambda port: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "voice_lab_worker_info", lambda: None)
    assert local_runtime.cmd_voice_doctor(_args()) == 1
    out = capsys.readouterr().out
    assert "PROBLEMS" in out
    assert "CANNOT start" in out


def test_voice_doctor_version_mismatch_exits_nonzero(monkeypatch, capsys):
    _healthy_doctor_stubs(monkeypatch)
    monkeypatch.setattr(
        local_runtime, "voice_lab_health",
        lambda: {"status": "ok", "api_build": "3d.0.2-old", "worker": "idle"},
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_worker_info",
        lambda: {"api_build": "3d.0.2-old", "venv_isolated": True},
    )
    assert local_runtime.cmd_voice_doctor(_args()) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_voice_doctor_output_has_no_secrets(monkeypatch, capsys):
    _healthy_doctor_stubs(monkeypatch)
    local_runtime.cmd_voice_doctor(_args())
    out = capsys.readouterr().out
    for needle in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"):
        assert needle not in out.upper() or "SECRET" not in out.upper()


# --- voice-restart ----------------------------------------------------------------------


def test_voice_restart_stops_owned_and_verifies(monkeypatch, capsys):
    killed = []
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: 4242)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: True)
    monkeypatch.setattr(local_runtime, "terminate_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(local_runtime, "_voice_lab_up", lambda out=print: True)
    monkeypatch.setattr(
        local_runtime, "voice_lab_ready", lambda: {"ready": True, "worker": "idle"}
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_profiles",
        lambda: {"profiles": [{"name": "fifi_luna"}, {"name": "fifi_warm"}],
                 "active": "fifi_luna"},
    )
    assert local_runtime.cmd_voice_restart(_args()) == 0
    assert killed == [4242]
    out = capsys.readouterr().out
    assert "Active profile  : fifi_luna" in out
    assert "Saved profiles  : 2" in out


def test_voice_restart_fails_when_store_comes_back_empty(monkeypatch, capsys):
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "_voice_lab_up", lambda out=print: True)
    monkeypatch.setattr(
        local_runtime, "voice_lab_ready", lambda: {"ready": True, "worker": "idle"}
    )
    monkeypatch.setattr(
        local_runtime, "voice_lab_profiles", lambda: {"profiles": [], "active": None}
    )
    assert local_runtime.cmd_voice_restart(_args()) == 1
    assert "EMPTY" in capsys.readouterr().out


def test_voice_restart_fails_when_not_ready(monkeypatch):
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "_voice_lab_up", lambda out=print: True)
    monkeypatch.setattr(local_runtime, "voice_lab_ready", lambda: {"ready": False})
    assert local_runtime.cmd_voice_restart(_args()) == 1


# --- startup watchdog (Phase 3D.0.3a hotfix) --------------------------------------------


def _watchdog_env(monkeypatch, tmp_path, outcomes):
    """Wire _voice_lab_up for a fresh start whose wait outcomes are scripted."""
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(local_runtime, "VOICE_LAB_DIR", tmp_path)
    monkeypatch.setattr(local_runtime, "voice_lab_health", lambda: None)
    monkeypatch.setattr(local_runtime, "read_voice_lab_pid", lambda: None)
    monkeypatch.setattr(local_runtime, "process_alive", lambda pid: False)
    monkeypatch.setattr(local_runtime, "_tail_worker_log", lambda n=12: ["last line"])
    starts = []
    monkeypatch.setattr(
        local_runtime, "start_voice_lab_process",
        lambda: starts.append(1) or object(),
    )
    results = iter(outcomes)
    monkeypatch.setattr(
        local_runtime, "wait_for_voice_lab_or_exit", lambda process, **k: next(results)
    )
    return starts


def test_watchdog_reports_exit_and_restarts_at_most_once(monkeypatch, tmp_path, capsys):
    starts = _watchdog_env(
        monkeypatch, tmp_path,
        [(None, 3221225477), (None, 3221225477)],  # segfault twice
    )
    assert local_runtime._voice_lab_up() is False
    out = capsys.readouterr().out
    assert "EXITED during startup (exit code 3221225477)" in out
    assert "log: last line" in out  # last safe log lines shown
    assert "max one automatic restart" in out
    assert "NOT restarting again" in out
    assert starts == [1, 1]  # exactly one automatic restart, never a loop


def test_watchdog_recovers_when_single_restart_succeeds(monkeypatch, tmp_path, capsys):
    starts = _watchdog_env(
        monkeypatch, tmp_path,
        [(None, 1), ({"engines_available": {}, "active_profile": "fifi_luna"}, None)],
    )
    assert local_runtime._voice_lab_up() is True
    assert starts == [1, 1]
    assert "fifi_luna" in capsys.readouterr().out


def test_watchdog_no_restart_when_worker_merely_slow(monkeypatch, tmp_path, capsys):
    """No exit code (still running, just silent) → no automatic restart."""
    starts = _watchdog_env(monkeypatch, tmp_path, [(None, None)])
    assert local_runtime._voice_lab_up() is False
    assert starts == [1]
    assert "did not answer /health" in capsys.readouterr().out


def test_new_subcommands_are_wired():
    parser = local_runtime.build_parser()
    assert parser.parse_args(["voice-doctor"]).func is local_runtime.cmd_voice_doctor
    assert parser.parse_args(["voice-restart"]).func is local_runtime.cmd_voice_restart


def test_profile_store_state_reads_real_store():
    """Direct, worker-independent store view used by voice-doctor."""
    state = local_runtime._profile_store_state()
    assert state["profile_count"] >= 1
    assert state["valid"] is True or state["error"]
