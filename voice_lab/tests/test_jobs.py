"""Background job manager: transitions, GPU lock, cancellation, cleanup."""

import threading
import time

import pytest

from app import progress
from app.jobs import JobManager


def wait_for(predicate, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def wait_terminal(manager, job_id, timeout=10.0):
    assert wait_for(
        lambda: manager.get(job_id)["status"] in ("completed", "failed", "cancelled"),
        timeout,
    ), f"job never finished: {manager.get(job_id)}"
    return manager.get(job_id)


@pytest.fixture(autouse=True)
def _isolated_job_dirs(monkeypatch, tmp_path):
    """Job temp AND persisted metadata always land in tmp — never the repo."""
    import app.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_TMP_DIR", tmp_path / "jobs")
    monkeypatch.setattr(jobs_module, "JOBS_META_DIR", tmp_path / "jobs-meta")
    monkeypatch.setattr(jobs_module.gpu, "pre_job_snapshot", lambda: {})


@pytest.fixture
def manager(monkeypatch, tmp_path):
    import app.jobs as jobs_module

    monkeypatch.setattr(jobs_module.gpu, "gpu_snapshot", lambda: {
        "name": "NVIDIA GeForce RTX 5060 Ti", "total_gb": 16.0, "used_gb": 8.0, "free_gb": 8.0,
    })
    return JobManager(max_workers=3, light_slots=2)


# --- lifecycle -------------------------------------------------------------------------


def test_job_completes_with_result_and_stages(manager):
    def fn(job, tmp_dir):
        progress.report("loading", message="Cargando modelo…", cache_hit=True)
        progress.report("generating", current_item=1, total_items=3, percentage=33,
                        message="Generando variante 1 de 3…")
        return {"answer": 42}

    job = manager.submit("voice_design", fn, heavy=True, model="m", engine="qwen3_tts")
    assert job["status"] in ("queued", "preparing")
    final = wait_terminal(manager, job["job_id"])
    assert final["status"] == "completed"
    assert final["result"] == {"answer": 42}
    assert final["cache_hit"] is True
    assert final["total_items"] == 3
    assert final["gpu_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert final["started_at"] and final["updated_at"]
    assert final["elapsed_seconds"] is not None


def test_job_failure_carries_error_code(manager):
    class Boom(Exception):
        error_code = "insufficient_vram"

    def fn(job, tmp_dir):
        raise Boom("VRAM insuficiente: se necesitan ~5 GB")

    job = manager.submit("voice_design", fn, heavy=True)
    final = wait_terminal(manager, job["job_id"])
    assert final["status"] == "failed"
    assert final["error_code"] == "insufficient_vram"
    assert "VRAM" in final["error"]


def test_job_status_stages_follow_reports(manager):
    gate = threading.Event()
    seen = []

    def fn(job, tmp_dir):
        progress.report("downloading", message="Descargando…", cache_hit=False)
        gate.wait(5)
        return {}

    job = manager.submit("x", fn, heavy=False)
    assert wait_for(lambda: manager.get(job["job_id"])["status"] == "downloading")
    seen.append(manager.get(job["job_id"]))
    gate.set()
    wait_terminal(manager, job["job_id"])
    assert seen[0]["cache_hit"] is False
    assert seen[0]["message"] == "Descargando…"


# --- cancellation + cleanup -------------------------------------------------------------


def test_cancel_running_job_cleans_temp_dir(manager, tmp_path):
    started = threading.Event()

    def fn(job, tmp_dir):
        (tmp_dir / "half-baked.wav").write_bytes(b"x" * 10)
        started.set()
        for _ in range(200):
            progress.check_cancelled()
            time.sleep(0.02)
        return {}

    job = manager.submit("voice_design", fn, heavy=True)
    assert started.wait(5)
    job_tmp = tmp_path / "jobs" / job["job_id"]
    assert wait_for(lambda: job_tmp.exists())
    manager.cancel(job["job_id"])
    final = wait_terminal(manager, job["job_id"])
    assert final["status"] == "cancelled"
    assert not job_tmp.exists()  # temporary audio removed safely


def test_cancel_queued_job_settles_immediately(manager):
    gate = threading.Event()

    def slow(job, tmp_dir):
        gate.wait(10)
        return {}

    first = manager.submit("a", slow, heavy=True)
    second = manager.submit("b", lambda job, tmp: {}, heavy=True)  # waits for the lock
    cancelled = manager.cancel(second["job_id"])
    assert cancelled["status"] == "cancelled"
    gate.set()
    wait_terminal(manager, first["job_id"])


# --- GPU concurrency --------------------------------------------------------------------


def test_only_one_heavy_qwen_job_at_a_time(manager):
    running = []
    peak = []
    lock = threading.Lock()

    def fn(job, tmp_dir):
        with lock:
            running.append(1)
            peak.append(len(running))
        time.sleep(0.15)
        with lock:
            running.pop()
        return {}

    ids = [manager.submit("heavy", fn, heavy=True)["job_id"] for _ in range(3)]
    for job_id in ids:
        assert wait_terminal(manager, job_id)["status"] == "completed"
    assert max(peak) == 1  # strictly serialized


def test_light_jobs_run_concurrently_but_bounded(manager):
    peak = []
    running = []
    lock = threading.Lock()
    gate = threading.Event()

    def fn(job, tmp_dir):
        with lock:
            running.append(1)
            peak.append(len(running))
        gate.wait(3)
        with lock:
            running.pop()
        return {}

    ids = [manager.submit("light", fn, heavy=False)["job_id"] for _ in range(3)]
    assert wait_for(lambda: len(peak) >= 2)
    time.sleep(0.2)
    gate.set()
    for job_id in ids:
        wait_terminal(manager, job_id)
    assert max(peak) == 2  # the light semaphore bounds concurrency


def test_worker_status_reflects_activity(manager):
    assert manager.worker_status() == "idle"
    gate = threading.Event()

    def fn(job, tmp_dir):
        progress.report("loading", message="Cargando…")
        gate.wait(5)
        return {}

    job = manager.submit("x", fn, heavy=True)
    assert wait_for(lambda: manager.worker_status() in ("busy", "warming"))
    assert wait_for(lambda: manager.worker_status() == "warming")  # loading stage
    gate.set()
    wait_terminal(manager, job["job_id"])
    assert manager.worker_status() == "idle"


def test_active_lists_only_unfinished_jobs(manager):
    gate = threading.Event()
    job = manager.submit("x", lambda j, t: gate.wait(5) or {}, heavy=False)
    assert wait_for(lambda: any(j["job_id"] == job["job_id"] for j in manager.active()))
    gate.set()
    wait_terminal(manager, job["job_id"])
    assert all(j["job_id"] != job["job_id"] for j in manager.active())
