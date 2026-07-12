"""Background jobs — long Voice Lab work off the event loop, with real progress.

Rules (Phase 3D.0.3):
- The FastAPI event loop never runs model work: jobs execute in a small,
  bounded thread pool. /health stays fast while anything runs.
- ONE heavy Qwen GPU job at a time (a global lock); light jobs (Kokoro / SAPI
  previews) share a small semaphore of their own.
- Every job gets a unique temp directory; cancellation and failure clean it.
- Progress comes from the engines themselves via app.progress.report() —
  stages like downloading/loading/warming/generating, with cache and VRAM
  facts attached. A cached model explicitly reports that no download happens.
- Errors carry a category code (queue_timeout, load_failed, generation_failed,
  cancelled, insufficient_vram, ...) — never a generic "worker unavailable".
"""

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app import gpu, progress
from app.config import CACHE_DIR, JOBS_META_DIR, get_settings

JOBS_TMP_DIR = CACHE_DIR / "jobs"

# Job statuses that are also progress stages (finer stages go in current_step).
# 3D.0.4 canonical names first; the 3D.0.3a names remain valid aliases.
STAGE_STATUSES = {
    "resource_preflight", "checking_cache", "cached", "loading_from_disk",
    "moving_to_gpu", "warming_cuda", "validating_synthesis", "generating",
    "validating_audio", "saving",
    "preparing", "downloading", "loading", "transferring", "warming", "freezing",
}
# Stages that count as "still loading a model" for worker_status + timeouts.
LOADING_STAGES = {
    "checking_cache", "cached", "loading_from_disk", "moving_to_gpu",
    "warming_cuda", "downloading", "loading", "transferring", "warming",
}
# "interrupted" = the worker restarted/crashed while the job was in flight —
# it is terminal; a restarted worker never pretends to resume GPU work.
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}

QUEUE_TIMEOUT_SECONDS = 900.0
HEARTBEAT_SECONDS = 2.0

# Identity of THIS worker process. The UI compares it across reconnects: a
# different instance id means the worker restarted and in-flight jobs died.
WORKER_INSTANCE_ID = uuid.uuid4().hex[:12]
WORKER_STARTED_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(
        self, max_workers: int = 4, light_slots: int = 2, heavy_slots: int = 1
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="voicelab-job"
        )
        # Bounded heavy-job worker (VOICE_LAB_HEAVY_JOB_CONCURRENCY, default 1):
        # one Qwen GPU job at a time, so a single model instance ever exists.
        self._heavy_lock = threading.BoundedSemaphore(max(1, heavy_slots))
        self._light_sem = threading.BoundedSemaphore(light_slots)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        # RLock: cancel() legitimately reaches _finish() while already holding
        # the lock (a plain Lock deadlocked the whole worker there).
        self._registry_lock = threading.RLock()
        self._consecutive_failures = 0
        self._recover_persisted()

    # -- persistence (Phase 3D.0.3a hotfix) ---------------------------------------------

    def _persist(self, job: dict[str, Any]) -> None:
        """Atomically persist the job's PUBLIC metadata (never tensors/models).

        This is the crash record: if the process dies mid-load (even a native
        segfault), the last persisted stage tells the next worker — and the
        user — exactly where it stopped."""
        from app.profiles.manager import atomic_write_json

        public = {k: v for k, v in job.items() if not k.startswith("_")}
        try:
            atomic_write_json(JOBS_META_DIR / f"{job['job_id']}.json", public)
        except Exception:
            pass  # persistence must never take a job down

    def _recover_persisted(self) -> None:
        """Load prior job records; anything non-terminal becomes interrupted.

        A restarted worker must never pretend to resume an in-memory GPU
        operation — interrupted jobs get a clear Spanish message and their
        temp files are removed. Completed records survive as history."""
        if not JOBS_META_DIR.is_dir():
            return
        for path in JOBS_META_DIR.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job_id = record.get("job_id")
            if not job_id:
                continue
            if record.get("status") not in TERMINAL:
                record["status"] = "interrupted"
                record["current_step"] = "interrupted"
                record["error_code"] = record.get("error_code") or "worker_restarted"
                record["error"] = "El worker se reinició durante la operación."
                record["message"] = "interrumpido — el worker se reinició"
                record["updated_at"] = _utc()
                shutil.rmtree(JOBS_TMP_DIR / job_id, ignore_errors=True)
                self._persist(record)
            with self._registry_lock:
                self._jobs.setdefault(job_id, record)

    # -- public API --------------------------------------------------------------------

    def submit(
        self,
        job_type: str,
        fn: Callable[[dict[str, Any], Path], Any],
        *,
        heavy: bool,
        model: str = "",
        engine: str = "",
    ) -> dict[str, Any]:
        """Queue fn(job, tmp_dir) on the pool; returns the job record snapshot."""
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "current_step": "queued",
            "current_item": None,
            "total_items": None,
            "percentage": None,
            "message": "en cola",
            "started_at": None,
            "updated_at": _utc(),
            "model": model,
            "engine": engine,
            "cache_hit": None,
            "bytes_downloaded": None,
            "total_bytes": None,
            "gpu_name": None,
            "vram_used_gb": None,
            "vram_free_gb": None,
            "error": None,
            "error_code": None,
            "result": None,
            "heavy": heavy,
            "worker_instance_id": WORKER_INSTANCE_ID,
            "worker_started_utc": WORKER_STARTED_UTC,
            "heartbeat_utc": _utc(),
            "_hb_monotonic": time.monotonic(),
            "_monotonic_start": None,
            "_monotonic_created": time.monotonic(),
        }
        with self._registry_lock:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = threading.Event()
        self._persist(job)
        self._executor.submit(self._run, job_id, fn, heavy)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._registry_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            public = {k: v for k, v in job.items() if not k.startswith("_") and k != "heavy"}
        if job["status"] in TERMINAL:
            public["elapsed_seconds"] = job.get("_final_elapsed", public.get("elapsed_seconds"))
            public["stalled"] = False
        else:
            start = job.get("_monotonic_start") or job.get("_monotonic_created")
            public["elapsed_seconds"] = round(time.monotonic() - start, 1)
            # Stalled = heartbeat silent while the worker itself is alive
            # (whoever serves this response IS alive).
            hb = job.get("_hb_monotonic") or start
            public["stalled"] = bool(
                job["status"] != "queued"
                and time.monotonic() - hb > get_settings().job_stall_seconds
            )
        return public

    def active(self) -> list[dict[str, Any]]:
        with self._registry_lock:
            ids = [j["job_id"] for j in self._jobs.values() if j["status"] not in TERMINAL]
        return [self.get(job_id) for job_id in ids]

    def active_heavy(self) -> dict[str, Any] | None:
        """The currently active heavy job (queued or running), if any."""
        with self._registry_lock:
            for job in self._jobs.values():
                if job["heavy"] and job["status"] not in TERMINAL:
                    job_id = job["job_id"]
                    break
            else:
                return None
        return self.get(job_id)

    def find_active(self, job_type: str, model: str = "") -> dict[str, Any] | None:
        """An in-flight job of this type (and model, if given) — for dedupe:
        a second identical warmup returns the SAME job, never a second load."""
        with self._registry_lock:
            for job in self._jobs.values():
                if (
                    job["job_type"] == job_type
                    and job["status"] not in TERMINAL
                    and (not model or job.get("model") == model)
                ):
                    job_id = job["job_id"]
                    break
            else:
                return None
        return self.get(job_id)

    def last_terminal(self, job_type: str, model: str = "") -> dict[str, Any] | None:
        """The most recently finished job of this type — timeout reconciliation."""
        with self._registry_lock:
            candidates = [
                job for job in self._jobs.values()
                if job["job_type"] == job_type and job["status"] in TERMINAL
                and (not model or job.get("model") == model)
            ]
            if not candidates:
                return None
            job_id = max(candidates, key=lambda j: j.get("updated_at") or "")["job_id"]
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._registry_lock:
            job = self._jobs.get(job_id)
            event = self._cancel_events.get(job_id)
        if job is None or event is None:
            return None
        event.set()
        with self._registry_lock:
            if job["status"] == "queued":  # never started — settle immediately
                self._finish(job, "cancelled", message="cancelado antes de empezar")
        return self.get(job_id)

    def worker_status(self) -> str:
        """idle | busy | warming | degraded — cheap, in-memory only."""
        if self._consecutive_failures >= 3:
            return "degraded"
        running = [j for j in self._jobs.values() if j["status"] not in TERMINAL | {"queued"}]
        if any(j["status"] in LOADING_STAGES for j in running):
            return "warming"
        return "busy" if running else "idle"

    # -- execution ----------------------------------------------------------------------

    def _run(self, job_id: str, fn: Callable, heavy: bool) -> None:
        job = self._jobs[job_id]
        event = self._cancel_events[job_id]
        if event.is_set() or job["status"] in TERMINAL:
            return

        lock_ctx = self._heavy_lock if heavy else self._light_sem
        deadline = time.monotonic() + QUEUE_TIMEOUT_SECONDS
        while not lock_ctx.acquire(timeout=1.0):
            if event.is_set():
                self._finish(job, "cancelled", message="cancelado en cola")
                return
            if time.monotonic() > deadline:
                self._finish(
                    job, "failed", error_code="queue_timeout",
                    error="otro trabajo pesado tardó demasiado — inténtalo de nuevo",
                )
                return
        tmp_dir = JOBS_TMP_DIR / job_id
        heartbeat_stop = threading.Event()
        try:
            if event.is_set():
                self._finish(job, "cancelled", message="cancelado")
                return
            job["_monotonic_start"] = time.monotonic()
            job["started_at"] = _utc()
            if heavy:
                # Resource picture BEFORE the heavy work touches RAM/VRAM —
                # visible in the job record for diagnosis (Phase 3D.0.3a).
                try:
                    job["resources_before"] = gpu.pre_job_snapshot()
                except Exception:
                    job["resources_before"] = None
            self._start_heartbeat(job, heartbeat_stop)
            self._update(job, "preparing", message="preparando")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            progress.set_reporter(lambda step, **fields: self._update(job, step, **fields))
            progress.set_cancel_event(event)
            try:
                result = fn(self.get(job_id), tmp_dir)
            finally:
                progress.clear_reporter()
                progress.clear_cancel_event()
            job["result"] = result
            self._finish(job, "completed", message="listo")
            self._consecutive_failures = 0
        except progress.Cancelled:
            self._cleanup(tmp_dir)
            self._finish(job, "cancelled", message="cancelado — archivos temporales eliminados")
        except Exception as exc:
            self._cleanup(tmp_dir)
            self._consecutive_failures += 1
            code = getattr(exc, "error_code", None) or "job_failed"
            diagnostic = self._diagnostic(job, exc)
            self._finish(
                job, "failed", error_code=code,
                error=f"{type(exc).__name__}: {exc}"[:300],
                diagnostic=diagnostic,
            )
            # One safe line into worker.log (stdout) — no stack traces leak
            # to the UI; the copyable detail lives in the job record.
            print(
                f"[voice-lab] job {job['job_id']} ({job['job_type']}) FAILED "
                f"code={code} stage={diagnostic['last_stage']} "
                f"model={diagnostic['model'] or '-'} "
                f"ram_free_gb={diagnostic['ram'].get('free_gb', '?')} "
                f"vram_free_gb={diagnostic['vram'].get('free_gb', '?')} "
                f"error={type(exc).__name__}",
                flush=True,
            )
        else:
            self._cleanup(tmp_dir)  # success: results were moved to final homes
        finally:
            heartbeat_stop.set()
            lock_ctx.release()

    def _start_heartbeat(self, job: dict[str, Any], stop: threading.Event) -> None:
        """Update + persist the job heartbeat every 2 s while it runs.

        Model loads block the job thread for minutes; the heartbeat proves the
        worker is alive (vs stalled) and leaves an on-disk record of the last
        stage even if the process dies natively (no chance to say goodbye)."""

        def beat() -> None:
            while not stop.wait(HEARTBEAT_SECONDS):
                with self._registry_lock:
                    if job["status"] in TERMINAL:
                        return
                    job["heartbeat_utc"] = _utc()
                    job["_hb_monotonic"] = time.monotonic()
                    status = job["status"]
                    stage_started = job.get("_stage_monotonic") or job.get(
                        "_monotonic_start"
                    ) or time.monotonic()
                self._persist(job)
                # Soft timeouts (Phase 3D.0.4): the blocking torch call cannot
                # be force-killed, but the JOB is truthfully marked failed so
                # the UI never waits forever. The stray thread's late updates
                # are ignored (terminal jobs never change).
                settings = get_settings()
                elapsed = time.monotonic() - stage_started
                if status in LOADING_STAGES and elapsed > settings.model_load_timeout_seconds:
                    self._finish(
                        job, "failed", error_code="model_load_timeout",
                        error=(
                            "La carga del modelo estuvo "
                            f"{settings.model_load_timeout_seconds:g}s sin avanzar "
                            "(VOICE_LAB_MODEL_LOAD_TIMEOUT_SECONDS)."
                        ),
                    )
                    return
                if status == "generating" and elapsed > settings.generation_timeout_seconds:
                    self._finish(
                        job, "failed", error_code="generation_timeout",
                        error=(
                            "La generación estuvo "
                            f"{settings.generation_timeout_seconds:g}s sin avanzar "
                            "(VOICE_LAB_GENERATION_TIMEOUT_SECONDS)."
                        ),
                    )
                    return

        threading.Thread(target=beat, daemon=True, name="voicelab-heartbeat").start()

    @staticmethod
    def _diagnostic(job: dict[str, Any], exc: Exception) -> dict[str, Any]:
        """Safe crash context: type, stage, memory — never a stack trace."""
        try:
            ram = gpu.system_ram_snapshot()
        except Exception:
            ram = {}
        try:
            vram = gpu.gpu_snapshot()
        except Exception:
            vram = {}
        return {
            "exception_type": type(exc).__name__,
            "summary": str(exc)[:300],
            "last_stage": job.get("current_step"),
            "model": job.get("model"),
            "ram": ram,
            "vram": vram,
        }

    def _update(self, job: dict[str, Any], step: str, **fields: Any) -> None:
        with self._registry_lock:
            if job["status"] in TERMINAL:
                return
            job["current_step"] = step
            if step in STAGE_STATUSES:
                # Any progress report is activity: the soft-timeout clock
                # restarts, so a slow-but-reporting job is never falsely killed
                # while a silently stuck one is.
                job["_stage_monotonic"] = time.monotonic()
                job["status"] = step
            for key in ("message", "current_item", "total_items", "percentage",
                        "model", "engine", "cache_hit", "bytes_downloaded", "total_bytes"):
                if key in fields:
                    job[key] = fields[key]
            job["updated_at"] = _utc()
        # VRAM facts at stage granularity only (nvidia-smi is a subprocess).
        snapshot = gpu.gpu_snapshot()
        if snapshot:
            with self._registry_lock:
                job["gpu_name"] = snapshot.get("name")
                job["vram_used_gb"] = snapshot.get("used_gb")
                job["vram_free_gb"] = snapshot.get("free_gb")
        self._persist(job)

    def _finish(self, job: dict[str, Any], status: str, **fields: Any) -> None:
        with self._registry_lock:
            if job["status"] in TERMINAL:
                return
            job["status"] = status
            job["current_step"] = status
            for key, value in fields.items():
                job[key] = value
            start = job.get("_monotonic_start") or job.get("_monotonic_created")
            if start is not None:
                job["_final_elapsed"] = round(time.monotonic() - start, 1)
            job["updated_at"] = _utc()
        self._persist(job)

    @staticmethod
    def _cleanup(tmp_dir: Path) -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
