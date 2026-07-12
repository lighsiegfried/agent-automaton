"""Fifi Voice Lab worker — local-only FastAPI service on 127.0.0.1:8766.

Run from the voice_lab directory with its OWN virtualenv:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8766
(or via:  python scripts/local_runtime.py voice-start)

Contract with the main project:
- The main API only ever talks HTTP to this worker; neural models never load
  inside the main process.
- Responses never contain secrets or absolute internal paths — audio files are
  reported relative to voice_lab/.
- Synthesis runs in the worker threadpool, never on the event loop.
- Engine failures fall back to the profile's fallback_engine; a total failure
  is a structured JSON error, never a crash.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.audio.playback import play_wav
from app.audio.validation import validate_wav
from app.build import (
    API_BUILD,
    python_fingerprint,
    repo_fingerprint,
    source_fingerprint,
    venv_isolated,
)
from app.config import (
    PREVIEWS_DIR,
    VOICE_LAB_ROOT,
    ensure_isolated_dirs,
    get_settings,
    pin_cache_env,
)
from app import auditions, gpu, resources
import app.jobs as jobs_mod
from app.jobs import WORKER_INSTANCE_ID, WORKER_STARTED_UTC
from app.designer import DEFAULT_MODE, FORM_OPTIONS, MODES, DesignerError, VoiceDesigner
from app.engines.base import get_engine, loaded_engines, unload_all
from app.engines.qwen3_tts import CLONE_MODEL_ID, DEFAULT_MODEL_ID, is_clone_base, is_voice_design
from app.jobs import JobManager
from app.profiles.designs import list_designs, load_design
from app.profiles.manager import ProfileError, ProfileManager
from app.profiles.schema import KNOWN_ENGINES, VoiceProfile

UI_PAGE = Path(__file__).resolve().parent / "static" / "designer.html"


class JobError(Exception):
    """Job failure with a category code and a safe user-facing message."""

    def __init__(self, message: str, code: str = "job_failed") -> None:
        super().__init__(message)
        self.error_code = code

VERSION = "0.1.0"

app = FastAPI(title="fifi-voice-lab", version=VERSION)
manager = ProfileManager()
designer = VoiceDesigner()
jobs = JobManager(heavy_slots=get_settings().heavy_job_concurrency)
_started_at = time.monotonic()

# Worker identity (Phase 3D.0.3a) — safe booleans and fingerprints only, never
# machine paths. Lets the UI and local_runtime prove WHICH process/code/repo
# they are talking to, so a stale worker can never masquerade as current.
WORKER_IDENTITY: dict[str, Any] = {}

# Profile-store state from the startup check; /ready refreshes it. A degraded
# store is reported loudly — it is never silently "repaired".
_store_state: dict[str, Any] = {"valid": True, "error": ""}

# Engine availability is dependency probing (imports), not model loading —
# still too slow for a liveness probe, so it is computed once and cached.
_engines_available_cache: dict[str, bool] | None = None


def _compute_identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "instance_id": WORKER_INSTANCE_ID,
        "started_utc": WORKER_STARTED_UTC,
        "version": VERSION,
        "api_build": API_BUILD,
        "source_fingerprint": source_fingerprint(),
        "repo_fingerprint": repo_fingerprint(),
        "python_fingerprint": python_fingerprint(),
        "venv_isolated": venv_isolated(),
    }


def _engines_available() -> dict[str, bool]:
    global _engines_available_cache
    if _engines_available_cache is None:
        _engines_available_cache = {
            name: get_engine(name).available()[0] for name in KNOWN_ENGINES
        }
    return _engines_available_cache


def _worker_status() -> str:
    """Job-manager status, degraded further by an invalid profile store."""
    status = jobs.worker_status()
    if status == "idle" and not _store_state.get("valid", True):
        return "degraded"
    return status


@app.on_event("startup")
def _startup() -> None:
    global WORKER_IDENTITY, _store_state, _engines_available_cache
    ensure_isolated_dirs()
    pin_cache_env()  # all model/cache downloads stay inside voice_lab/
    WORKER_IDENTITY = _compute_identity()
    _engines_available_cache = None
    _store_state = manager.store_check()


class PreviewRequest(BaseModel):
    profile: str = ""  # empty = active profile (or the configured default)
    text: str = ""  # empty = the configured preview sentence


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    profile: str = ""
    play: bool = True  # speak on this host (Fifi's normal mode)


class SelectRequest(BaseModel):
    profile: str = Field(min_length=1)


class UnloadRequest(BaseModel):
    engine: str = ""  # empty = unload every loaded engine


class WarmRequest(BaseModel):
    profile: str = ""  # empty = active profile
    ui_build: str = ""


def _relative(path: Path) -> str:
    """Report files relative to voice_lab/ — absolute paths never leave the worker."""
    try:
        return path.relative_to(VOICE_LAB_ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve_profile(name: str) -> VoiceProfile:
    requested = name or manager.active_profile_name() or get_settings().default_profile
    return manager.get_profile(requested)


def _fallback_chain(profile: VoiceProfile) -> list[str]:
    """Preferred order (Phase 3D.0.1): selected engine -> kokoro -> windows_sapi.

    The profile's own fallback_engine still participates, but never ahead of
    kokoro; duplicates collapse while preserving order.
    """
    chain: list[str] = []
    for name in (profile.engine, "kokoro", profile.fallback_engine, "windows_sapi"):
        if name and name not in chain:
            chain.append(name)
    return chain


def _synthesize_with_fallback(text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
    """Walk the fallback chain; report the engine ACTUALLY used."""
    attempts: list[str] = []
    for engine_name in _fallback_chain(profile):
        result = get_engine(engine_name).synthesize(text, profile, out_path)
        if result["status"] == "ok":
            payload = {
                **result,  # result["engine"] is the engine actually used
                "requested_engine": profile.engine,
                "fallback_used": engine_name != profile.engine,
            }
            if attempts:
                payload["fallback_reason"] = "; ".join(attempts)
            return payload
        attempts.append(f"{engine_name}: {result.get('message')}")
    return {
        "status": "error",
        "engine": profile.engine,
        "requested_engine": profile.engine,
        "fallback_used": True,
        "message": "all engines failed — " + "; ".join(attempts),
    }


def _run_synthesis(text: str, profile_name: str, play: bool, kind: str) -> dict[str, Any]:
    """Blocking synthesis pipeline — always called via run_in_threadpool."""
    try:
        profile = _resolve_profile(profile_name)
    except ProfileError as exc:
        return {"status": "error", "message": str(exc)}

    out_path = PREVIEWS_DIR / f"{kind}_{profile.name}_{uuid.uuid4().hex[:8]}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = _synthesize_with_fallback(text, profile, out_path)
    if result["status"] != "ok":
        return {**result, "profile": profile.name}

    check = validate_wav(out_path)
    payload: dict[str, Any] = {
        **result,
        "profile": profile.name,
        "file": _relative(out_path),
        "audio_valid": check["valid"],
    }
    if not check["valid"]:
        payload["status"] = "error"
        payload["message"] = f"synthesized audio failed validation: {check['reason']}"
        return payload
    payload.setdefault("duration_seconds", check["duration_seconds"])
    if play and get_settings().playback_enabled:
        payload["playback"] = play_wav(out_path)
    return payload


# --- endpoints ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness — must stay FAST and never load models. A busy worker running
    a heavy GPU job still answers here (worker: busy|warming, not down)."""
    return {
        "status": "ok",
        "app": "fifi-voice-lab",
        "version": VERSION,
        "api_build": API_BUILD,
        "instance_id": WORKER_INSTANCE_ID,  # changes on every worker restart
        "started_utc": WORKER_STARTED_UTC,
        "worker": _worker_status(),  # idle | busy | warming | degraded
        "engines_available": _engines_available(),
        "active_profile": manager.active_profile_name(),
    }


@app.get("/worker/info")
def worker_info() -> dict[str, Any]:
    """Who exactly is serving this port: process, code version, repo, and
    interpreter — safe fingerprints only, never machine paths."""
    return {
        "status": "ok",
        **(WORKER_IDENTITY or _compute_identity()),
        "worker": _worker_status(),
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "profile_store": _store_state,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness (vs liveness): profile store, job queue, environment, and
    engine readiness. Never loads a model. A busy worker is still ready."""
    global _store_state
    _store_state = manager.store_check()  # re-check: files may have been fixed
    queue_ok = jobs.worker_status() != "degraded"
    try:
        jobs_mod.JOBS_TMP_DIR.mkdir(parents=True, exist_ok=True)
        probe = jobs_mod.JOBS_TMP_DIR / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        environment_ok = True
    except OSError:
        environment_ok = False
    return {
        "ready": bool(_store_state["valid"] and queue_ok and environment_ok),
        "worker": _worker_status(),
        "active_jobs": len(jobs.active()),
        "profile_store": _store_state,
        "engines_available": _engines_available(),
        "environment_ok": environment_ok,
        "api_build": API_BUILD,
    }


@app.get("/profiles")
def profiles() -> dict[str, Any]:
    return {
        "profiles": [p.public_dict() for p in manager.list_profiles()],
        "active": manager.active_profile_name(),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "active_profile": manager.active_profile_name(),
        "active": manager.active() or {},
        "engines_loaded": loaded_engines(),
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "playback_enabled": get_settings().playback_enabled,
    }


@app.post("/preview")
async def preview(request: PreviewRequest) -> dict[str, Any]:
    text = request.text.strip() or get_settings().preview_text
    return await run_in_threadpool(_run_synthesis, text, request.profile, True, "preview")


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> dict[str, Any]:
    return await run_in_threadpool(
        _run_synthesis, request.text.strip(), request.profile, request.play, "say"
    )


@app.post("/profile/select")
async def profile_select(request: SelectRequest) -> dict[str, Any]:
    try:
        active = await run_in_threadpool(manager.select, request.profile)
    except ProfileError as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "ok", "active": active}


# --- Voice Designer (Phase 3D.0.2) ---------------------------------------------------


class GenerateRequest(BaseModel):
    profile_name: str = Field(min_length=1)
    mode: str = DEFAULT_MODE
    language: str = "es"
    age: str = ""
    gender: str = ""
    timbre: str = ""
    pitch: str = ""
    energy: str = ""
    speed: str = ""
    accent: str = ""
    emotion: str = ""
    personality: str = ""
    instruction: str = ""  # free-form design text
    preview_text: str = ""
    count: int = 3


class VariantRequest(BaseModel):
    profile_name: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)


class RenameRequest(VariantRequest):
    label: str = ""


def _designer_call(func, *args) -> dict[str, Any]:
    try:
        return {"status": "ok", **func(*args)}
    except DesignerError as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/ui")
def ui():
    """The local Voice Designer UI — a single self-contained page."""
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/designer")
def designer_state() -> dict[str, Any]:
    return {
        "status": "ok",
        "modes": list(MODES),
        "default_mode": DEFAULT_MODE,
        "options": FORM_OPTIONS,
        "default_preview_text": get_settings().preview_text,
        "working": designer.working_identities(),
        "designs": list_designs(),
        "active": manager.active_profile_name(),
        "profiles": [p.name for p in manager.list_profiles()],
    }


@app.post("/designer/generate")
async def designer_generate(request: GenerateRequest) -> dict[str, Any]:
    return await run_in_threadpool(
        _designer_call, designer.generate, request.model_dump()
    )


@app.post("/designer/regenerate")
async def designer_regenerate(request: VariantRequest) -> dict[str, Any]:
    return await run_in_threadpool(
        _designer_call, designer.regenerate, request.profile_name, request.variant_id
    )


@app.post("/designer/rename")
async def designer_rename(request: RenameRequest) -> dict[str, Any]:
    return _designer_call(
        designer.rename_variant, request.profile_name, request.variant_id, request.label
    )


@app.delete("/designer/variant")
async def designer_delete_variant(profile_name: str, variant_id: str) -> dict[str, Any]:
    return _designer_call(designer.delete_variant, profile_name, variant_id)


@app.post("/designer/freeze")
async def designer_freeze(request: VariantRequest) -> dict[str, Any]:
    return await run_in_threadpool(
        _designer_call, designer.freeze, request.profile_name, request.variant_id
    )


@app.post("/designer/reference")
async def designer_reference(
    profile_name: str = Form(...),
    language: str = Form("es"),
    transcript: str = Form(...),
    authorization_statement: str = Form(""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Authorized EXTERNAL reference clone — refused without authorization."""
    wav_bytes = await file.read()
    return await run_in_threadpool(
        _designer_call,
        designer.freeze_reference_clone,
        profile_name,
        language,
        wav_bytes,
        transcript,
        authorization_statement,
    )


@app.get("/designer/audio")
def designer_audio(profile_name: str, variant_id: str):
    """Stream one variant WAV for in-browser playback / A-B comparison."""
    try:
        path = designer.variant_audio_path(profile_name, variant_id)
    except DesignerError as exc:
        return {"status": "error", "message": str(exc)}
    if not path.is_file():
        return {"status": "error", "message": "variant audio is missing"}
    return FileResponse(path, media_type="audio/wav")


@app.get("/profile/active")
def profile_active() -> dict[str, Any]:
    active = manager.active() or {}
    return {
        "status": "ok",
        "profile": active.get("profile"),
        "engine": active.get("engine"),
        "updated_utc": active.get("updated_utc"),
    }


# --- background jobs (Phase 3D.0.3) ---------------------------------------------------


class PreviewJobRequest(BaseModel):
    profile: str = Field(min_length=1)
    text: str = ""
    force: bool = False  # "Regenerar prueba": skip + replace the cache
    ui_build: str = ""  # frontend build id; mismatches are refused (preflight)


class CompareJobRequest(BaseModel):
    profile_a: str = Field(min_length=1)
    profile_b: str = Field(min_length=1)
    text: str = ""
    ui_build: str = ""


class FreezeJobRequest(VariantRequest):
    warm_clone: bool = True
    ui_build: str = ""


class VoiceDesignJobRequest(GenerateRequest):
    allow_ollama_unload: bool = False  # explicit per-job UI consent
    ui_build: str = ""


def _profile_is_heavy(profile_name: str) -> bool:
    try:
        return manager.get_profile(profile_name).engine == "qwen3_tts"
    except ProfileError:
        return False  # the job itself will fail fast with a clean error


def _model_resident(model_id: str) -> bool:
    """Is this Qwen model already on the GPU? (Then no new VRAM is needed.)"""
    if not model_id:
        return False
    from app.engines.base import _instances

    qwen = _instances.get("qwen3_tts")
    return qwen is not None and model_id in getattr(qwen, "_models", {})


def _memory_summary() -> dict[str, Any]:
    """Compact per-resource picture for error payloads (display GB only)."""
    result = resources.check_resources()
    checks = result["checks"]
    constrained = {"system_ram": "ram", "vram": "vram", "disk": "disk"}.get(
        result["blocking"]
    )
    return {
        "ram_free_gb": checks["system_ram"]["free_gb"],
        "vram_free_gb": checks["vram"]["free_gb"],
        "disk_free_gb": checks["disk"]["free_gb"],
        "constrained": constrained,
    }


def _job_refusal(
    status_code: int,
    error_code: str,
    user_message: str,
    *,
    technical: str = "",
    retryable: bool,
    preflight: dict[str, Any] | None = None,
) -> JSONResponse:
    """A refused job is a STRUCTURED answer (Phase 3D.0.3a, req. 7) — the real
    reason in Spanish, a safe technical summary, and the worker's state.
    Never just "no se pudo iniciar"."""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_code": error_code,
            "user_message": user_message,
            "message": user_message,  # legacy key: old clients still show the truth
            "technical": technical,
            "retryable": retryable,
            "http_status": status_code,
            "worker": _worker_status(),
            "active_heavy_job": jobs.active_heavy(),
            "memory": _memory_summary(),
            "preflight": preflight,
        },
    )


# Per-resource refusal metadata (Phase 3D.0.4): the messages are resource-
# SPECIFIC — disk is not RAM, RAM is not VRAM. A generic "no hay espacio"
# does not exist anywhere in this worker.
_RESOURCE_REFUSALS = {
    "system_ram": ("insufficient_system_ram", "No hay suficiente memoria RAM disponible."),
    "vram": ("insufficient_vram", "No hay suficiente memoria VRAM disponible."),
    "disk": ("insufficient_disk", "No hay suficiente espacio de almacenamiento."),
}


def preflight_result(
    *,
    heavy: bool,
    ui_build: str = "",
    engine: str = "",
    model: str = "",
    profiles: list[str] | None = None,
    allow_ollama_unload: bool = False,
) -> dict[str, Any]:
    """The structured generation preflight (Phase 3D.0.4, Part B).

    A job is only rejected for a VERIFIED blocking condition: never because a
    model is already loaded, never because VRAM is merely in use while the
    free amount passes the byte-exact threshold, never for idle GPU."""
    checks: dict[str, Any] = {}
    blocking: tuple[int, str, str, str, bool] | None = None  # http, code, msg, tech, retry

    def block(http: int, code: str, message: str, technical: str, retryable: bool):
        nonlocal blocking
        if blocking is None:
            blocking = (http, code, message, technical, retryable)

    if ui_build and ui_build != API_BUILD:
        block(
            409, "version_mismatch",
            "La interfaz y el worker tienen versiones diferentes. Reinicia Voice Lab.",
            f"ui_build={ui_build!r} api_build={API_BUILD!r}", False,
        )

    store = manager.store_check()
    checks["profile_store"] = {"ok": store["valid"], **store}
    if not store["valid"]:
        block(
            503, "profile_store_invalid",
            "El almacén de perfiles de voz no es válido: " + store["error"],
            f"profile_count={store['profile_count']} active={store['active_profile']!r}",
            True,
        )
    for name in profiles or []:
        try:
            manager.get_profile(name)
        except ProfileError as exc:
            checks["profile_store"]["unknown_profile"] = name
            block(
                404, "unknown_profile",
                f"El perfil {name!r} no existe en el almacén compartido.",
                str(exc), False,
            )

    if engine:
        ok, reason = get_engine(engine).available()
        checks["engine"] = {"ok": ok, "name": engine, "reason": "" if ok else reason}
        if not ok:
            block(
                503, "engine_unavailable",
                f"El motor {engine} no está disponible en este worker.",
                reason, False,
            )
    else:
        checks["engine"] = {"ok": True, "name": None}

    model_loaded = _model_resident(model)
    cache_state = gpu.model_cache_status(model) if model else "cached"
    checks["model_cache"] = {
        # "cached" means no download is required; "loaded" means generation
        # reuses the resident instance — NEITHER is ever a rejection reason.
        "ok": True,
        "state": cache_state,
        "loaded": model_loaded,
        "download_required": cache_state == "not_installed",
    }

    conflict = jobs.active_heavy() if heavy else None
    checks["job_queue"] = {
        "ok": conflict is None,
        "active_heavy_job": conflict,
    }
    if conflict is not None:
        block(
            409, "heavy_job_running",
            "Ya hay un trabajo pesado en curso — espera a que termine e "
            "inténtalo de nuevo.",
            f"job {conflict['job_id']} ({conflict['job_type']}) status={conflict['status']}",
            True,
        )

    # Resource checks — BYTES vs bytes inside resources.check_resources().
    # A resident model needs no new VRAM, RAM, or downloads: generation reuses
    # the loaded instance, so those checks are skipped — never failed.
    resource_result = resources.check_resources()
    for key in ("system_ram", "vram", "disk"):
        entry = dict(resource_result["checks"][key])
        if not heavy:
            entry["skipped"] = "light job"
            entry["ok"] = True
        elif model_loaded:
            entry["skipped"] = "modelo ya cargado — se reutiliza la instancia"
            entry["ok"] = True
        elif entry["ok"] is None:
            entry["note"] = "no medible — se permite (el motor tiene su propia protección OOM)"
        checks[key] = entry
    if heavy:
        for key in ("system_ram", "vram", "disk"):
            if checks[key].get("ok") is False:
                if key == "vram" and allow_ollama_unload and gpu.allow_temporary_ollama_unload():
                    checks[key]["note"] = "se liberará Ollama temporalmente (consentido)"
                    continue
                code, message = _RESOURCE_REFUSALS[key]
                block(
                    503, code, message,
                    (
                        f"{key}: libre {checks[key]['free_gb']} GB, "
                        f"requerido {checks[key]['required_gb']} GB "
                        f"(comparación en bytes: {checks[key]['free_bytes']} >= "
                        f"{checks[key]['required_bytes']})"
                    ),
                    True,
                )

    try:
        jobs_mod.JOBS_TMP_DIR.mkdir(parents=True, exist_ok=True)
        probe = jobs_mod.JOBS_TMP_DIR / ".preflight"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["worker_environment"] = {"ok": True, "job_dir_writable": True}
    except OSError as exc:
        checks["worker_environment"] = {"ok": False, "job_dir_writable": False}
        block(
            503, "storage_unwritable",
            "No se puede escribir en el directorio de trabajos de Voice Lab.",
            type(exc).__name__, False,
        )

    result: dict[str, Any] = {
        "allowed": blocking is None,
        "checks": checks,
        "blocking_reason": blocking[1] if blocking else None,
    }
    if blocking:
        result["_refusal"] = blocking
    return result


def _preflight(
    *,
    heavy: bool,
    ui_build: str = "",
    engine: str = "",
    model: str = "",
    profiles: list[str] | None = None,
    allow_ollama_unload: bool = False,
) -> JSONResponse | None:
    """Gate a job submission. None = allowed; otherwise the refusal response."""
    result = preflight_result(
        heavy=heavy, ui_build=ui_build, engine=engine, model=model,
        profiles=profiles, allow_ollama_unload=allow_ollama_unload,
    )
    if result["allowed"]:
        return None
    http, code, message, technical, retryable = result.pop("_refusal")
    return _job_refusal(
        http, code, message, technical=technical, retryable=retryable,
        preflight=result,
    )


def _gate_headroom(kind: str, allow_ollama_unload: bool) -> None:
    """Refuse (clearly) or make room BEFORE a heavy load — never just OOM."""
    from app import progress
    from app.engines.base import _instances

    qwen = _instances.get("qwen3_tts")
    target = DEFAULT_MODEL_ID if kind == "voice_design" else CLONE_MODEL_ID
    if qwen is not None and target in getattr(qwen, "_models", {}):
        return  # already resident: no new VRAM/RAM needed

    # RAM protection (hotfix, req. 8): before loading Qwen, drop every UNUSED
    # Voice Lab model and garbage-collect, then verify free system RAM. On a
    # 16 GB machine a load under memory pressure dies NATIVELY (no traceback)
    # — refuse clearly instead of letting Windows kill the worker silently.
    if qwen is not None:
        for model_id in list(getattr(qwen, "_models", {})):
            if model_id != target:
                progress.report(
                    "preparing",
                    message="Liberando modelos de Voice Lab no usados antes de cargar…",
                )
                qwen.unload_model(model_id)  # ref release -> gc -> CUDA cache
    if get_settings().exclusive_design_mode and kind == "voice_design":
        # Exclusive design mode (req. 15): also free Kokoro. The main API is
        # never stopped; Ollama is only touched via the separate,
        # doubly-consented unload path below.
        from app.engines.base import unload_all

        progress.report(
            "preparing",
            message="Modo diseño exclusivo: liberando modelos de Voice Lab no usados…",
        )
        unload_all("kokoro")
    import gc

    gc.collect()
    # Byte-exact RAM re-check after cleanup (resources module, Phase 3D.0.4).
    ram_check = resources.check_resources()["checks"]["system_ram"]
    if ram_check["ok"] is False:
        raise JobError(
            "No hay suficiente memoria RAM disponible. "
            f"Hay {ram_check['free_gb']} GB libres y el mínimo configurado es "
            f"{ram_check['required_gb']:g} GB (VOICE_LAB_MIN_FREE_SYSTEM_RAM_GB). "
            "Cierra aplicaciones e inténtalo de nuevo.",
            "insufficient_system_ram",
        )
    decision = gpu.headroom_check(
        gpu.VRAM_ESTIMATES_GB[kind], allow_ollama_unload=allow_ollama_unload
    )
    if decision.get("unload_ollama_first"):
        progress.report(
            "preparing",
            message="Liberando temporalmente el modelo de Ollama (se recargará solo)…",
        )
        gpu.unload_ollama_temporarily()
        return
    if not decision.get("ok"):
        raise JobError(decision.get("message", "VRAM insuficiente"), "insufficient_vram")


def _job_voice_design(payload: dict[str, Any]):
    def run(job: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
        from app import progress

        progress.report(
            "resource_preflight",
            message="Comprobando RAM, VRAM y almacenamiento antes de cargar…",
        )
        _gate_headroom("voice_design", payload.get("allow_ollama_unload", False))
        try:
            manifest = designer.generate(payload)
        except DesignerError as exc:
            raise JobError(str(exc), "generation_failed") from exc
        # completed is ONLY allowed when every requested variant produced
        # validated audio; partial success identifies exactly which failed.
        failed = manifest.get("failed_variants") or []
        if failed:
            good = len(manifest.get("variants") or [])
            detail = "; ".join(
                f"variante {f['index'] + 1}: {f['error']}" for f in failed
            )
            raise JobError(
                f"{good} variantes válidas, {len(failed)} fallidas — {detail}",
                "partial_generation_failed",
            )
        if not manifest.get("variants"):
            raise JobError(
                "la generación terminó sin producir ningún audio válido",
                "no_valid_audio",
            )
        return manifest

    return run


def _job_freeze(profile_name: str, variant_id: str, warm_clone: bool):
    def run(job: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
        from app import progress

        progress.report("freezing", message="Congelando identidad (referencia + perfil)…")
        try:
            result = designer.freeze(profile_name, variant_id)
        except DesignerError as exc:
            raise JobError(str(exc), "freeze_failed") from exc
        if not warm_clone:
            return result
        # Sequential policy: VoiceDesign already generated the variants; make
        # room for the clone model if the card is tight, THEN warm the clone.
        from app.engines.base import _instances

        decision = gpu.headroom_check(gpu.VRAM_ESTIMATES_GB["clone"])
        qwen = _instances.get("qwen3_tts")
        if not decision.get("ok") and qwen is not None:
            progress.report(
                "preparing",
                message="Liberando el modelo VoiceDesign para hacer sitio al de clonación…",
            )
            qwen.unload_model(DEFAULT_MODEL_ID)
        progress.report("warming", message="Cargando el modelo de clonación y validando la identidad…")
        profile = manager.get_profile(profile_name)
        warm_path = tmp_dir / "warm.wav"
        warm = _synthesize_with_fallback("Identidad de voz lista.", profile, warm_path)
        result["warm"] = {
            "status": warm.get("status"),
            "engine": warm.get("engine"),
            "fallback_used": warm.get("fallback_used"),
            "seconds": warm.get("seconds"),
        }
        return result

    return run


def _job_preview(profile_name: str, text: str, force: bool = False):
    def run(job: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
        try:
            return auditions.run_audition(
                manager, _synthesize_with_fallback, profile_name, text, force=force
            )
        except ProfileError as exc:
            raise JobError(str(exc), "unknown_profile") from exc
        except RuntimeError as exc:
            raise JobError(str(exc), "generation_failed") from exc

    return run


def _job_compare(a: str, b: str, text: str):
    def run(job: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
        from app import progress

        results = {}
        for slot, name in (("a", a), ("b", b)):
            progress.report("generating", message=f"Sintetizando {name}…",
                            current_item=1 if slot == "a" else 2, total_items=2)
            try:
                results[slot] = auditions.run_audition(
                    manager, _synthesize_with_fallback, name, text
                )
            except (ProfileError, RuntimeError) as exc:
                raise JobError(f"{name}: {exc}", "generation_failed") from exc
        return {"a": results["a"], "b": results["b"], "text": text or auditions.DEFAULT_AUDITION_TEXT}

    return run


def _accepted(job: dict[str, Any], model: str = "") -> JSONResponse:
    """HTTP 202 + the queued job record, with honest cache facts: cached
    models never say 'descargando'."""
    if model:
        job = {**job, "model_cache": gpu.model_cache_status(model)}
    return JSONResponse(status_code=202, content=job)


@app.post("/jobs/voice-design")
def jobs_voice_design(request: VoiceDesignJobRequest):
    refusal = _preflight(
        heavy=True, ui_build=request.ui_build, engine="qwen3_tts",
        model=DEFAULT_MODEL_ID, allow_ollama_unload=request.allow_ollama_unload,
    )
    if refusal is not None:
        return refusal
    payload = request.model_dump()
    job = jobs.submit(
        "voice_design", _job_voice_design(payload), heavy=True,
        model=DEFAULT_MODEL_ID, engine="qwen3_tts",
    )
    return _accepted(job, DEFAULT_MODEL_ID)


@app.post("/jobs/profile-preview")
def jobs_profile_preview(request: PreviewJobRequest):
    heavy = _profile_is_heavy(request.profile)
    model = ""
    if heavy:
        model = manager.get_profile(request.profile).model
    refusal = _preflight(
        heavy=heavy, ui_build=request.ui_build, profiles=[request.profile],
        engine="qwen3_tts" if heavy else "", model=model,
    )
    if refusal is not None:
        return refusal
    job = jobs.submit(
        "profile_preview", _job_preview(request.profile, request.text, request.force),
        heavy=heavy,
    )
    return _accepted(job, model)


@app.post("/jobs/profile-compare")
def jobs_profile_compare(request: CompareJobRequest):
    heavy = _profile_is_heavy(request.profile_a) or _profile_is_heavy(request.profile_b)
    refusal = _preflight(
        heavy=heavy, ui_build=request.ui_build,
        profiles=[request.profile_a, request.profile_b],
        engine="qwen3_tts" if heavy else "",
    )
    if refusal is not None:
        return refusal
    job = jobs.submit(
        "profile_compare",
        _job_compare(request.profile_a, request.profile_b, request.text),
        heavy=heavy,
    )
    return _accepted(job)


@app.post("/jobs/freeze")
def jobs_freeze(request: FreezeJobRequest):
    refusal = _preflight(
        heavy=True, ui_build=request.ui_build, engine="qwen3_tts", model=CLONE_MODEL_ID
    )
    if refusal is not None:
        return refusal
    job = jobs.submit(
        "freeze", _job_freeze(request.profile_name, request.variant_id, request.warm_clone),
        heavy=True, model=CLONE_MODEL_ID, engine="qwen3_tts",
    )
    return _accepted(job, CLONE_MODEL_ID)


@app.get("/jobs/active")
def jobs_active() -> dict[str, Any]:
    return {"status": "ok", "jobs": jobs.active(), "worker": jobs.worker_status()}


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        return {"status": "error", "error_code": "unknown_job", "message": "unknown job"}
    return job


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str) -> dict[str, Any]:
    job = jobs.cancel(job_id)
    if job is None:
        return {"status": "error", "error_code": "unknown_job", "message": "unknown job"}
    return job


# --- model / resource coordination -----------------------------------------------------


@app.get("/models/status")
async def models_status() -> dict[str, Any]:
    report = await run_in_threadpool(gpu.resource_report)
    return {"status": "ok", "worker": jobs.worker_status(), **report}


@app.get("/resources/status")
async def resources_status() -> dict[str, Any]:
    """The canonical resource picture (Phase 3D.0.4): disk vs system RAM vs
    dedicated VRAM, clearly separated, byte-exact — no paths, no secrets."""
    snap = await run_in_threadpool(resources.snapshot)
    policy = await run_in_threadpool(resources.check_resources)
    settings = get_settings()
    return {
        "status": "ok",
        "worker": _worker_status(),
        **snap,
        "thresholds": {
            "min_free_system_ram_gb": settings.min_free_system_ram_gb,
            "min_free_vram_gb": settings.min_free_vram_gb,
            "min_free_disk_gb": settings.min_free_disk_gb,
        },
        "checks": policy["checks"],
        "blocking": policy["blocking"],
    }


class PreflightRequest(BaseModel):
    kind: str = "voice_design"  # voice_design | freeze | profile_preview
    profile: str = ""
    ui_build: str = ""


@app.post("/jobs/preflight")
async def jobs_preflight(request: PreflightRequest) -> dict[str, Any]:
    """Dry-run preflight for the UI: what WOULD happen if this job started."""
    kind = request.kind
    if kind == "voice_design":
        heavy, engine, model, profiles_list = True, "qwen3_tts", DEFAULT_MODEL_ID, None
    elif kind == "freeze":
        heavy, engine, model, profiles_list = True, "qwen3_tts", CLONE_MODEL_ID, None
    else:  # profile_preview
        heavy = _profile_is_heavy(request.profile)
        model = ""
        if heavy:
            model = manager.get_profile(request.profile).model
        engine = "qwen3_tts" if heavy else ""
        profiles_list = [request.profile] if request.profile else None
    result = await run_in_threadpool(
        lambda: preflight_result(
            heavy=heavy, ui_build=request.ui_build, engine=engine, model=model,
            profiles=profiles_list,
        )
    )
    refusal = result.pop("_refusal", None)
    checks = result["checks"]
    return {
        "status": "ok",
        **result,
        "summary": {
            "ram_libre_gb": checks.get("system_ram", {}).get("free_gb"),
            "vram_libre_gb": checks.get("vram", {}).get("free_gb"),
            "almacenamiento_libre_gb": checks.get("disk", {}).get("free_gb"),
            "motor_solicitado": engine or "kokoro/windows",
            "modelo_en_cache": checks.get("model_cache", {}).get("state") == "cached",
            "modelo_ya_cargado": checks.get("model_cache", {}).get("loaded"),
            "trabajo_pesado_activo": bool(checks.get("job_queue", {}).get("active_heavy_job")),
            "resultado": "listo" if result["allowed"] else "bloqueado",
            "motivo": refusal[2] if refusal else None,
        },
    }


@app.post("/models/unload")
async def models_unload(request: UnloadRequest) -> dict[str, Any]:
    """Free Voice Lab VRAM. Cached model FILES are never touched."""
    unloaded = await run_in_threadpool(unload_all, request.engine or None)
    return {"status": "ok", "unloaded": unloaded, "note": "los archivos en caché no se borran"}


# --- profile details / deletion ---------------------------------------------------------


def _profile_mode(profile: VoiceProfile, design: dict[str, Any] | None) -> str:
    if profile.engine != "qwen3_tts":
        return "built-in"
    if is_clone_base(profile.model):
        return (design or {}).get("mode") or "clone"
    if is_voice_design(profile.model) and profile.style_instruction:
        return "voice_design"
    return "built-in"  # CustomVoice named timbres are built-in speakers


def _profile_details(profile: VoiceProfile) -> dict[str, Any]:
    design = load_design(profile.name)
    profile_file = manager.profiles_dir / f"{profile.name}.json"
    try:
        stat = profile_file.stat()
        created = datetime_from_ts(stat.st_ctime)
        updated = datetime_from_ts(stat.st_mtime)
    except OSError:
        created = updated = None
    available, availability_reason = get_engine(profile.engine).available()
    reference = (design or {}).get("reference") or {}
    return {
        **profile.public_dict(),
        "mode": _profile_mode(profile, design),
        "deletable": profile.provenance.get("source") == "voice_designer",
        "active": manager.active_profile_name() == profile.name,
        "available": available,
        "availability_reason": availability_reason if not available else "",
        "fallback_chain": _fallback_chain(profile),
        "design_fields": (design or {}).get("fields") or {},
        "design_instruction": (design or {}).get("instruction") or profile.style_instruction,
        "reference_text": reference.get("text", ""),
        "authorization": reference.get("authorization") or {},
        "created": created,
        "updated": updated,
    }


def datetime_from_ts(ts: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat(timespec="seconds")


@app.get("/profiles/details")
def profiles_details() -> dict[str, Any]:
    return {
        "status": "ok",
        "active": manager.active_profile_name(),
        "profiles": [_profile_details(p) for p in manager.list_profiles()],
    }


@app.delete("/profiles/{name}")
async def profiles_delete(name: str) -> dict[str, Any]:
    """Delete a USER-CREATED profile (designer output). Built-ins are refused."""
    try:
        profile = manager.get_profile(name)
    except ProfileError as exc:
        return {"status": "error", "message": str(exc)}
    if profile.provenance.get("source") != "voice_designer":
        return {
            "status": "error",
            "message": f"{name!r} es un perfil integrado — no se puede eliminar.",
        }
    if manager.active_profile_name() == name:
        return {
            "status": "error",
            "message": f"{name!r} es la voz activa — activa otra voz antes de eliminarla.",
        }

    def _delete() -> None:
        import shutil

        (manager.profiles_dir / f"{name}.json").unlink(missing_ok=True)
        from app.profiles.designs import design_path

        design_path(name).unlink(missing_ok=True)
        shutil.rmtree(designer.identities_dir / name, ignore_errors=True)
        auditions.drop_profile_auditions(name)

    await run_in_threadpool(_delete)
    return {"status": "ok", "deleted": name}


@app.get("/previews/audition/{key}")
def audition_audio(key: str):
    """Stream a cached audition WAV for in-browser playback (legacy path)."""
    try:
        wav_path, _meta = auditions.audition_paths(key)
    except ValueError:
        return {"status": "error", "message": "invalid audition key"}
    if not wav_path.is_file():
        return {"status": "error", "message": "audition audio not found"}
    return FileResponse(wav_path, media_type="audio/wav")


@app.get("/audio/previews/{preview_id}")
def preview_audio(preview_id: str, v: str = ""):
    """Stable preview audio URL (Phase 3D.0.4, req. 27).

    audio/wav with content-length, real 404s, traversal rejected by the strict
    id validator, no absolute paths anywhere. Regenerated previews get a new
    ?v= so no stale browser cache can play old audio; no-store keeps it safe."""
    try:
        wav_path, _meta = auditions.audition_paths(preview_id)
    except ValueError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error_code": "invalid_preview_id",
                     "message": "identificador de preview inválido"},
        )
    if not wav_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error_code": "preview_not_found",
                     "message": "la preview no existe (genera una nueva)"},
        )
    return FileResponse(
        wav_path, media_type="audio/wav", headers={"Cache-Control": "no-store"}
    )


def _job_warm(profile: VoiceProfile):
    """Background warmup: load the profile's engine, synthesize silently,
    validate the audio. Stages come from the engine (checking_cache/cached/
    loading_from_disk/moving_to_gpu/warming_cuda) plus validating_synthesis."""

    def run(job: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
        from app import progress

        if profile.engine == "qwen3_tts":
            kind = "voice_design" if is_voice_design(profile.model) else "clone"
            _gate_headroom(kind, False)
        warm_path = tmp_dir / "warm.wav"
        result = _synthesize_with_fallback("Sistema de voz listo.", profile, warm_path)
        if result.get("status") != "ok":
            raise JobError(
                f"el calentamiento falló: {result.get('message')}", "warm_failed"
            )
        progress.report("validating_synthesis", message="Validando la síntesis de prueba…")
        check = validate_wav(warm_path)
        if not check["valid"]:
            raise JobError(
                f"el audio de calentamiento no es válido: {check['reason']}",
                "warm_invalid_audio",
            )
        return {
            "ready": True,
            "profile": profile.name,
            "requested_engine": profile.engine,
            "engine": result.get("engine"),
            "fallback_used": bool(result.get("fallback_used")),
            "model": profile.model,
            "seconds": result.get("seconds"),
            "duration_seconds": check["duration_seconds"],
        }

    return run


@app.post("/warm")
def warm(request: WarmRequest):
    """ASYNCHRONOUS warmup (Phase 3D.0.4): HTTP 202 + a pollable job.

    Idempotent: an already-warm engine answers ready immediately (200) and a
    duplicate request while the same warmup runs returns the SAME job — a
    model instance is never loaded twice."""
    try:
        profile = _resolve_profile(request.profile)
    except ProfileError as exc:
        return _job_refusal(
            404, "unknown_profile", f"El perfil solicitado no existe.",
            technical=str(exc), retryable=False,
        )
    heavy = profile.engine == "qwen3_tts"
    engine_obj = get_engine(profile.engine)
    already = engine_obj.loaded and (not heavy or _model_resident(profile.model))
    if already:
        last = jobs.last_terminal("warm", model=profile.model)
        return {
            "status": "ready",
            "already_loaded": True,
            "profile": profile.name,
            "engine": profile.engine,
            "model": profile.model,
            "message": "El motor ya está cargado.",
            "last_warm_job": last,
        }
    active = jobs.find_active("warm", model=profile.model)
    if active is not None:
        return JSONResponse(
            status_code=202,
            content={**active, "deduplicated": True, "profile": profile.name,
                     "initial_status": active["status"]},
        )
    refusal = _preflight(
        heavy=heavy, engine=profile.engine if heavy else "", model=profile.model,
        ui_build=request.ui_build,
    )
    if refusal is not None:
        return refusal
    job = jobs.submit(
        "warm", _job_warm(profile), heavy=heavy,
        model=profile.model, engine=profile.engine,
    )
    return JSONResponse(
        status_code=202,
        content={
            **job,
            "model_cache": gpu.model_cache_status(profile.model),
            "profile": profile.name,
            "initial_status": job["status"],
        },
    )


@app.post("/unload")
async def unload(request: UnloadRequest) -> dict[str, Any]:
    unloaded = await run_in_threadpool(unload_all, request.engine or None)
    return {"status": "ok", "unloaded": unloaded}
