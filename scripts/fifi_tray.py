"""Fifi Windows system-tray app (Phase 3E) — host only, no Electron.

A small tray icon that orchestrates the whole local stack by REUSING
scripts/local_runtime.py (it shells its subcommands — it never re-implements
startup) and reflects Fifi's state (stopped / starting / warming / ready /
listening / muted / processing / degraded / error).

Design / safety:
- Orchestration is delegated to local_runtime, which only ever stops processes
  IT owns; the tray never kills unrelated Docker or Python processes.
- Single-instance: one tray, and it relies on the runtime's own single-wake /
  single-worker guarantees for the services.
- Notifications are de-duplicated so a flapping service cannot spam the user.
- The settings window is a read-only localhost page that exposes NO secrets.

The pystray/Pillow/plyer imports are OPTIONAL and lazy (install
requirements-desktop.txt); the controller logic below has none of them so it is
fully unit-testable. Python (not PowerShell): AllSigned blocks unsigned .ps1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tray_state as ts

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
LOCAL_RUNTIME = SCRIPTS_DIR / "local_runtime.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import llm_smoke  # noqa: E402  (shared .env/env reader)

RUNTIME_DIR = PROJECT_ROOT / "storage" / "runtime"
LOG_DIR = PROJECT_ROOT / "storage" / "logs"
TRAY_LOCK_FILE = RUNTIME_DIR / "fifi_tray.lock"
TRAY_LOG_FILE = LOG_DIR / "tray.log"
PROFILES_DIR = PROJECT_ROOT / "config" / "voices" / "profiles"
VOICE_LAB_CLI = PROJECT_ROOT / "voice_lab" / "scripts" / "voice_lab.py"

MAX_START_ATTEMPTS = 4          # bounded — never loop forever waiting on Docker
START_BACKOFF_SECONDS = 5.0


# --- startup policy (FIFI_* env, read like the other host scripts) -----------------


class StartupPolicy:
    def __init__(self, get=llm_smoke.config_value):
        def flag(name, default):
            return get(name, "true" if default else "false").strip().lower() in (
                "1", "true", "yes", "on"
            )

        self.auto_start = flag("FIFI_AUTO_START", False)
        try:
            self.startup_delay_seconds = int(get("FIFI_STARTUP_DELAY_SECONDS", "20"))
        except ValueError:
            self.startup_delay_seconds = 20
        mode = get("FIFI_DEFAULT_INTERACTION_MODE", "wake").strip().lower()
        self.default_interaction_mode = mode if mode in ("wake", "ptt") else "wake"
        self.start_minimized = flag("FIFI_START_MINIMIZED", True)
        self.auto_warm_llm = flag("FIFI_AUTO_WARM_LLM", True)
        self.auto_warm_stt = flag("FIFI_AUTO_WARM_STT", True)
        self.auto_warm_tts = flag("FIFI_AUTO_WARM_TTS", True)
        self.keep_ollama_on_exit = flag("FIFI_KEEP_OLLAMA_RUNNING_ON_EXIT", False)
        self.keep_voice_lab_on_exit = flag("FIFI_KEEP_VOICE_LAB_RUNNING_ON_EXIT", False)

    def as_dict(self) -> dict:
        return {
            "auto_start": self.auto_start,
            "startup_delay_seconds": self.startup_delay_seconds,
            "default_interaction_mode": self.default_interaction_mode,
            "start_minimized": self.start_minimized,
            "auto_warm_llm": self.auto_warm_llm,
            "auto_warm_stt": self.auto_warm_stt,
            "auto_warm_tts": self.auto_warm_tts,
            "keep_ollama_on_exit": self.keep_ollama_on_exit,
            "keep_voice_lab_on_exit": self.keep_voice_lab_on_exit,
        }


# --- single-instance lock ----------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=15)
        except Exception:
            return False
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class InstanceLock:
    """One-tray guarantee via a PID lock file. A stale lock (dead PID) is reclaimed."""

    def __init__(self, path: Path = TRAY_LOCK_FILE, pid: int | None = None, alive=_pid_alive):
        self.path = Path(path)
        self.pid = pid if pid is not None else os.getpid()
        self._alive = alive

    def owner_pid(self) -> int | None:
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def held_by_other(self) -> bool:
        owner = self.owner_pid()
        return owner is not None and owner != self.pid and self._alive(owner)

    def acquire(self) -> bool:
        """True if we now hold the lock; False if a LIVE other tray holds it."""
        if self.held_by_other():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(self.pid), encoding="utf-8")
        return True

    def release(self) -> None:
        if self.owner_pid() == self.pid:
            self.path.unlink(missing_ok=True)


# --- notifications (de-duplicated) -------------------------------------------------


def _default_notify_backend(title: str, message: str) -> None:
    try:
        from plyer import notification

        notification.notify(title=title, message=message, app_name="Fifi", timeout=5)
    except Exception:
        pass  # notifications are best-effort; never crash the tray over one


class Notifier:
    """Sends desktop notifications, suppressing repeats of the same key."""

    def __init__(self, backend=_default_notify_backend, clock=time.monotonic,
                 min_repeat_seconds=60.0):
        self.backend = backend
        self.clock = clock
        self.min_repeat_seconds = min_repeat_seconds
        self._last: dict[str, float] = {}

    def notify(self, title: str, message: str, key: str | None = None) -> bool:
        key = key or f"{title}:{message}"
        now = self.clock()
        last = self._last.get(key)
        if last is not None and (now - last) < self.min_repeat_seconds:
            return False  # de-duplicated
        self._last[key] = now
        self.backend(title, message)
        return True


# --- settings snapshot (no secrets) ------------------------------------------------


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    if not profiles_dir.is_dir():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.json"))


def _active_profile() -> str | None:
    try:
        data = json.loads((PROJECT_ROOT / "config" / "voices" / "active.json").read_text("utf-8"))
        return data.get("profile")
    except (OSError, ValueError):
        return None


def settings_snapshot(policy: StartupPolicy, autostart_enabled: bool,
                      get=llm_smoke.config_value) -> dict:
    """Non-secret settings for the settings window. NEVER include tokens/paths-with-secrets."""
    return {
        "active_voice": _active_profile(),
        "wake_threshold": get("WAKE_WORD_THRESHOLD", "0.55"),
        "wake_hotkey": get("WAKE_WORD_MUTE_HOTKEY", "ctrl+alt+m"),
        "ptt_hotkey": get("PTT_HOTKEY", "ctrl+alt+space"),
        "interaction_mode": policy.default_interaction_mode,
        "autostart_enabled": autostart_enabled,
        "real_windows_tools": get("ENABLE_REAL_WINDOWS_TOOLS", "false"),
        "runtime_mode": get("VOICE_LAB_RUNTIME", "host"),
        "tts_engine": get("TTS_ENGINE", "windows"),
    }


def fifi_autostart_status_enabled() -> bool:
    """Whether auto-start is currently scheduled (best-effort)."""
    try:
        import fifi_autostart

        return bool(fifi_autostart.status().get("enabled"))
    except Exception:
        return False


def _escape(value) -> str:
    import html

    return html.escape(str(value))


def render_settings_html(snapshot: dict) -> str:
    """Read-only settings page. Renders only the non-secret snapshot keys."""
    rows = "".join(
        f"<tr><td>{_escape(k)}</td><td>{_escape(v)}</td></tr>" for k, v in snapshot.items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Fifi settings</title>"
        "<style>body{font-family:sans-serif;margin:2rem}"
        "table{border-collapse:collapse}td{border:1px solid #ccc;padding:.35rem .8rem}"
        "td:first-child{color:#555}</style></head><body>"
        "<h2>Fifi settings</h2>"
        f"<table>{rows}</table>"
        "<p><em>Read-only. Secrets (tokens, keys, credentials) are never shown here.</em></p>"
        "</body></html>"
    )


class SettingsServer:
    """Serves the settings page on 127.0.0.1 in a background thread (lazy stdlib)."""

    def __init__(self, html: str, host: str = "127.0.0.1", port: int = 0):
        self._html = html
        self.host = host
        self.port = port
        self._httpd = None
        self._thread = None

    def set_html(self, html: str) -> None:
        self._html = html

    def start(self) -> str:
        import http.server
        import threading

        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = server._html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence access logs
                pass

        self._httpd = http.server.HTTPServer((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None


# --- runtime runner (shells local_runtime; never re-implements startup) ------------


def _default_runtime_runner(argv: list[str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAY_LOG_FILE, "ab") as log:
        return subprocess.run(
            [sys.executable, str(LOCAL_RUNTIME), *argv],
            cwd=str(PROJECT_ROOT), stdout=log, stderr=log,
        ).returncode


# Only these runtime subcommands are ever issued by the tray. stop/wake-stop/
# voice-stop each stop ONLY processes the runtime owns — so the tray can never
# terminate an unrelated Docker container or Python process.
ALLOWED_RUNTIME_COMMANDS = {
    "start", "stop", "restart", "wake-start", "wake-stop", "voice-start",
    "voice-stop", "voice-warm", "voice-mode", "voice-ui", "warm", "unload",
    "voice-unload", "model-status", "wake-status",
}


def _api_base() -> str:
    host = llm_smoke.config_value("API_HOST", "127.0.0.1")
    port = llm_smoke.config_value("API_PORT", "8000")
    return f"http://{host}:{port}"


class DefaultTextApi:
    """Thin client for the /text endpoints on the local API (lazy httpx)."""

    def pending(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/text/pending", timeout=5.0).json().get("pending")
        except Exception:
            return None

    def confirm(self, action_id: str, phrase: str) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/text/confirm",
                              json={"action_id": action_id, "phrase": phrase}, timeout=10.0).json()
        except Exception:
            return None

    def cancel(self) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/text/cancel", timeout=5.0).json()
        except Exception:
            return None

    # -- unified cross-domain pending (Phase 4B.1) --------------------------------

    def pending_summary(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/pending", timeout=5.0).json().get("pending")
        except Exception:
            return None

    def confirm_domain(self, domain: str, action_id: str) -> dict | None:
        """Confirm the active pending using the EXACT phrase for its domain."""
        endpoint = {"text": "/text/confirm", "browser": "/browser/form/confirm"}.get(domain)
        phrase = {"text": "insert text", "browser": "confirm form fill"}.get(domain)
        if endpoint is None:
            return None
        try:
            import httpx

            return httpx.post(f"{_api_base()}{endpoint}",
                              json={"action_id": action_id, "phrase": phrase}, timeout=10.0).json()
        except Exception:
            return None

    def cancel_pending(self) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/pending/cancel", timeout=5.0).json()
        except Exception:
            return None


class DefaultMemoryApi:
    """Thin client for the /memory endpoints (Phase 5A). All calls are best-effort
    and never raise — the tray degrades gracefully when the API is down."""

    def status(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/memory/status", timeout=5.0).json().get("data")
        except Exception:
            return None

    def pending(self) -> dict | None:
        """The active pending action, but only when it is a memory domain."""
        try:
            import httpx

            summary = httpx.get(f"{_api_base()}/pending", timeout=5.0).json().get("pending")
        except Exception:
            return None
        if summary and str(summary.get("domain", "")).startswith("memory"):
            return summary
        return None

    def confirm(self, phrase: str) -> dict | None:
        """Confirm the pending memory with its exact phrase (a menu click is the
        explicit confirmation; wake never reaches here)."""
        try:
            import httpx

            return httpx.post(f"{_api_base()}/memory/confirm", json={"phrase": phrase}, timeout=10.0).json()
        except Exception:
            return None

    def cancel(self) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/memory/cancel", timeout=5.0).json()
        except Exception:
            return None

    def search(self, query: str) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/memory/search", json={"query": query}, timeout=10.0).json().get("data")
        except Exception:
            return None

    def list(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/memory/list", timeout=10.0).json().get("data")
        except Exception:
            return None


class DefaultTaskApi:
    """Thin client for the /tasks endpoints (Phase 5B). Best-effort, never raises."""

    def _get(self, path):
        try:
            import httpx

            return httpx.get(f"{_api_base()}{path}", timeout=10.0).json()
        except Exception:
            return None

    def _post(self, path, json=None):
        try:
            import httpx

            return httpx.post(f"{_api_base()}{path}", json=json or {}, timeout=15.0).json()
        except Exception:
            return None

    def list(self) -> list | None:
        body = self._get("/tasks")
        return body.get("tasks") if body else None

    def current(self) -> dict | None:
        """The most recent task (the tray's notion of the active task)."""
        tasks = self.list()
        if not tasks:
            return None
        top = tasks[0]
        detail = self._get(f"/tasks/{top['task_id']}")
        return detail.get("task") if detail else top

    def approve(self, task_id) -> dict | None:
        return self._post(f"/tasks/{task_id}/approve", {"phrase": "aprobar plan"})

    def resume(self, task_id, phrase="") -> dict | None:
        return self._post(f"/tasks/{task_id}/resume", {"phrase": phrase})

    def cancel(self, task_id) -> dict | None:
        return self._post(f"/tasks/{task_id}/cancel")

    def audit(self, task_id) -> dict | None:
        return self._get(f"/tasks/{task_id}/audit")


class DefaultScheduleApi:
    """Thin client for the /schedules endpoints (Phase 5C). Best-effort, never raises."""

    def _get(self, path):
        try:
            import httpx

            return httpx.get(f"{_api_base()}{path}", timeout=10.0).json()
        except Exception:
            return None

    def _post(self, path):
        try:
            import httpx

            return httpx.post(f"{_api_base()}{path}", timeout=15.0).json()
        except Exception:
            return None

    def list(self) -> list | None:
        body = self._get("/schedules")
        return (body or {}).get("data", {}).get("schedules") if body else None

    def pending(self) -> dict | None:
        """The active broker pending, when it belongs to a domain a scheduled task
        would use (a scheduled effect awaiting a fresh confirmation)."""
        try:
            import httpx

            summary = httpx.get(f"{_api_base()}/pending", timeout=5.0).json().get("pending")
        except Exception:
            return None
        return summary

    def pause(self, schedule_id):
        return self._post(f"/schedules/{schedule_id}/pause")

    def resume(self, schedule_id):
        return self._post(f"/schedules/{schedule_id}/resume")

    def run_now(self, schedule_id):
        return self._post(f"/schedules/{schedule_id}/run")

    def runs(self, schedule_id):
        body = self._get(f"/schedules/{schedule_id}/runs")
        return (body or {}).get("runs") if body else None


class DefaultActivityApi:
    """Thin client for the /activity endpoints (Phase 5D). The tray reads the SAME
    overview the web UI does, so their state stays consistent."""

    def url(self) -> str:
        return f"{_api_base()}/activity"

    def overview(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/activity/overview", timeout=5.0).json()
        except Exception:
            return None


class DefaultKnowledgeApi:
    """Thin client for the /knowledge endpoints (Phase 6A). Best-effort, never raises."""

    def url(self) -> str:
        return f"{_api_base()}/activity"          # the vault is inspected via the Activity Center UI

    def status(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/knowledge/status", timeout=5.0).json().get("data")
        except Exception:
            return None

    def upload(self, path: str) -> dict | None:
        try:
            import os
            import httpx

            with open(path, "rb") as fh:
                return httpx.post(f"{_api_base()}/knowledge/documents/upload",
                                  files={"file": (os.path.basename(path), fh.read())}, timeout=60.0).json()
        except Exception:
            return None

    def search(self, query: str) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/knowledge/search", json={"query": query},
                              timeout=15.0).json().get("data")
        except Exception:
            return None


class DefaultSecurityApi:
    """Thin client for the /security endpoints (Phase 6B). Best-effort, never raises.

    The tray only performs the local, non-typed security actions: read status, unlock
    via Windows Hello, lock, request a capability-scoped elevation, and revoke it. It
    never sends a password (the localhost/desktop page does that) and never changes the
    base profile (that stays behind the spoken 'confirmar cambio de perfil')."""

    def status(self) -> dict | None:
        try:
            import httpx

            return httpx.get(f"{_api_base()}/security/status", timeout=5.0).json().get("data")
        except Exception:
            return None

    def unlock(self, method: str = "windows_hello", secret: str = "") -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/security/unlock",
                              json={"method": method, "secret": secret}, timeout=10.0).json()
        except Exception:
            return None

    def lock(self) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/security/lock", timeout=5.0).json()
        except Exception:
            return None

    def request_elevation(self, capability: str) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/security/elevation/prepare",
                              json={"capability": capability}, timeout=10.0).json()
        except Exception:
            return None

    def revoke_elevation(self) -> dict | None:
        try:
            import httpx

            return httpx.post(f"{_api_base()}/security/elevation/revoke", timeout=5.0).json()
        except Exception:
            return None


# The exact confirmation phrase for each memory pending-domain (a menu click sends
# it verbatim; there is no bare-"yes" path).
MEMORY_DOMAIN_PHRASE = {
    "memory_write": "confirmar memoria",
    "memory_forget": "confirmar olvido",
    "memory_delete": "eliminar memoria permanentemente",
}


def render_memory_html(data: dict) -> str:
    """Read-only local memory viewer. Renders titles/types/dates + content — the
    user's own stored memories (secrets are never stored). No secrets, no tokens."""
    import html

    memories = (data or {}).get("memories", []) if data else []
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(m.get('type','')))}</td>"
        f"<td>{html.escape(str(m.get('title','')))}</td>"
        f"<td>{html.escape(str(m.get('content','')))}</td>"
        f"<td>{html.escape(str(m.get('date','')))}</td>"
        "</tr>"
        for m in memories
    )
    empty = "" if memories else "<p><em>No memories stored yet.</em></p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Fifi memory</title>"
        "<style>body{font-family:sans-serif;margin:2rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.35rem .8rem;"
        "text-align:left;vertical-align:top}th{background:#f3f3f3}</style></head><body>"
        f"<h2>Fifi memory <small>({len(memories)})</small></h2>"
        f"{empty}"
        "<table><tr><th>Type</th><th>Title</th><th>Content</th><th>Updated</th></tr>"
        f"{rows}</table>"
        "<p><em>Read-only. Sensitive data is blocked before storage and never shown here.</em></p>"
        "</body></html>"
    )


class TrayController:
    """State + orchestration for the tray. UI-agnostic and fully testable.

    ``run_runtime(argv) -> int`` shells local_runtime; ``notifier`` sends
    de-duplicated notifications; ``select_voice_fn`` / ``toggle_mute_fn`` /
    ``sleep`` are injectable. The controller records every runtime command it
    issues in ``self.issued`` so tests can prove it only ever stops owned work.
    """

    def __init__(self, *, policy=None, run_runtime=_default_runtime_runner, notifier=None,
                 select_voice_fn=None, toggle_mute_fn=None, on_update=None,
                 sleep=time.sleep, text_api=None, memory_api=None, task_api=None,
                 schedule_api=None, activity_api=None, knowledge_api=None,
                 security_api=None):
        self.policy = policy or StartupPolicy()
        self.run_runtime = run_runtime
        self.notifier = notifier or Notifier()
        self.select_voice_fn = select_voice_fn or self._default_select_voice
        self.toggle_mute_fn = toggle_mute_fn or self._default_toggle_mute
        self.on_update = on_update or (lambda state: None)
        self.sleep = sleep
        self.text_api = text_api or DefaultTextApi()
        self.memory_api = memory_api or DefaultMemoryApi()
        self.task_api = task_api or DefaultTaskApi()
        self.schedule_api = schedule_api or DefaultScheduleApi()
        self.activity_api = activity_api or DefaultActivityApi()
        self.knowledge_api = knowledge_api or DefaultKnowledgeApi()
        self.security_api = security_api or DefaultSecurityApi()
        self.state = ts.STOPPED
        self.issued: list[list[str]] = []
        self._running = True
        self._settings_server = None
        self._settings_url = None
        self._memory_server = None
        self._memory_url = None

    # -- helpers -------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self.state = state
        self.on_update(state)

    def _runtime(self, argv: list[str]) -> int:
        assert argv and argv[0] in ALLOWED_RUNTIME_COMMANDS, f"disallowed runtime cmd: {argv}"
        self.issued.append(list(argv))
        return self.run_runtime(argv)

    def _default_select_voice(self, profile: str) -> bool:
        try:
            return subprocess.run(
                [sys.executable, str(VOICE_LAB_CLI), "select", profile],
                cwd=str(PROJECT_ROOT),
            ).returncode == 0
        except OSError:
            return False

    def _default_toggle_mute(self) -> bool:
        try:
            import keyboard

            keyboard.send(llm_smoke.config_value("WAKE_WORD_MUTE_HOTKEY", "ctrl+alt+m"))
            return True
        except Exception:
            return False

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> bool:
        """Bring the stack up via local_runtime, with bounded Docker backoff."""
        if ts.is_running(self.state):
            return True
        self._set_state(ts.STARTING)
        for attempt in range(1, MAX_START_ATTEMPTS + 1):
            self._set_state(ts.WARMING)
            if self._runtime(["start"]) == 0:
                if self.policy.auto_warm_tts:
                    self._runtime(["voice-warm"])
                return self._enter_default_mode()
            if attempt < MAX_START_ATTEMPTS:
                self.notifier.notify(
                    "Fifi", f"Startup attempt {attempt} failed — retrying…",
                    key="start_retry",
                )
                self.sleep(START_BACKOFF_SECONDS * attempt)  # bounded backoff
        self._set_state(ts.ERROR)
        self.notifier.notify("Fifi", "Could not start after several attempts.", key="start_failed")
        return False

    def _enter_default_mode(self) -> bool:
        if self.policy.default_interaction_mode == "wake":
            if self._runtime(["wake-start"]) == 0:
                self._set_state(ts.LISTENING)
            else:
                self._set_state(ts.DEGRADED)
                self.notifier.notify("Fifi", "Ready, but wake mode did not start "
                                     "(push-to-talk still works).", key="wake_degraded")
        else:
            self._set_state(ts.READY)
        self.notifier.notify("Fifi", "Fifi is ready.", key="ready")
        return True

    def enable_wake_mode(self) -> bool:
        if not ts.is_running(self.state):
            return False
        ok = self._runtime(["wake-start"]) == 0
        self._set_state(ts.LISTENING if ok else ts.DEGRADED)
        return ok

    def enable_ptt_mode(self) -> bool:
        if not ts.is_running(self.state):
            return False
        self._runtime(["wake-stop"])   # stop hands-free listening; PTT is held-key
        self._set_state(ts.READY)
        return True

    def mute(self) -> bool:
        if self.state not in (ts.LISTENING, ts.READY):
            return False
        self.toggle_mute_fn()
        self._set_state(ts.MUTED)
        return True

    def unmute(self) -> bool:
        if self.state != ts.MUTED:
            return False
        self.toggle_mute_fn()
        self._set_state(ts.LISTENING)
        return True

    def set_runtime_mode(self, mode: str) -> bool:
        if mode not in ts.RUNTIME_MODES:
            return False
        return self._runtime(["voice-mode", mode]) == 0

    def select_voice(self, profile: str) -> bool:
        ok = self.select_voice_fn(profile)
        self.notifier.notify("Fifi", f"Voice set to {profile}." if ok
                             else f"Could not set voice {profile}.", key="voice")
        return ok

    def release_vram(self) -> bool:
        a = self._runtime(["unload"]) == 0
        b = self._runtime(["voice-unload"]) == 0
        return a and b

    def model_status(self) -> int:
        return self._runtime(["model-status"])

    def open_designer(self) -> int:
        return self._runtime(["voice-ui"])

    def open_settings(self, *, autostart_status_fn=None, server_factory=None, browser=None):
        """Open the read-only localhost settings page (no secrets shown)."""
        import webbrowser

        if autostart_status_fn is None:
            autostart_status_fn = lambda: fifi_autostart_status_enabled()  # noqa: E731
        snapshot = settings_snapshot(self.policy, autostart_status_fn())
        html = render_settings_html(snapshot)
        if self._settings_server is None:
            self._settings_server = (server_factory or SettingsServer)(html)
            self._settings_url = self._settings_server.start()
        else:
            self._settings_server.set_html(html)
        (browser or webbrowser.open)(self._settings_url)
        return self._settings_url

    def open_logs(self) -> bool:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(LOG_DIR))  # noqa: S606 — opens a folder, not a program
            else:
                subprocess.run(["xdg-open", str(LOG_DIR)])
            return True
        except Exception:
            return False

    # -- pending text draft (Phase 4A) ---------------------------------------------

    def view_pending_draft(self) -> dict | None:
        """Show the current pending text draft (does not confirm it)."""
        pending = self.text_api.pending()
        if pending:
            self.notifier.notify(
                "Fifi draft",
                f"{pending.get('character_count')} chars into "
                f"{pending.get('target_application')} — confirm or cancel from the menu.",
                key="draft_pending",
            )
        else:
            self.notifier.notify("Fifi", "No pending text draft.", key="draft_none")
        return pending

    def confirm_insertion(self) -> bool:
        """Clicking this menu item IS the explicit confirmation for insertion.

        Wake detection never reaches here — only a deliberate menu click does.
        """
        pending = self.text_api.pending()
        if not pending:
            self.notifier.notify("Fifi", "No pending draft to confirm.", key="draft_none")
            return False
        result = self.text_api.confirm(pending["action_id"], "insert text")
        status = (result or {}).get("status")
        self.notifier.notify("Fifi", f"Insertion {status}.", key="draft_confirm")
        return status in ("executed", "simulated")

    def cancel_insertion(self) -> bool:
        self.text_api.cancel()
        self.notifier.notify("Fifi", "Pending draft cancelled.", key="draft_cancel")
        return True

    # -- unified cross-domain pending action (Phase 4B.1) -------------------------

    def view_pending_action(self) -> dict | None:
        """Show the single pending sensitive action across any domain."""
        summary = self.text_api.pending_summary()
        if summary:
            self.notifier.notify(
                "Fifi pending",
                f"{summary.get('domain')} action for {summary.get('target')}. "
                f"Say '{summary.get('required_confirmation_phrase')}' or confirm from the menu.",
                key="pending_action",
            )
        else:
            self.notifier.notify("Fifi", "No pending action.", key="pending_none")
        return summary

    def confirm_pending(self) -> bool:
        """Confirm the active pending using its domain's exact phrase (a menu
        click is an explicit confirmation; wake never reaches here)."""
        summary = self.text_api.pending_summary()
        if not summary:
            self.notifier.notify("Fifi", "Nothing pending to confirm.", key="pending_none")
            return False
        result = self.text_api.confirm_domain(summary["domain"], summary["action_id"])
        status = (result or {}).get("status")
        self.notifier.notify("Fifi", f"{summary['domain']} action {status}.", key="pending_confirm")
        return status in ("executed", "simulated", "filled")

    def cancel_pending(self) -> bool:
        self.text_api.cancel_pending()
        self.notifier.notify("Fifi", "Pending action cancelled.", key="pending_cancel")
        return True

    # -- personal memory (Phase 5A) -----------------------------------------------

    def memory_status(self) -> dict | None:
        """Notify how many memories are stored (read-only, no content)."""
        data = self.memory_api.status()
        if data is None:
            self.notifier.notify("Fifi", "Memory status unavailable.", key="memory_status")
            return None
        counts = data.get("counts", {}) or {}
        self.notifier.notify(
            "Fifi memory",
            f"{counts.get('active', 0)} active, {counts.get('forgotten', 0)} forgotten.",
            key="memory_status",
        )
        return data

    def view_pending_memory(self) -> dict | None:
        """Show the pending memory proposal (does not confirm it)."""
        pending = self.memory_api.pending()
        if pending:
            self.notifier.notify(
                "Fifi memory",
                f"Pending {pending.get('domain')} — say "
                f"'{pending.get('required_confirmation_phrase')}' or confirm from the menu.",
                key="memory_pending",
            )
        else:
            self.notifier.notify("Fifi", "No pending memory.", key="memory_none")
        return pending

    def confirm_memory(self) -> bool:
        """Confirm the pending memory with its exact phrase. A menu click IS the
        explicit confirmation — wake detection never reaches this path."""
        pending = self.memory_api.pending()
        if not pending:
            self.notifier.notify("Fifi", "No pending memory to confirm.", key="memory_none")
            return False
        phrase = MEMORY_DOMAIN_PHRASE.get(pending.get("domain", ""))
        if not phrase:
            return False
        result = self.memory_api.confirm(phrase)
        status = (result or {}).get("status")
        self.notifier.notify("Fifi", f"Memory {status}.", key="memory_confirm")
        return status == "executed"

    def cancel_memory(self) -> bool:
        self.memory_api.cancel()
        self.notifier.notify("Fifi", "Pending memory cancelled.", key="memory_cancel")
        return True

    def search_memory(self, query: str) -> dict | None:
        data = self.memory_api.search(query)
        count = (data or {}).get("count", 0)
        self.notifier.notify("Fifi memory", f"{count} match(es) for '{query}'.", key="memory_search")
        return data

    def open_memory_viewer(self, *, server_factory=None, browser=None) -> str | None:
        """Open the read-only local memory viewer in the browser (no secrets)."""
        import webbrowser

        data = self.memory_api.list()
        if data is None:
            self.notifier.notify("Fifi", "Memory viewer unavailable.", key="memory_viewer")
            return None
        html = render_memory_html(data)
        if self._memory_server is None:
            self._memory_server = (server_factory or SettingsServer)(html)
            self._memory_url = self._memory_server.start()
        else:
            self._memory_server.set_html(html)
        (browser or webbrowser.open)(self._memory_url)
        return self._memory_url

    # -- multi-step tasks (Phase 5B) ----------------------------------------------

    def task_current(self) -> dict | None:
        task = self.task_api.current()
        if task:
            self.notifier.notify(
                "Fifi task", f"{task.get('title')} — {task.get('status')} "
                f"(step {task.get('current_step')}/{len(task.get('steps', []))}).",
                key="task_current")
        else:
            self.notifier.notify("Fifi", "No task.", key="task_none")
        return task

    def view_plan(self) -> dict | None:
        task = self.task_api.current()
        if not task:
            self.notifier.notify("Fifi", "No plan.", key="task_none")
            return None
        lines = ", ".join(
            f"{s['position']}:{s['intent']}[{s['risk_level']}]" for s in task.get("steps", []))
        self.notifier.notify("Fifi plan", lines or "(empty plan)", key="task_plan")
        return task

    def approve_task(self) -> bool:
        """A deliberate menu click approves the current plan (read-only + local
        steps may then begin; effects still need their own domain phrase)."""
        task = self.task_api.current()
        if not task:
            self.notifier.notify("Fifi", "No plan to approve.", key="task_none")
            return False
        result = self.task_api.approve(task["task_id"])
        self.notifier.notify("Fifi", f"Plan {(result or {}).get('status')}.", key="task_approve")
        return (result or {}).get("status") == "ok"

    def resume_task(self) -> dict | None:
        """Resume the current task. If it's paused for a confirmation, the reply
        names the exact domain phrase the user must say — the tray never confirms an
        effect on its own."""
        task = self.task_api.current()
        if not task:
            self.notifier.notify("Fifi", "No task to resume.", key="task_none")
            return None
        result = self.task_api.resume(task["task_id"])
        phrase = (result or {}).get("required_phrase")
        if phrase:
            self.notifier.notify("Fifi task", f"Say '{phrase}' to proceed.", key="task_resume")
        else:
            self.notifier.notify("Fifi task",
                                 f"Task {(result or {}).get('data', {}).get('task', {}).get('status')}.",
                                 key="task_resume")
        return result

    def cancel_task(self) -> bool:
        task = self.task_api.current()
        if not task:
            self.notifier.notify("Fifi", "No task to cancel.", key="task_none")
            return False
        self.task_api.cancel(task["task_id"])
        self.notifier.notify("Fifi", "Task cancelled (done steps are not undone).",
                             key="task_cancel")
        return True

    def task_pending_confirmation(self) -> dict | None:
        """Show the pending confirmation for the current task (does not confirm)."""
        task = self.task_api.current()
        if not task or task.get("status") != "awaiting_confirmation":
            self.notifier.notify("Fifi", "No task confirmation pending.", key="task_pending_none")
            return None
        step = next((s for s in task.get("steps", [])
                     if s.get("status") == "awaiting_confirmation"), {})
        self.notifier.notify(
            "Fifi task", f"Waiting for '{step.get('confirmation_phrase')}' "
            f"(step {step.get('position')}: {step.get('intent')}).", key="task_pending")
        return step

    def task_audit_summary(self) -> list | None:
        task = self.task_api.current()
        if not task:
            self.notifier.notify("Fifi", "No task audit.", key="task_none")
            return None
        body = self.task_api.audit(task["task_id"])
        events = (body or {}).get("events", [])
        self.notifier.notify(
            "Fifi task audit", ", ".join(e["event"] for e in events[:8]) or "(no events)",
            key="task_audit")
        return events

    # -- schedules + reminders (Phase 5C) -----------------------------------------

    def _next_schedule(self) -> dict | None:
        scheds = self.schedule_api.list() or []
        return scheds[0] if scheds else None

    def schedule_upcoming(self) -> list | None:
        scheds = self.schedule_api.list()
        if not scheds:
            self.notifier.notify("Fifi", "No upcoming schedules.", key="sched_none")
            return scheds
        lines = "; ".join(f"{s['title']} @ {(s.get('next_run_at') or '')[:16]}" for s in scheds[:5])
        self.notifier.notify("Fifi schedules", lines, key="sched_upcoming")
        return scheds

    def create_reminder(self) -> None:
        """The tray can't type — guide the user to the voice/API create flow, which
        always requires the 'confirmar programación' confirmation."""
        self.notifier.notify(
            "Fifi", "Say e.g. 'recuérdame llamar a Ana mañana a las nueve', then "
            "'confirmar programación'.", key="sched_create")

    def pause_schedule(self) -> bool:
        s = self._next_schedule()
        if not s:
            self.notifier.notify("Fifi", "No schedule to pause.", key="sched_none")
            return False
        result = self.schedule_api.pause(s["schedule_id"])
        self.notifier.notify("Fifi", f"Paused '{s['title']}'.", key="sched_pause")
        return (result or {}).get("status") == "ok"

    def resume_schedule(self) -> bool:
        for s in (self.schedule_api.list() or []):
            if s.get("status") == "paused":
                self.schedule_api.resume(s["schedule_id"])
                self.notifier.notify("Fifi", f"Resumed '{s['title']}'.", key="sched_resume")
                return True
        self.notifier.notify("Fifi", "No paused schedule.", key="sched_none")
        return False

    def run_schedule_now(self) -> dict | None:
        s = self._next_schedule()
        if not s:
            self.notifier.notify("Fifi", "No schedule to run.", key="sched_none")
            return None
        result = self.schedule_api.run_now(s["schedule_id"])
        self.notifier.notify("Fifi", f"Ran '{s['title']}' now.", key="sched_run")
        return result

    def pending_scheduled_effect(self) -> dict | None:
        """A scheduled task may have prepared an effect that awaits a fresh domain
        confirmation — surface the exact phrase (the tray never sends on its own)."""
        summary = self.schedule_api.pending()
        if summary:
            self.notifier.notify(
                "Fifi scheduled effect",
                f"{summary.get('domain')} awaits '{summary.get('required_confirmation_phrase')}'.",
                key="sched_pending")
        else:
            self.notifier.notify("Fifi", "No pending scheduled effect.", key="sched_pending_none")
        return summary

    def schedule_recent_results(self) -> list | None:
        scheds = self.schedule_api.list() or []
        results = [{"title": s["title"], "result": s.get("last_result", ""),
                    "last_run": s.get("last_run_at")} for s in scheds if s.get("last_run_at")]
        if results:
            self.notifier.notify(
                "Fifi results",
                "; ".join(f"{r['title']}: {r['result']}" for r in results[:5]),
                key="sched_results")
        else:
            self.notifier.notify("Fifi", "No recent schedule results.", key="sched_results_none")
        return results

    # -- Activity Center (Phase 5D) -----------------------------------------------

    def open_activity_center(self, *, browser=None) -> str:
        """Open the localhost Activity Center UI in the browser."""
        import webbrowser

        url = self.activity_api.url()
        (browser or webbrowser.open)(url)
        return url

    def activity_status(self) -> dict | None:
        """Notify a summary built from the SAME overview the web UI renders — so the
        tray and the Activity Center always show consistent state."""
        ov = self.activity_api.overview()
        if not ov or ov.get("error_code"):
            self.notifier.notify("Fifi", "Activity Center unavailable.", key="activity_status")
            return None
        pend = 1 if ov.get("pending") else 0
        task = ov.get("active_task")
        progress = task.get("progress") if task else "—"
        nxt = ov.get("next_schedule")
        next_title = nxt.get("title") if nxt else "none"
        errors = ov.get("recent_errors", 0)
        self.notifier.notify(
            "Fifi activity",
            f"{pend} pending · task {progress} · next: {next_title} · {errors} issue(s)",
            key="activity_status")
        return ov

    # -- Knowledge Vault (Phase 6A) -----------------------------------------------

    def open_knowledge_vault(self, *, browser=None) -> str:
        import webbrowser

        url = self.knowledge_api.url()
        (browser or webbrowser.open)(url)
        return url

    def import_document(self, *, chooser=None) -> dict | None:
        """Import a document the user explicitly selects (rejected server-side if
        unsupported/dangerous). ``chooser`` returns a path or None."""
        path = chooser() if chooser else None
        if not path:
            self.notifier.notify("Fifi", "No document selected.", key="knowledge_import")
            return None
        result = self.knowledge_api.upload(path)
        status = (result or {}).get("status")
        self.notifier.notify("Fifi Knowledge", f"Import {status or 'failed'}.", key="knowledge_import")
        return result

    def search_documents(self, query: str) -> dict | None:
        data = self.knowledge_api.search(query)
        count = (data or {}).get("count", 0)
        self.notifier.notify("Fifi Knowledge", f"{count} passage(s) for '{query}'.",
                             key="knowledge_search")
        return data

    def knowledge_status(self) -> dict | None:
        st = self.knowledge_api.status()
        if not st or st.get("enabled") is False:
            self.notifier.notify("Fifi", "Knowledge Vault is off.", key="knowledge_status")
            return st
        counts = st.get("counts", {}) or {}
        self.notifier.notify(
            "Fifi Knowledge",
            f"{counts.get('ready', 0)} indexed · embedder {st.get('embedder')}.",
            key="knowledge_status")
        return st

    # -- security profiles (Phase 6B) ---------------------------------------------

    def security_status(self) -> dict | None:
        """Notify the current profile, locked state, and any active elevation."""
        st = self.security_api.status()
        if not st:
            self.notifier.notify("Fifi", "Security status unavailable.", key="security_status")
            return None
        if st.get("enabled") is False:
            self.notifier.notify("Fifi", "Security profiles are off (standard access).",
                                 key="security_status")
            return st
        elev = st.get("elevation")
        elev_txt = f" · elevated: {elev.get('capability')}" if elev else ""
        state = "locked" if st.get("locked") else f"profile {st.get('profile')}"
        self.notifier.notify("Fifi security", f"{state}{elev_txt}.", key="security_status")
        return st

    def unlock_security(self) -> bool:
        """Unlock via Windows Hello (a menu click is an explicit local action; wake
        never reaches here). Password unlock is done on the localhost/desktop page —
        the tray never carries a typed secret."""
        result = self.security_api.unlock("windows_hello", "")
        status = (result or {}).get("status")
        if status == "ok":
            self.notifier.notify("Fifi security",
                                 f"Unlocked as {(result or {}).get('data', {}).get('profile')}.",
                                 key="security_unlock")
            return True
        self.notifier.notify(
            "Fifi security",
            "Could not unlock with Windows Hello — use the localhost security page for a password.",
            key="security_unlock")
        return False

    def lock_security(self) -> bool:
        """Lock immediately (drops any elevation)."""
        self.security_api.lock()
        self.notifier.notify("Fifi security", "Locked.", key="security_lock")
        return True

    def request_elevation(self, capability: str) -> dict | None:
        """Request a temporary, capability-scoped elevation (never global). Expires
        automatically; a real external effect still needs its own domain confirmation."""
        result = self.security_api.request_elevation(capability)
        status = (result or {}).get("status")
        if status == "ok":
            data = (result or {}).get("data", {}).get("elevation", {})
            self.notifier.notify(
                "Fifi security",
                f"Elevated for '{capability}' until {data.get('expires_at', 'soon')} "
                "(a confirmation is still required).", key="security_elevate")
        else:
            code = (result or {}).get("error_code", "unavailable")
            self.notifier.notify("Fifi security",
                                 f"Elevation for '{capability}' rejected ({code}).",
                                 key="security_elevate")
        return result

    def revoke_elevation(self) -> bool:
        self.security_api.revoke_elevation()
        self.notifier.notify("Fifi security", "Elevation revoked.", key="security_revoke")
        return True

    def stop(self) -> bool:
        """Stop ONLY owned processes; honour the keep-Docker-on-exit flags."""
        self._runtime(["wake-stop"])
        if not self.policy.keep_voice_lab_on_exit:
            self._runtime(["voice-stop"])
        stop_cmd = ["stop"]
        if not self.policy.keep_ollama_on_exit:
            stop_cmd.append("--stop-ollama")
        self._runtime(stop_cmd)
        self._set_state(ts.STOPPED)
        return True

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def exit(self) -> None:
        """Safe shutdown: stop owned work, then let the UI tear down the icon."""
        self.stop()
        self._running = False


# --- icon rendering (lazy Pillow) --------------------------------------------------


def render_icon(state: str, size: int = 64):
    """A simple status-dot icon (no assets). Lazy Pillow import."""
    from PIL import Image, ImageDraw

    style = ts.icon_style(state)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=style["color"] + (255,))
    return image


# --- pystray adapter (thin; not unit-tested) --------------------------------------


def run(argv: list[str] | None = None) -> int:  # pragma: no cover - requires a display
    import argparse

    parser = argparse.ArgumentParser(description="Fifi system-tray app (Windows host).")
    parser.add_argument("--no-autostart-check", action="store_true")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("The Fifi tray runs on the Windows host only.")
        return 1
    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        print("Tray dependencies missing. Install them on the host:")
        print("  .venv\\Scripts\\python.exe -m pip install -r requirements-desktop.txt")
        return 1

    lock = InstanceLock()
    if not lock.acquire():
        print(f"Another Fifi tray is already running (pid {lock.owner_pid()}).")
        return 1

    import pystray

    policy = StartupPolicy()
    icon = pystray.Icon("fifi", render_icon(ts.STOPPED), ts.icon_style(ts.STOPPED)["text"])

    def _on_update(state: str) -> None:
        icon.icon = render_icon(state)
        icon.title = ts.icon_style(state)["text"]
        icon.update_menu()

    controller = TrayController(policy=policy, on_update=_on_update)
    controller = _attach_menu(icon, controller)

    def _setup(icon):
        icon.visible = True
        if policy.auto_start:
            controller.start()

    try:
        icon.run(setup=_setup)
    finally:
        lock.release()
    return 0


def _attach_menu(icon, controller: "TrayController") -> "TrayController":  # pragma: no cover
    import pystray

    def enabled(key):
        return lambda item: ts.menu_enabled(controller.state).get(key, False)

    def action(fn):
        def handler(icon, item):
            fn()
            icon.update_menu()
        return handler

    profile_items = [
        pystray.MenuItem(name, action(lambda n=name: controller.select_voice(n)))
        for name in list_profiles()
    ]
    mode_items = [
        pystray.MenuItem(m, action(lambda mm=m: controller.set_runtime_mode(mm)))
        for m in ts.RUNTIME_MODES
    ]
    icon.menu = pystray.Menu(
        pystray.MenuItem("Start Fifi", action(controller.start), enabled=enabled(ts.START)),
        pystray.MenuItem("Stop Fifi", action(controller.stop), enabled=enabled(ts.STOP)),
        pystray.MenuItem("Restart Fifi", action(controller.restart), enabled=enabled(ts.RESTART)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Wake mode", action(controller.enable_wake_mode), enabled=enabled(ts.WAKE_MODE)),
        pystray.MenuItem("Push-to-talk mode", action(controller.enable_ptt_mode), enabled=enabled(ts.PTT_MODE)),
        pystray.MenuItem("Mute microphone", action(controller.mute), enabled=enabled(ts.MUTE)),
        pystray.MenuItem("Unmute microphone", action(controller.unmute), enabled=enabled(ts.UNMUTE)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("View pending action", action(controller.view_pending_action)),
        pystray.MenuItem("Confirm pending action", action(controller.confirm_pending)),
        pystray.MenuItem("Cancel pending action", action(controller.cancel_pending)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Memory", pystray.Menu(
            pystray.MenuItem("Memory status", action(controller.memory_status)),
            pystray.MenuItem("View pending memory", action(controller.view_pending_memory)),
            pystray.MenuItem("Confirm pending memory", action(controller.confirm_memory)),
            pystray.MenuItem("Cancel pending memory", action(controller.cancel_memory)),
            pystray.MenuItem("Open memory viewer", action(controller.open_memory_viewer)),
        )),
        pystray.MenuItem("Tasks", pystray.Menu(
            pystray.MenuItem("Current task", action(controller.task_current)),
            pystray.MenuItem("View plan", action(controller.view_plan)),
            pystray.MenuItem("Approve plan", action(controller.approve_task)),
            pystray.MenuItem("Resume task", action(controller.resume_task)),
            pystray.MenuItem("Cancel task", action(controller.cancel_task)),
            pystray.MenuItem("Pending confirmation", action(controller.task_pending_confirmation)),
            pystray.MenuItem("Audit summary", action(controller.task_audit_summary)),
        )),
        pystray.MenuItem("Schedules", pystray.Menu(
            pystray.MenuItem("Upcoming schedules", action(controller.schedule_upcoming)),
            pystray.MenuItem("Create reminder", action(controller.create_reminder)),
            pystray.MenuItem("Pause schedule", action(controller.pause_schedule)),
            pystray.MenuItem("Resume schedule", action(controller.resume_schedule)),
            pystray.MenuItem("Run schedule now", action(controller.run_schedule_now)),
            pystray.MenuItem("Pending scheduled effect", action(controller.pending_scheduled_effect)),
            pystray.MenuItem("Recent results", action(controller.schedule_recent_results)),
        )),
        pystray.MenuItem("Open Activity Center", action(controller.open_activity_center)),
        pystray.MenuItem("Activity status", action(controller.activity_status)),
        pystray.MenuItem("Knowledge", pystray.Menu(
            pystray.MenuItem("Open Knowledge Vault", action(controller.open_knowledge_vault)),
            pystray.MenuItem("Index status", action(controller.knowledge_status)),
        )),
        pystray.MenuItem("Security", pystray.Menu(
            pystray.MenuItem("Security status", action(controller.security_status)),
            pystray.MenuItem("Unlock (Windows Hello)", action(controller.unlock_security)),
            pystray.MenuItem("Lock now", action(controller.lock_security)),
            pystray.MenuItem("Request elevation", pystray.Menu(*[
                pystray.MenuItem(cap, action(lambda c=cap: controller.request_elevation(c)))
                for cap in ts.ELEVATION_CAPABILITIES])),
            pystray.MenuItem("Revoke elevation", action(controller.revoke_elevation)),
        )),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Voice profile", pystray.Menu(*profile_items) if profile_items else pystray.Menu(pystray.MenuItem("(none)", None, enabled=False))),
        pystray.MenuItem("Runtime mode", pystray.Menu(*mode_items)),
        pystray.MenuItem("Open Voice Designer", action(controller.open_designer)),
        pystray.MenuItem("Open logs", action(controller.open_logs)),
        pystray.MenuItem("Model status", action(controller.model_status)),
        pystray.MenuItem("Release VRAM", action(controller.release_vram), enabled=enabled(ts.RELEASE_VRAM)),
        pystray.MenuItem("Settings", action(controller.open_settings)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Exit", action(lambda: (controller.exit(), icon.stop()))),
    )
    return controller


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
