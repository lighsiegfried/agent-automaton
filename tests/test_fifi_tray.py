"""Tray controller, single-instance lock, notifier, policy, settings (Phase 3E).

The pystray/Pillow UI is NOT exercised here (it needs a display); every piece of
behaviour lives in importable classes with injected dependencies.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fifi_tray  # noqa: E402
import tray_state as ts  # noqa: E402


# --- helpers -----------------------------------------------------------------------


def make_policy(**over):
    env = {
        "FIFI_AUTO_START": "false",
        "FIFI_STARTUP_DELAY_SECONDS": "20",
        "FIFI_DEFAULT_INTERACTION_MODE": "wake",
        "FIFI_AUTO_WARM_TTS": "true",
        "FIFI_KEEP_OLLAMA_RUNNING_ON_EXIT": "false",
        "FIFI_KEEP_VOICE_LAB_RUNNING_ON_EXIT": "false",
    }
    env.update({k: str(v) for k, v in over.items()})
    return fifi_tray.StartupPolicy(get=lambda name, default="": env.get(name, default))


class FakeRuntime:
    """Records issued local_runtime argvs; returns per-command exit codes."""

    def __init__(self, codes=None, default=0):
        self.codes = codes or {}
        self.default = default
        self.calls: list = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.codes.get(argv[0], self.default)


class RecordingNotifier:
    def __init__(self):
        self.sent: list = []

    def notify(self, title, message, key=None):
        self.sent.append({"title": title, "message": message, "key": key})
        return True


def make_controller(runtime=None, policy=None, **kw):
    return fifi_tray.TrayController(
        policy=policy or make_policy(),
        run_runtime=runtime or FakeRuntime(),
        notifier=RecordingNotifier(),
        select_voice_fn=kw.get("select_voice_fn", lambda p: True),
        toggle_mute_fn=kw.get("toggle_mute_fn", lambda: True),
        sleep=kw.get("sleep", lambda s: None),
    )


# --- StartupPolicy -----------------------------------------------------------------


def test_startup_policy_parses_env():
    policy = make_policy(FIFI_AUTO_START="true", FIFI_STARTUP_DELAY_SECONDS="30",
                         FIFI_DEFAULT_INTERACTION_MODE="ptt")
    assert policy.auto_start is True
    assert policy.startup_delay_seconds == 30
    assert policy.default_interaction_mode == "ptt"
    assert policy.keep_ollama_on_exit is False


def test_startup_policy_bad_mode_defaults_wake():
    assert make_policy(FIFI_DEFAULT_INTERACTION_MODE="garbage").default_interaction_mode == "wake"


# --- InstanceLock ------------------------------------------------------------------


def test_instance_lock_acquire_and_release(tmp_path):
    lock = fifi_tray.InstanceLock(path=tmp_path / "t.lock", pid=111, alive=lambda p: True)
    assert lock.acquire() is True
    assert lock.owner_pid() == 111
    lock.release()
    assert lock.owner_pid() is None


def test_instance_lock_blocks_second_live_instance(tmp_path):
    path = tmp_path / "t.lock"
    fifi_tray.InstanceLock(path=path, pid=111, alive=lambda p: True).acquire()
    other = fifi_tray.InstanceLock(path=path, pid=222, alive=lambda p: True)
    assert other.held_by_other() is True
    assert other.acquire() is False   # a live tray already holds it


def test_instance_lock_reclaims_stale(tmp_path):
    path = tmp_path / "t.lock"
    path.write_text("999", encoding="utf-8")           # dead previous tray
    lock = fifi_tray.InstanceLock(path=path, pid=222, alive=lambda p: False)
    assert lock.held_by_other() is False
    assert lock.acquire() is True
    assert lock.owner_pid() == 222


# --- Notifier ----------------------------------------------------------------------


def test_notifier_dedupes_within_window():
    sent = []
    clock = {"t": 0.0}
    n = fifi_tray.Notifier(backend=lambda t, m: sent.append(m),
                           clock=lambda: clock["t"], min_repeat_seconds=60)
    assert n.notify("Fifi", "ready", key="ready") is True
    assert n.notify("Fifi", "ready", key="ready") is False   # deduped
    clock["t"] = 61.0
    assert n.notify("Fifi", "ready", key="ready") is True    # window elapsed
    assert sent == ["ready", "ready"]


def test_notifier_distinct_keys_not_deduped():
    n = fifi_tray.Notifier(backend=lambda t, m: None, clock=lambda: 0.0)
    assert n.notify("Fifi", "a", key="a") is True
    assert n.notify("Fifi", "b", key="b") is True


# --- settings snapshot -------------------------------------------------------------


def test_render_settings_html_shows_only_given_values():
    html = fifi_tray.render_settings_html(
        {"active_voice": "fifi_warm", "wake_threshold": "0.6", "real_windows_tools": "false"}
    )
    assert "fifi_warm" in html and "0.6" in html
    assert "Read-only" in html
    # Renders ONLY the snapshot it was given — a value not in the dict can't leak.
    assert "sk-supersecret-token" not in html


def test_open_settings_starts_server_and_opens_browser():
    opened = []

    class FakeServer:
        def __init__(self, html):
            self.html = html

        def start(self):
            return "http://127.0.0.1:9999"

        def set_html(self, html):
            self.html = html

    c = make_controller()
    url = c.open_settings(
        autostart_status_fn=lambda: True,
        server_factory=FakeServer,
        browser=lambda u: opened.append(u),
    )
    assert url == "http://127.0.0.1:9999"
    assert opened == ["http://127.0.0.1:9999"]
    # Reopening reuses the same server (does not bind a new port).
    c.open_settings(autostart_status_fn=lambda: True, server_factory=FakeServer,
                    browser=lambda u: opened.append(u))
    assert opened == ["http://127.0.0.1:9999", "http://127.0.0.1:9999"]


def test_settings_snapshot_has_no_secrets():
    env = {"WAKE_WORD_THRESHOLD": "0.6", "ENABLE_REAL_WINDOWS_TOOLS": "false"}
    snap = fifi_tray.settings_snapshot(
        make_policy(), autostart_enabled=True, get=lambda k, d="": env.get(k, d)
    )
    assert snap["wake_threshold"] == "0.6"
    assert snap["autostart_enabled"] is True
    assert snap["real_windows_tools"] == "false"
    # No secret-bearing keys are exposed.
    for key in snap:
        assert "token" not in key and "password" not in key


# --- controller: startup -----------------------------------------------------------


def test_start_happy_path_enters_wake_mode():
    runtime = FakeRuntime()
    c = make_controller(runtime)
    assert c.start() is True
    assert c.state == ts.LISTENING
    assert ["start"] in c.issued
    assert ["wake-start"] in c.issued
    assert any(n["key"] == "ready" for n in c.notifier.sent)


def test_start_ptt_mode_ends_ready():
    c = make_controller(FakeRuntime(), policy=make_policy(FIFI_DEFAULT_INTERACTION_MODE="ptt"))
    assert c.start() is True
    assert c.state == ts.READY
    assert ["wake-start"] not in c.issued


def test_start_retries_bounded_then_errors():
    runtime = FakeRuntime(codes={"start": 1})   # start always fails
    sleeps = []
    c = make_controller(runtime, sleep=lambda s: sleeps.append(s))
    assert c.start() is False
    assert c.state == ts.ERROR
    starts = [cmd for cmd in c.issued if cmd == ["start"]]
    assert len(starts) == fifi_tray.MAX_START_ATTEMPTS      # bounded, never infinite
    assert len(sleeps) == fifi_tray.MAX_START_ATTEMPTS - 1  # backoff between attempts
    assert sleeps == sorted(sleeps)                          # increasing backoff
    assert any(n["key"] == "start_failed" for n in c.notifier.sent)


def test_start_degraded_when_wake_fails():
    c = make_controller(FakeRuntime(codes={"wake-start": 1}))
    assert c.start() is True
    assert c.state == ts.DEGRADED


# --- controller: mic + modes -------------------------------------------------------


def test_mute_and_unmute_toggle_state():
    toggles = []
    c = make_controller(toggle_mute_fn=lambda: toggles.append(1) or True)
    c.start()
    assert c.state == ts.LISTENING
    assert c.mute() is True and c.state == ts.MUTED
    assert c.mute() is False                       # already muted
    assert c.unmute() is True and c.state == ts.LISTENING
    assert len(toggles) == 2


def test_set_runtime_mode_valid_and_invalid():
    c = make_controller(FakeRuntime())
    assert c.set_runtime_mode("daily") is True
    assert ["voice-mode", "daily"] in c.issued
    assert c.set_runtime_mode("nonsense") is False   # rejected, no command issued
    assert ["voice-mode", "nonsense"] not in c.issued


def test_select_voice_calls_selector_and_notifies():
    picked = []
    c = make_controller(select_voice_fn=lambda p: picked.append(p) or True)
    assert c.select_voice("fifi_warm") is True
    assert picked == ["fifi_warm"]
    assert any(n["key"] == "voice" for n in c.notifier.sent)


def test_release_vram_unloads_both():
    c = make_controller(FakeRuntime())
    assert c.release_vram() is True
    assert ["unload"] in c.issued and ["voice-unload"] in c.issued


# --- controller: safe shutdown + ownership -----------------------------------------


def test_stop_only_issues_owned_stop_commands():
    c = make_controller(FakeRuntime())
    c.stop()
    assert c.state == ts.STOPPED
    assert ["wake-stop"] in c.issued
    assert ["voice-stop"] in c.issued
    assert ["stop", "--stop-ollama"] in c.issued


def test_keep_flags_preserve_docker_services():
    c = make_controller(FakeRuntime(),
                        policy=make_policy(FIFI_KEEP_OLLAMA_RUNNING_ON_EXIT="true",
                                           FIFI_KEEP_VOICE_LAB_RUNNING_ON_EXIT="true"))
    c.stop()
    assert ["voice-stop"] not in c.issued           # Voice Lab left running
    assert ["stop"] in c.issued                      # plain stop, no --stop-ollama
    assert ["stop", "--stop-ollama"] not in c.issued


def test_exit_stops_then_marks_not_running():
    c = make_controller(FakeRuntime())
    c.start()
    c.exit()
    assert c.state == ts.STOPPED
    assert c._running is False


def test_controller_only_ever_issues_allowlisted_commands():
    """No tray action can invoke anything outside the owned-stop allowlist."""
    c = make_controller(FakeRuntime())
    c.start(); c.mute(); c.unmute(); c.set_runtime_mode("daily")
    c.release_vram(); c.model_status(); c.restart(); c.exit()
    for cmd in c.issued:
        assert cmd[0] in fifi_tray.ALLOWED_RUNTIME_COMMANDS


def test_runtime_rejects_disallowed_command():
    c = make_controller(FakeRuntime())
    with pytest.raises(AssertionError):
        c._runtime(["rm", "-rf", "/"])   # anything outside the allowlist is refused


# --- pending-draft integration (Phase 4A) ------------------------------------------


class FakeTextApi:
    def __init__(self, pending=None):
        self._pending = pending
        self.confirmed = []
        self.cancelled = False

    def pending(self):
        return self._pending

    def confirm(self, action_id, phrase):
        self.confirmed.append((action_id, phrase))
        return {"status": "simulated"}

    def cancel(self):
        self.cancelled = True
        return {"status": "cancelled"}


def _controller_with_text(api):
    return fifi_tray.TrayController(policy=make_policy(), run_runtime=FakeRuntime(),
                                    notifier=RecordingNotifier(), text_api=api)


def test_view_pending_draft_reports_summary():
    api = FakeTextApi(pending={"character_count": 42, "target_application": "notepad",
                               "action_id": "id1"})
    c = _controller_with_text(api)
    assert c.view_pending_draft()["action_id"] == "id1"
    assert any(n["key"] == "draft_pending" for n in c.notifier.sent)


def test_confirm_insertion_sends_exact_phrase():
    api = FakeTextApi(pending={"action_id": "id1", "character_count": 5,
                               "target_application": "notepad"})
    c = _controller_with_text(api)
    assert c.confirm_insertion() is True
    assert api.confirmed == [("id1", "insert text")]   # explicit menu click confirms


def test_confirm_insertion_without_pending_is_noop():
    api = FakeTextApi(pending=None)
    c = _controller_with_text(api)
    assert c.confirm_insertion() is False
    assert api.confirmed == []


def test_cancel_insertion():
    api = FakeTextApi(pending={"action_id": "id1"})
    c = _controller_with_text(api)
    assert c.cancel_insertion() is True
    assert api.cancelled is True


# --- unified cross-domain pending (Phase 4B.1) -------------------------------------


class FakeUnifiedApi:
    def __init__(self, summary=None):
        self._summary = summary
        self.confirmed = []
        self.cancelled = False

    def pending_summary(self):
        return self._summary

    def confirm_domain(self, domain, action_id):
        self.confirmed.append((domain, action_id))
        return {"status": "filled" if domain == "browser" else "simulated"}

    def cancel_pending(self):
        self.cancelled = True
        return {"status": "cancelled"}


def test_view_pending_action_shows_domain_and_phrase():
    api = FakeUnifiedApi(summary={"domain": "browser", "action_id": "form1",
                                  "target": "Sign up", "required_confirmation_phrase": "confirmar formulario"})
    c = _controller_with_text(api)
    assert c.view_pending_action()["domain"] == "browser"
    assert any(n["key"] == "pending_action" for n in c.notifier.sent)


def test_confirm_pending_uses_domain_phrase():
    api = FakeUnifiedApi(summary={"domain": "browser", "action_id": "form1",
                                  "target": "Sign up", "required_confirmation_phrase": "x"})
    c = _controller_with_text(api)
    assert c.confirm_pending() is True
    assert api.confirmed == [("browser", "form1")]   # tray never guesses the phrase


def test_confirm_pending_without_pending_is_noop():
    c = _controller_with_text(FakeUnifiedApi(summary=None))
    assert c.confirm_pending() is False


def test_cancel_pending_hits_broker():
    api = FakeUnifiedApi(summary={"domain": "text", "action_id": "id1"})
    c = _controller_with_text(api)
    assert c.cancel_pending() is True
    assert api.cancelled is True
