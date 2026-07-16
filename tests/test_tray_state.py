"""Pure tray state + menu model (Phase 3E)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import tray_state as ts  # noqa: E402


def test_all_nine_states_present():
    assert set(ts.STATES) == {
        "stopped", "starting", "warming", "ready", "listening", "muted",
        "processing", "degraded", "error",
    }


def test_icon_style_covers_every_state():
    for state in ts.STATES:
        style = ts.icon_style(state)
        assert len(style["color"]) == 3
        assert style["text"].startswith("Fifi:")


def test_running_and_busy_classification():
    assert ts.is_running(ts.LISTENING) and ts.is_running(ts.READY)
    assert not ts.is_running(ts.STOPPED)
    assert ts.is_busy(ts.STARTING) and ts.is_busy(ts.WARMING)
    assert not ts.is_busy(ts.READY)


def test_menu_enabled_stopped():
    en = ts.menu_enabled(ts.STOPPED)
    assert en[ts.START] is True
    assert en[ts.STOP] is False
    assert en[ts.MUTE] is False
    assert en[ts.WAKE_MODE] is False
    assert en[ts.EXIT] is True          # exit is never locked out


def test_menu_enabled_listening_offers_mute_not_unmute():
    en = ts.menu_enabled(ts.LISTENING)
    assert en[ts.MUTE] is True
    assert en[ts.UNMUTE] is False
    assert en[ts.START] is False
    assert en[ts.RELEASE_VRAM] is True


def test_menu_enabled_muted_offers_unmute():
    en = ts.menu_enabled(ts.MUTED)
    assert en[ts.UNMUTE] is True
    assert en[ts.MUTE] is False


def test_menu_disabled_while_busy():
    en = ts.menu_enabled(ts.STARTING)
    assert en[ts.START] is False
    assert en[ts.STOP] is False
    assert en[ts.WAKE_MODE] is False
    assert en[ts.EXIT] is True
    assert en[ts.MODEL_STATUS] is True  # read-only always available


def test_menu_enabled_error_allows_restart_paths():
    en = ts.menu_enabled(ts.ERROR)
    assert en[ts.START] is True
