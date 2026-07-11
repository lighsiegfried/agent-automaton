import os
import sys
from pathlib import Path

import pytest

from app.tools.browser_tools import search_web
from app.tools.windows_tools import open_app, open_folder, validate_folder

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only behavior")


# --- app allowlist -----------------------------------------------------------


def test_open_app_rejects_unknown_app(settings):
    result = open_app(app="photoshop")
    assert "not in the allowlist" in result["error"]


def test_open_app_rejects_empty_name(settings):
    assert "error" in open_app(app="  ")


def test_open_app_spanish_alias_maps_to_canonical(settings):
    settings.enable_real_windows_tools = False
    result = open_app(app="calculadora")
    assert result["simulated"] is True
    assert "calculator" in result["would_do"]


def test_open_app_respects_custom_allowlist(settings):
    settings.allowed_apps = "notepad"
    assert "error" not in open_app(app="notepad")
    assert "error" in open_app(app="chrome")


@windows_only
def test_open_app_real_mode_launches_allowlisted(settings, launches):
    settings.enable_real_windows_tools = True
    result = open_app(app="calculator")
    assert result["simulated"] is False
    assert launches.popen_calls == [["cmd", "/c", "start", "", "calc"]]


@windows_only
def test_open_app_real_mode_still_rejects_unknown(settings, launches):
    settings.enable_real_windows_tools = True
    result = open_app(app="regedit")
    assert "error" in result
    assert launches.popen_calls == []


# --- folder safety -----------------------------------------------------------


def test_open_folder_rejects_missing_folder(settings):
    result = open_folder(path=str(Path("C:/definitely/not/a/real/folder/xyz123")))
    assert "does not exist" in result["error"]


@windows_only
def test_open_folder_rejects_system_locations(settings):
    for env_var in ("SystemRoot", "ProgramFiles"):
        root = os.environ.get(env_var)
        assert root, f"{env_var} not set"
        assert "refused" in open_folder(path=root)["error"]
    system32 = str(Path(os.environ["SystemRoot"]) / "System32")
    assert "refused" in open_folder(path=system32)["error"]


def test_validate_folder_defaults_to_home():
    path, error = validate_folder("")
    assert error is None
    assert path == Path.home().resolve()


def test_validate_folder_resolves_spanish_alias():
    path, error = validate_folder("descargas")
    if error is not None:
        pytest.skip("no Downloads folder on this machine")
    assert path.name == "Downloads"


# --- simulated vs real mode --------------------------------------------------


def test_simulated_mode_never_launches_anything(settings, launches):
    settings.enable_real_windows_tools = False
    open_app(app="notepad")
    open_folder(path=str(Path.home()))
    search_web(query="cats")
    assert launches.popen_calls == []
    assert launches.startfile_calls == []
    assert launches.browser_urls == []


@windows_only
def test_open_folder_real_mode_opens_validated_path(settings, launches):
    settings.enable_real_windows_tools = True
    result = open_folder(path=str(Path.home()))
    assert result["simulated"] is False
    assert launches.startfile_calls == [str(Path.home().resolve())]


@windows_only
def test_search_web_real_mode_opens_browser(settings, launches):
    settings.enable_real_windows_tools = True
    result = search_web(query="local voice models")
    assert result["simulated"] is False
    assert launches.browser_urls == ["https://duckduckgo.com/?q=local+voice+models"]


def test_search_web_rejects_empty_query(settings):
    assert "error" in search_web(query="   ")
