import pytest

from app.config import get_settings
from app.tools.registry import load_tools

load_tools()


@pytest.fixture(scope="session", autouse=True)
def _test_db(tmp_path_factory):
    """Point the command log at a throwaway database for the whole run."""
    settings = get_settings()
    original = settings.db_path
    settings.db_path = tmp_path_factory.mktemp("storage") / "test.db"
    yield
    settings.db_path = original


@pytest.fixture
def settings():
    """The (cached) settings object; mutations are restored after the test."""
    s = get_settings()
    saved = (
        s.enable_real_windows_tools,
        s.require_confirmation,
        s.allowed_apps,
        s.enable_llm_planner,
        s.enable_voice,
        s.tts_engine,
        s.enable_response_generator,
        s.voice_speak_command_response,
        s.agent_name,
    )
    yield s
    (
        s.enable_real_windows_tools,
        s.require_confirmation,
        s.allowed_apps,
        s.enable_llm_planner,
        s.enable_voice,
        s.tts_engine,
        s.enable_response_generator,
        s.voice_speak_command_response,
        s.agent_name,
    ) = saved


class LaunchRecorder:
    """Records what would have been launched instead of launching it."""

    def __init__(self) -> None:
        self.startfile_calls: list[str] = []
        self.popen_calls: list[list[str]] = []
        self.browser_urls: list[str] = []


@pytest.fixture(autouse=True)
def launches(monkeypatch):
    """Stub every launch primitive so no test can ever start a real app."""
    recorder = LaunchRecorder()
    monkeypatch.setattr(
        "os.startfile",
        lambda path: recorder.startfile_calls.append(str(path)),
        raising=False,  # os.startfile only exists on Windows
    )
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda args, **kwargs: recorder.popen_calls.append(list(args)),
    )
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url, **kwargs: recorder.browser_urls.append(url),
    )
    return recorder
