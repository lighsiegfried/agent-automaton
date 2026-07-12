import os

# Determinism: the suite is written for the code defaults (optional features
# OFF), with tests flipping a flag when they need it on. A developer's local
# .env may enable voice / planner / response for live use — env vars take
# precedence over .env in pydantic-settings, so pin the defaults here BEFORE the
# Settings object is ever constructed. This keeps the suite independent of the
# machine's .env without weakening any test.
for _flag in (
    "ENABLE_VOICE",
    "ENABLE_LLM_PLANNER",
    "ENABLE_RESPONSE_GENERATOR",
    "ENABLE_REAL_WINDOWS_TOOLS",
    "VOICE_SPEAK_COMMAND_RESPONSE",
    "ENABLE_WAKE_WORD",
):
    os.environ[_flag] = "false"
# The suite is written for the "windows" TTS engine; a developer's .env may
# select "voice_lab" for live use — pin it like the flags above.
os.environ["TTS_ENGINE"] = "windows"

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.tools.registry import load_tools  # noqa: E402

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
        s.stt_device,
        s.stt_compute_type,
        s.stt_allow_cpu_fallback,
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
        s.stt_device,
        s.stt_compute_type,
        s.stt_allow_cpu_fallback,
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
