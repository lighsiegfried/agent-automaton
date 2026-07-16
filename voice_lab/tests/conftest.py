"""Voice Lab test fixtures.

This suite is run SEPARATELY from the main suite (pytest.ini scopes the main
run to tests/):

    python -m pytest voice_lab/tests

voice_lab has its own `app` package; inserting voice_lab/ first makes
`import app` resolve here, never to the main project. Everything heavy (GPU,
engines, audio devices, downloads, playback) is mocked — the fixtures below
provide fake engines and a temporary profiles directory.
"""

import json
import os
import sys
import wave
from pathlib import Path

VOICE_LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VOICE_LAB_ROOT))

# Determinism: env vars beat voice_lab/.env in pydantic-settings. The user may
# customize the local preview sentence; the suite always tests the canonical
# Phase 3D.0.1 comparison text. Pinned BEFORE settings are ever constructed.
os.environ["VOICE_LAB_PREVIEW_TEXT"] = (
    "Hola, soy Fifi. El sistema está listo y puedo ayudarte con tus tareas. "
    "¿En qué trabajaremos hoy?"
)
# The suite must not depend on the machine's voice_lab/.env: pin the resource
# thresholds and the runtime mode to their canonical Phase 3D.0.5 values, and
# DISABLE the background idle unloader (tests drive idle_unload_tick directly).
os.environ["VOICE_LAB_MIN_FREE_SYSTEM_RAM_GB"] = "2.0"
os.environ["VOICE_LAB_MIN_FREE_VRAM_GB"] = "2.0"
os.environ["VOICE_LAB_MIN_FREE_DISK_GB"] = "5.0"
os.environ["VOICE_LAB_MODE"] = "daily"
os.environ["VOICE_LAB_IDLE_UNLOAD_SECONDS"] = "0"
os.environ["VOICE_LAB_MAX_LOADED_HEAVY_MODELS"] = "1"
os.environ["VOICE_LAB_ALLOW_KOKORO_FALLBACK"] = "true"
os.environ["VOICE_LAB_DESIGNER_AUTO_UNLOAD_OLLAMA"] = "false"

import pytest  # noqa: E402

from app.engines.base import TtsEngine  # noqa: E402
from app.profiles.manager import ProfileManager  # noqa: E402

PROJECT_ROOT = VOICE_LAB_ROOT.parent


def write_wav(path: Path, seconds: float = 0.3, rate: int = 24000) -> None:
    """A tiny, valid, non-silent WAV (stdlib only — no numpy/soundfile)."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(rate * seconds))


class FakeEngine(TtsEngine):
    """Deterministic engine double: no GPU, no downloads, no audio device."""

    def __init__(self, name: str, *, dep_ok: bool = True, fail: str = "") -> None:
        super().__init__()
        self.name = name
        self.dep_ok = dep_ok
        self.fail = fail  # "" | "load" | "synthesize" | "oom"
        self.load_calls = 0
        self.synth_calls = 0
        self.texts: list[str] = []

    def available(self):
        return (self.dep_ok, "" if self.dep_ok else f"{self.name} deps missing")

    def _load(self, profile):
        self.load_calls += 1
        if self.fail == "load":
            from app.engines.base import EngineUnavailable

            raise EngineUnavailable(f"{self.name} model missing")

    def _synthesize(self, text, profile, out_path):
        self.synth_calls += 1
        self.texts.append(text)
        if self.fail == "synthesize":
            raise RuntimeError(f"{self.name} exploded")
        if self.fail == "oom":
            from app.engines.base import EngineUnavailable

            raise EngineUnavailable(f"GPU out of memory while running {self.name}")
        write_wav(out_path)
        return {"sample_rate": 24000, "device": "cpu"}


PROFILE_TEMPLATE = {
    "engine": "kokoro",
    "model": "hexgrad/Kokoro-82M",
    "language": "en",
    "speaker": "af_heart",
    "speed": 1.0,
    "style_instruction": "",
    "sample_rate": 24000,
    "device": "cuda",
    "dtype": "float16",
    "fallback_engine": "windows_sapi",
    "provenance": {"source": "test"},
}


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    """A temporary config/voices/ with two profiles; active.json not yet written."""
    profiles = tmp_path / "voices" / "profiles"
    profiles.mkdir(parents=True)
    for name in ("fifi_warm", "fifi_calm"):
        payload = {**PROFILE_TEMPLATE, "name": name}
        (profiles / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path / "voices"


@pytest.fixture
def manager(voices_dir: Path) -> ProfileManager:
    return ProfileManager(
        profiles_dir=voices_dir / "profiles", active_path=voices_dir / "active.json"
    )


@pytest.fixture(autouse=True)
def fresh_coordinator(tmp_path, monkeypatch):
    """Every test gets a clean coordinator on a temp state dir (mode=daily from
    the pinned env), never the machine's persisted mode. When app.main is loaded
    (worker tests), point its module-global `coordinator` at the fresh one so the
    endpoints and the gate use the isolated instance."""
    from app import coordinator as coord_mod

    fresh = coord_mod.reset_coordinator(state_dir=tmp_path / "coordinator")
    worker = sys.modules.get("app.main")
    if worker is not None:
        monkeypatch.setattr(worker, "coordinator", fresh, raising=False)
    return fresh
