"""Text-to-speech.

Engine selected via TTS_ENGINE:
- "windows": Windows SAPI through pyttsx3 (optional dependency from
  requirements-voice.txt). Falls back to simulation when not installed.
- "simulated": always simulate.

TTS never blocks or breaks the API: with ENABLE_VOICE=false, a missing
dependency, or an engine failure, speak() returns a structured result
instead of raising.
"""

from typing import Any

from app.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

try:
    import pyttsx3
except ImportError:  # optional dependency
    pyttsx3 = None

INSTALL_HINT = (
    "pyttsx3 is not installed. Install the optional voice dependencies: "
    "pip install -r requirements-voice.txt"
)


class TextToSpeechService:
    def __init__(self, engine: str | None = None) -> None:
        self.engine = (engine or get_settings().tts_engine).lower()

    def available(self) -> bool:
        return self.engine == "windows" and pyttsx3 is not None

    def _simulate(self, text: str, note: str) -> dict[str, Any]:
        return {
            "simulated": True,
            "action": "speak",
            "would_do": f"Speak with {self.engine}: {text!r}",
            "note": note,
        }

    def speak(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"action": "speak", "error": "Empty text."}

        if not get_settings().enable_voice:
            return self._simulate(text, "Voice is disabled (ENABLE_VOICE=false).")
        if self.engine != "windows":
            return self._simulate(text, f"TTS engine {self.engine!r} has no real backend yet.")
        if pyttsx3 is None:
            return self._simulate(text, INSTALL_HINT)

        try:
            driver = pyttsx3.init()
            driver.say(text)
            driver.runAndWait()
        except Exception as exc:  # audio failures must not crash the API
            log.exception("TTS failed")
            return {"action": "speak", "error": f"TTS failed: {exc}"}
        return {"simulated": False, "action": "speak", "spoke": text, "engine": "windows"}
