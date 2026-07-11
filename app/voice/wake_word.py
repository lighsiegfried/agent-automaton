"""Wake word detection — PLACEHOLDER (Phase 3C).

Planned implementation: openWakeWord (or Porcupine) listening on the default
microphone and triggering the STT pipeline when the wake word ("fifi",
configurable via WAKE_WORD) is heard. Intentionally inactive: nothing
instantiates this class yet and no microphone is ever opened.
"""

from typing import Any

from app.config import get_settings


class WakeWordDetector:
    def __init__(self, wake_word: str | None = None) -> None:
        self.wake_word = wake_word or get_settings().wake_word

    def listen(self) -> dict[str, Any]:
        """Placeholder: returns what it would do instead of listening."""
        return {
            "simulated": True,
            "action": "wake_word_listen",
            "would_do": f"Listen on the microphone for the wake word {self.wake_word!r}",
        }
