"""Text/speech output tools."""

from typing import Any

from app.schemas.commands import SafetyLevel
from app.tools.registry import registry
from app.voice.tts import TextToSpeechService


@registry.register(
    name="speak",
    description="Speak text out loud through the TTS engine",
    safety_level=SafetyLevel.SAFE,
)
def speak(text: str = "") -> dict[str, Any]:
    # Simulated unless ENABLE_VOICE=true and a TTS backend is installed;
    # the service handles both gracefully.
    return TextToSpeechService().speak(text)
