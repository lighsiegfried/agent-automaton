"""Windows SAPI engine (pyttsx3) — the dependable fallback voice.

No GPU, no downloads: whatever voices Windows ships. Also the default
`fallback_engine` for every neural profile, so a broken neural stack always
degrades to a working voice instead of silence.

Host-only by design: SAPI is a Windows facility. Inside the Linux Voice Lab
container it is DISABLED (non-Windows host, or VOICE_LAB_DISABLE_SAPI set), so
the container's fallback chain terminates at Kokoro instead of a broken/absent
SAPI. The Windows fallback the main API relies on lives on the HOST, unaffected.
"""

import os
import sys
from pathlib import Path
from typing import Any

from app.engines.base import EngineUnavailable, TtsEngine
from app.profiles.schema import VoiceProfile

INSTALL_HINT = (
    "pyttsx3 is not installed in the voice_lab environment: "
    "run python voice_lab/scripts/setup.py"
)

# SAPI rate offset per 1.0x of profile speed; pyttsx3 default rate is ~200 wpm.
_BASE_WPM = 180


class WindowsSapiEngine(TtsEngine):
    name = "windows_sapi"

    def __init__(self) -> None:
        super().__init__()
        self._pyttsx3 = None

    def available(self) -> tuple[bool, str]:
        # Disabled off Windows (e.g. the Linux container) or when explicitly
        # turned off — SAPI is a host-only facility. The chain falls to Kokoro.
        disabled = os.environ.get("VOICE_LAB_DISABLE_SAPI", "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if disabled or sys.platform != "win32":
            return False, "windows_sapi is host-only (disabled on this worker)"
        try:
            import pyttsx3  # noqa: F401
        except Exception:
            return False, INSTALL_HINT
        return True, ""

    def _load(self, profile: VoiceProfile) -> None:
        try:
            import pyttsx3
        except Exception as exc:
            raise EngineUnavailable(INSTALL_HINT) from exc
        self._pyttsx3 = pyttsx3

    def _synthesize(self, text: str, profile: VoiceProfile, out_path: Path) -> dict[str, Any]:
        # pyttsx3 engines are not reusable across COM apartments — init per call.
        driver = self._pyttsx3.init()
        try:
            driver.setProperty("rate", int(_BASE_WPM * profile.speed))
            if profile.speaker:
                for voice in driver.getProperty("voices") or []:
                    if profile.speaker.lower() in (voice.name or "").lower():
                        driver.setProperty("voice", voice.id)
                        break
            driver.save_to_file(text, str(out_path))
            driver.runAndWait()
        finally:
            try:
                driver.stop()
            except Exception:
                pass
        return {"sample_rate": profile.sample_rate, "device": "cpu"}
