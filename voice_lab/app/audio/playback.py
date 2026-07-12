"""Local WAV playback for previews — best-effort, never raises.

Playback is a convenience for the human comparing voices; a missing audio
device must never fail a synthesis request.
"""

import sys
from pathlib import Path
from typing import Any


def play_wav(path: Path | str) -> dict[str, Any]:
    """Play a WAV file synchronously on the worker host. Returns a status dict."""
    wav = Path(path)
    if not wav.is_file():
        return {"played": False, "note": "audio file missing"}
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(str(wav), winsound.SND_FILENAME)
            return {"played": True, "backend": "winsound"}
        except Exception as exc:
            return {"played": False, "note": f"playback failed: {type(exc).__name__}"}
    try:
        import sounddevice as sd
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav), dtype="float32")
        sd.play(audio, sample_rate)
        sd.wait()
        return {"played": True, "backend": "sounddevice"}
    except Exception as exc:
        return {"played": False, "note": f"playback failed: {type(exc).__name__}"}
