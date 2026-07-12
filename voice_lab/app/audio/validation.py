"""WAV validation — stdlib-only sanity checks for synthesized audio.

Catches the classic neural-TTS failure modes (empty file, zero frames, pure
silence) so a broken engine is reported instead of "spoken" silently.
"""

import wave
from pathlib import Path
from typing import Any


def validate_wav(path: Path | str, min_duration_seconds: float = 0.05) -> dict[str, Any]:
    wav_path = Path(path)
    if not wav_path.is_file():
        return {"valid": False, "reason": "file missing"}
    try:
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            data = wav.readframes(frames)
    except (wave.Error, OSError, EOFError) as exc:
        return {"valid": False, "reason": f"unreadable WAV: {type(exc).__name__}"}

    duration = frames / rate if rate else 0.0
    result = {
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_seconds": round(duration, 2),
    }
    if duration < min_duration_seconds:
        return {**result, "valid": False, "reason": "audio too short"}
    if not any(data):  # every byte zero => digital silence
        return {**result, "valid": False, "reason": "audio is pure silence"}
    return {**result, "valid": True, "reason": ""}
