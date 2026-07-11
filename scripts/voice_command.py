"""Record a short clip (or take a .wav file) and send it to /voice/command.

Usage (from the project root, with the server running):
  python scripts/voice_command.py                    # record 5s from the mic
  python scripts/voice_command.py --seconds 8
  python scripts/voice_command.py --file clip.wav    # send an existing WAV
  python scripts/voice_command.py --confirm          # approve a sensitive action

Recording requires the optional voice dependencies:
  pip install -r requirements-voice.txt
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import httpx

DEFAULT_SERVER = "http://127.0.0.1:8000"


def record_clip(seconds: float, sample_rate: int) -> Path:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit(
            "Recording requires the optional voice dependencies:\n"
            "  pip install -r requirements-voice.txt"
        )

    print(f"Recording {seconds:g}s at {sample_rate} Hz - speak now...")
    frames = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Done recording.")

    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(name, frames, sample_rate)
    return Path(name)


def agent_name(server: str) -> str:
    """The assistant's name from /identity; falls back to 'Fifi' quietly."""
    try:
        response = httpx.get(f"{server}/identity", timeout=5.0)
        response.raise_for_status()
        return response.json().get("agent_name") or "Fifi"
    except (httpx.HTTPError, ValueError):
        return "Fifi"


def send(server: str, wav_path: Path, confirm: bool) -> dict:
    try:
        with open(wav_path, "rb") as audio:
            response = httpx.post(
                f"{server}/voice/command",
                params={"confirm": confirm},
                files={"file": (wav_path.name, audio, "audio/wav")},
                timeout=120.0,  # first transcription may load the whisper model
            )
    except httpx.ConnectError:
        sys.exit(
            f"Server not reachable at {server}.\n"
            "Start it first:  .venv\\Scripts\\python.exe -m uvicorn app.main:app"
        )
    if response.status_code != 200:
        sys.exit(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a voice command to agent-automaton.")
    parser.add_argument("--seconds", type=float, default=5.0, help="recording length (default 5)")
    parser.add_argument("--rate", type=int, default=16000, help="sample rate (default 16000)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"API base URL (default {DEFAULT_SERVER})")
    parser.add_argument("--file", help="send this .wav instead of recording")
    parser.add_argument("--confirm", action="store_true", help="approve a sensitive action")
    args = parser.parse_args()

    recorded = None
    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            sys.exit(f"File not found: {wav_path}")
    else:
        recorded = record_clip(args.seconds, args.rate)
        wav_path = recorded

    try:
        data = send(args.server, wav_path, args.confirm)
    finally:
        if recorded is not None:
            recorded.unlink(missing_ok=True)

    if data.get("status") != "ok":
        sys.exit(f"[{data.get('status')}] {data.get('message')}")

    name = agent_name(args.server)
    command = data["command"]
    print()
    print(f"{name} heard    : {data['transcription']!r}  (language: {data.get('language')})")
    print(f"{name} response : {data.get('assistant_message') or command['message']}")
    print(f"Planner : {command['planner']}")
    print(f"Intent  : {command['intent']}")
    print(f"Safety  : {command.get('safety_level')} -> {command['status']}")
    if command.get("result"):
        print(f"Result  : {command['result']}")
    if data.get("speech"):
        print(f"Speech  : {data['speech']}")
    if command["status"] == "needs_confirmation":
        print("\nRe-run with --confirm to approve this action.")


if __name__ == "__main__":
    main()
