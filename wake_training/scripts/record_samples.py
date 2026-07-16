"""Record OPTIONAL authorized "Fifi" samples for training/verification.

Training works WITHOUT any personal recordings (synthetic piper voices cover the
positives). These recordings only improve robustness and enable the optional
speaker verifier. They stay local and gitignored (datasets/positive/user/).

    wake_training/.venv/Scripts/python.exe wake_training/scripts/record_samples.py
        [--count 12] [--seconds 2.5] [--phrase "Fifi"]

The script walks you through several DISTANCES, VOLUMES and ROOM conditions so
the samples reflect real use. Only your spoken wake word is recorded, on an
explicit key press — there is no always-on capture.

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import sys
from pathlib import Path

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(WAKE_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(WAKE_TRAINING_ROOT))

from trainer import paths  # noqa: E402

SAMPLE_RATE = 16000


def sample_plan(count: int) -> list[dict]:
    """A varied capture plan: distances x volumes x rooms, cycled to ``count``."""
    conditions = [
        {"distance": "near (0.5 m)", "volume": "normal", "room": "quiet"},
        {"distance": "near (0.5 m)", "volume": "soft", "room": "quiet"},
        {"distance": "mid (1.5 m)", "volume": "normal", "room": "quiet"},
        {"distance": "mid (1.5 m)", "volume": "loud", "room": "some noise"},
        {"distance": "far (3 m)", "volume": "normal", "room": "some noise"},
        {"distance": "far (3 m)", "volume": "loud", "room": "reverberant"},
        {"distance": "mid (1.5 m)", "volume": "normal", "room": "music playing"},
        {"distance": "near (0.5 m)", "volume": "fast", "room": "quiet"},
    ]
    return [dict(conditions[i % len(conditions)], index=i + 1) for i in range(count)]


def next_index(directory: Path) -> int:
    """The next 1-based sample index given existing recordings."""
    existing = list(directory.glob("fifi_*.wav")) if directory.is_dir() else []
    return len(existing) + 1


def recording_path(index: int, root: Path | None = None) -> Path:
    directory = (root or paths.USER_RECORDINGS_DIR)
    return directory / f"fifi_{index:03d}.wav"


def _record_one(seconds: float) -> "object":
    import sounddevice as sd

    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record authorized Fifi samples (optional).")
    parser.add_argument("--count", type=int, default=12, help="how many samples to record")
    parser.add_argument("--seconds", type=float, default=2.5, help="seconds per sample")
    parser.add_argument("--phrase", default="Fifi", help="phrase to say each time")
    args = parser.parse_args(argv)

    try:
        import soundfile as sf  # noqa: F401
        import sounddevice  # noqa: F401
    except Exception:
        print("Recording needs sounddevice + soundfile: install requirements.txt first.")
        return 1

    directory = paths.USER_RECORDINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    start = next_index(directory)
    plan = sample_plan(args.count)
    print("=== Fifi authorized-sample recorder (optional, local only) ===")
    print(f"Saving to       : {directory}  (gitignored)")
    print(f"Say             : {args.phrase!r} clearly at the prompt.\n")

    import soundfile as sf

    recorded = 0
    for step in plan:
        idx = start + step["index"] - 1
        print(f"[{step['index']}/{args.count}] distance={step['distance']} "
              f"volume={step['volume']} room={step['room']}")
        try:
            input(f"    Press Enter, then say {args.phrase!r} ({args.seconds:g}s)... ")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped early — recordings so far are kept.")
            break
        audio = _record_one(args.seconds)
        out = recording_path(idx, directory)
        sf.write(str(out), audio, SAMPLE_RATE)
        recorded += 1
        print(f"    saved {out.name}")
    print(f"\nRecorded {recorded} sample(s). Training still works without these; "
          "they improve robustness and enable the optional verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
