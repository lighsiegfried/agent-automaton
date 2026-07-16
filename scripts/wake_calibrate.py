"""Local wake-word calibration listener (Phase 3D.2) — Windows host only.

Listens with the INSTALLED custom model, shows live wake scores, and lets you
label what you hear so it can recommend a threshold. It DOES NOT execute
commands: unlike scripts/fifi_wake.py it constructs no /voice/command client and
has no submit path at all — feeding audio only ever records score/timestamp/label
observations (never idle audio).

    python scripts/local_runtime.py wake-calibrate           # recommended entry
    python scripts/wake_calibrate.py [--threshold X] [--fresh] [--store PATH]

Keys while running:  [c] last detection was me   [f] last detection was a false
alarm   [m] I said it but nothing fired   Ctrl+C to stop and see the recommendation.

Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fifi_wake  # noqa: E402  (reused WakeConfig + mic loop; NOT its command client)
from app.voice import wake_calibration, wake_metadata  # noqa: E402


class CalibrationSink:
    """Feed target for the mic loop: scores frames and records observations.

    This is the ONLY thing the microphone feeds during calibration. It has no
    command client and no code path that can submit an utterance — it can only
    call ``session.observe`` and print the live score.
    """

    def __init__(self, service, session, out=print, clock=time.monotonic):
        self.service = service
        self.session = session
        self.out = out
        self.clock = clock
        self._last_print = 0.0

    def feed(self, frame) -> None:
        score = self.service.score(frame)
        is_speech = self.service.is_speech(frame)
        detected = self.session.observe(score, is_speech)
        now = self.clock()
        if now - self._last_print >= 1.0:
            self.out(
                f"score {score:.3f} (thr {self.session.threshold:g}) "
                f"speech={is_speech} detections={self.session.detections}"
            )
            self._last_print = now
        if detected:
            self.out(f"DETECTION score={score:.3f} — [c]orrect / [f]alse alarm?")

    def cancel(self) -> None:  # mic-loop hook on device error; nothing to undo
        pass


def _installed_threshold(default: float) -> float:
    doc = wake_metadata.read_metadata(PROJECT_ROOT / "models" / "wake_words" / "fifi.metadata.json")
    if doc and isinstance(doc.get("threshold"), (int, float)):
        return float(doc["threshold"])
    return default


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fifi wake calibration (no command execution).")
    parser.add_argument("--threshold", type=float, help="threshold to test (default: installed)")
    parser.add_argument("--store", help="calibration JSONL path")
    parser.add_argument("--fresh", action="store_true", help="clear previous observations first")
    args = parser.parse_args(argv)

    deps_ok, missing = fifi_wake.check_wake_deps()
    if not deps_ok:
        print("Missing wake dependencies: " + ", ".join(missing))
        print("  .venv\\Scripts\\python.exe -m pip install -r requirements-wakeword.txt")
        return 1
    if not fifi_wake.wake_model_present():
        print(f"No installed wake model at {fifi_wake.wake_model_path()} — install one first "
              "(wake_training/scripts/wake_trainer.py install <candidate>).")
        return 1

    from app.voice.wake_word import WakeWordService

    config = fifi_wake.WakeConfig.load()
    threshold = args.threshold if args.threshold is not None else _installed_threshold(config.threshold)
    service = WakeWordService(
        model_path=config.model_path, threshold=threshold, vad_threshold=config.vad_threshold
    )
    loaded = service.load()
    if loaded.get("status") != "ok":
        print(f"Wake model unavailable: {loaded.get('message')}")
        return 1

    store = wake_calibration.CalibrationStore(args.store or wake_calibration.DEFAULT_STORE_PATH)
    if args.fresh:
        store.clear()
    session = wake_calibration.CalibrationSession(threshold=threshold, store=store)
    sink = CalibrationSink(service, session)

    def _mark(label_fn, name):
        def handler():
            obs = label_fn()
            print(f"marked {name}" if obs else f"(no recent detection to mark {name})")
        return handler

    import keyboard

    keyboard.add_hotkey("c", _mark(session.mark_correct, "correct"))
    keyboard.add_hotkey("f", _mark(session.mark_false_positive, "false_positive"))
    keyboard.add_hotkey("m", _mark(session.mark_missed, "missed"))

    print(f"Calibrating with threshold {threshold:g} — say 'Fifi' from different distances.")
    print("Keys: [c] correct  [f] false alarm  [m] I said it but nothing fired  Ctrl+C to finish.")
    print("This mode NEVER executes a command; it only records scores + your labels.\n")
    try:
        fifi_wake._audio_loop(sink, config, stop_check=lambda: False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass

    recommendation = wake_calibration.recommend_threshold(session.observations, threshold)
    print("\n=== Calibration summary ===")
    print(f"Detections      : {session.detections}")
    print(f"Labels          : {session.labels}")
    print(f"Current thr     : {recommendation['current']}")
    print(f"Recommended thr : {recommendation['recommended']}  ({recommendation['rationale']})")
    print(f"Observations at : {store.path}")
    print("Apply it by setting WAKE_WORD_THRESHOLD in .env (and re-run wake-doctor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
