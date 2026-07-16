"""Runtime wiring for wake calibration (Phase 3D.2): the wake-calibrate command
dispatches to the listener, the metadata path is derived correctly, and the
calibration sink has NO command-execution path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_runtime  # noqa: E402
import wake_calibrate  # noqa: E402
from app.voice import wake_calibration  # noqa: E402


class FakeService:
    def __init__(self, score, speech):
        self._score, self._speech = score, speech

    def score(self, frame):
        return self._score

    def is_speech(self, frame):
        return self._speech


# --- command dispatch --------------------------------------------------------------


def test_wake_calibrate_subcommand_dispatches(monkeypatch):
    captured = {}

    def fake_run(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(local_runtime.wake_calibrate, "run", fake_run)
    args = local_runtime.build_parser().parse_args(
        ["wake-calibrate", "--threshold", "0.7", "--fresh"]
    )
    assert args.func is local_runtime.cmd_wake_calibrate
    assert args.func(args) == 0
    assert captured["argv"] == ["--threshold", "0.7", "--fresh"]


def test_wake_calibrate_no_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(local_runtime.wake_calibrate, "run",
                        lambda argv: captured.setdefault("argv", argv) or 0)
    args = local_runtime.build_parser().parse_args(["wake-calibrate"])
    assert args.func(args) == 0
    assert captured["argv"] == []


def test_wake_metadata_path_derivation(monkeypatch):
    monkeypatch.setattr(
        local_runtime.fifi_wake, "wake_model_path",
        lambda: Path("models/wake_words/fifi.onnx"),
    )
    assert local_runtime.wake_metadata_path().name == "fifi.metadata.json"


# --- the sink cannot execute commands ----------------------------------------------


def test_calibration_sink_records_detection_and_has_no_command_path():
    session = wake_calibration.CalibrationSession(threshold=0.5)
    sink = wake_calibrate.CalibrationSink(
        FakeService(0.9, True), session, out=lambda *a, **k: None, clock=lambda: 0.0
    )
    sink.feed(b"frame")
    assert session.detections == 1
    # No submit/client/command anywhere on the sink — calibration cannot act.
    for attr in ("submit", "client", "handle_utterance", "execute", "command"):
        assert not hasattr(sink, attr)


def test_calibration_sink_respects_vad_gate():
    session = wake_calibration.CalibrationSession(threshold=0.5)
    sink = wake_calibrate.CalibrationSink(
        FakeService(0.95, False), session, out=lambda *a, **k: None, clock=lambda: 0.0
    )
    sink.feed(b"frame")   # high score but not speech
    assert session.detections == 0
