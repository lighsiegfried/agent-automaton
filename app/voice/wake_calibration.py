"""Local wake-word calibration — observe scores, collect labels, recommend a
threshold. NEVER executes commands and NEVER stores idle audio.

This is the shared core for both ``wake_training/scripts/wake_trainer.py
calibrate`` and ``scripts/local_runtime.py wake-calibrate``. A ``CalibrationSession``
is fed wake scores (from the same VAD-gated model the runtime uses) and records
ONLY ``{utc, score, label}`` observations — never frames, never audio. It has no
command client and no code path that could submit to /voice/command, so a
calibration run physically cannot trigger an action.

Pure stdlib (no pydantic, no torch) so it imports cleanly in the main app, the
isolated trainer venv, and the test suite.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE_PATH = PROJECT_ROOT / "storage" / "runtime" / "wake_calibration.jsonl"

# Observation labels.
DETECTION = "detection"          # score crossed the threshold (automatic)
CORRECT = "correct"              # user: that detection was me saying the wake word
MISSED = "missed"               # user: I said it but nothing fired (peak score)
FALSE_POSITIVE = "false_positive"  # user: that detection was NOT me / not the word

# Recommendation guard rails.
THRESHOLD_FLOOR = 0.1
THRESHOLD_CEIL = 0.95


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- persistence (append-only JSONL of score/label observations) ------------------


class CalibrationStore:
    """Append-only JSONL store of observations. Contains no audio, ever."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH):
        self.path = Path(path)

    def append(self, observation: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def clear(self) -> None:
        """Atomically truncate the store (start a fresh calibration)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        os.close(fd)
        os.replace(tmp, self.path)


# --- the no-command calibration session -------------------------------------------


@dataclass
class CalibrationSession:
    """Fed wake scores; records observations and user labels. No command path.

    ``observe(score)`` is called once per scored frame. When a score crosses the
    threshold it records a ``detection`` and remembers it so the user can label
    the most recent detection ``correct`` or ``false_positive``. ``mark_missed``
    records the peak score seen since the last detection (a wake word that did
    not fire). There is deliberately no ``submit``/``execute`` here.
    """

    threshold: float
    store: CalibrationStore | None = None
    observations: list[dict] = field(default_factory=list)
    last_score: float = 0.0
    peak_score: float = 0.0
    detections: int = 0
    labels: dict = field(default_factory=lambda: {CORRECT: 0, MISSED: 0, FALSE_POSITIVE: 0})
    _last_detection: dict | None = None

    def _record(self, label: str, score: float) -> dict:
        obs = {"utc": _utc(), "score": round(float(score), 4), "label": label,
               "threshold": round(float(self.threshold), 4)}
        self.observations.append(obs)
        if self.store is not None:
            self.store.append(obs)
        return obs

    def observe(self, score: float, is_speech: bool = True) -> bool:
        """Record one frame's wake score. Returns True if it was a detection.

        A detection requires the score to cross the threshold AND the frame to be
        speech — the same VAD gate the runtime applies — so calibration reflects
        real triggering, not TV/noise spikes.
        """
        self.last_score = float(score)
        self.peak_score = max(self.peak_score, float(score))
        if score >= self.threshold and is_speech:
            obs = self._record(DETECTION, score)
            self.detections += 1
            self._last_detection = obs
            self.peak_score = 0.0  # start a fresh peak window after a detection
            return True
        return False

    def mark_correct(self) -> dict | None:
        """Label the most recent detection a true positive."""
        if self._last_detection is None:
            return None
        self.labels[CORRECT] += 1
        return self._record(CORRECT, self._last_detection["score"])

    def mark_false_positive(self) -> dict | None:
        """Label the most recent detection a false alarm."""
        if self._last_detection is None:
            return None
        self.labels[FALSE_POSITIVE] += 1
        return self._record(FALSE_POSITIVE, self._last_detection["score"])

    def mark_missed(self) -> dict:
        """Record that a spoken wake word did NOT fire (uses the peak score)."""
        self.labels[MISSED] += 1
        obs = self._record(MISSED, self.peak_score)
        self.peak_score = 0.0
        return obs

    def summary(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "detections": self.detections,
            "labels": dict(self.labels),
            "observations": len(self.observations),
            "last_score": round(self.last_score, 4),
        }


# --- threshold recommendation -----------------------------------------------------


def recommend_threshold(
    observations: list[dict],
    current_threshold: float,
    margin: float = 0.05,
) -> dict:
    """Recommend a threshold from labelled observations.

    True positives (``correct`` + ``missed``) are scores that SHOULD fire; false
    positives are scores that should NOT. The ideal threshold sits just above the
    worst false positive and at/below the weakest true positive. When those
    overlap, we take the midpoint and flag it.
    """
    true_scores = [
        o["score"] for o in observations if o.get("label") in (CORRECT, MISSED)
    ]
    false_scores = [
        o["score"] for o in observations if o.get("label") == FALSE_POSITIVE
    ]

    def clamp(v: float) -> float:
        return round(min(THRESHOLD_CEIL, max(THRESHOLD_FLOOR, v)), 3)

    if not true_scores and not false_scores:
        return {
            "recommended": round(current_threshold, 3),
            "current": round(current_threshold, 3),
            "separable": None,
            "rationale": "no labelled observations yet — keeping the current threshold",
            "n_true": 0,
            "n_false": 0,
        }

    true_min = min(true_scores) if true_scores else None
    false_max = max(false_scores) if false_scores else None

    if false_max is None:                     # only true positives seen
        recommended = clamp(true_min - margin)
        rationale = "no false positives observed — set just below the weakest true positive"
        separable = True
    elif true_min is None:                    # only false positives seen
        recommended = clamp(false_max + margin)
        rationale = "no true positives observed — set just above the worst false positive"
        separable = True
    elif false_max + 1e-9 < true_min:         # cleanly separable
        recommended = clamp(false_max + (true_min - false_max) / 2.0)
        rationale = "true and false positives are separable — midpoint of the gap"
        separable = True
    else:                                     # overlap — compromise
        recommended = clamp((true_min + false_max) / 2.0)
        rationale = (
            "true and false positives OVERLAP — chose a compromise; consider "
            "retraining or collecting more samples"
        )
        separable = False

    return {
        "recommended": recommended,
        "current": round(current_threshold, 3),
        "separable": separable,
        "rationale": rationale,
        "n_true": len(true_scores),
        "n_false": len(false_scores),
        "true_score_min": true_min,
        "false_score_max": false_max,
    }


def summarize_store(path: str | Path, current_threshold: float) -> dict:
    """Load a calibration store and produce a recommendation summary."""
    store = CalibrationStore(path)
    observations = store.load()
    counts: dict[str, int] = {}
    for obs in observations:
        counts[obs.get("label", "?")] = counts.get(obs.get("label", "?"), 0) + 1
    return {
        "path": str(store.path),
        "total_observations": len(observations),
        "label_counts": counts,
        "recommendation": recommend_threshold(observations, current_threshold),
    }
