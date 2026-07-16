"""Evaluation: recall, false-rejection rate, false-positives/hour, precision, a
threshold sweep, inference latency, CPU use, model size, and a breakdown by
acoustic condition (quiet / noisy / far-field / music / conversation / keyboard).

A "scored clip" is ``{"label": 0|1, "score": float, "condition": str,
"duration_seconds": float}`` — produced by running a model over labelled
validation clips. The scoring runner is injectable so all the metric math is
testable without openWakeWord; the real runner (``onnx_score_runner``) loads the
model lazily.

Model comparison (``rank_candidates``) uses ONLY these evaluation metrics — never
training accuracy — per the phase requirement.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import threshold as threshold_mod


# --- scoring ----------------------------------------------------------------------


def score_clips(runner, clips: list[dict]) -> list[dict]:
    """Score each clip with ``runner(path) -> float``; preserves label/condition."""
    scored = []
    for clip in clips:
        score = float(runner(clip["path"]))
        scored.append({
            "label": int(clip["label"]),
            "condition": clip.get("condition", "unknown"),
            "duration_seconds": float(clip.get("duration_seconds", 0.0)),
            "score": score,
        })
    return scored


# --- core metrics -----------------------------------------------------------------


def _confusion(scored: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in scored:
        fired = row["score"] >= threshold
        if row["label"] == 1:
            tp += fired
            fn += not fired
        else:
            fp += fired
            tn += not fired
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _negative_hours(scored: list[dict]) -> float:
    seconds = sum(r["duration_seconds"] for r in scored if r["label"] == 0)
    return seconds / 3600.0


def metrics_at_threshold(
    scored: list[dict], threshold: float, negative_hours: float | None = None
) -> dict:
    """recall, false-rejection rate, precision, FP count, and FP/hour."""
    c = _confusion(scored, threshold)
    positives = c["tp"] + c["fn"]
    recall = c["tp"] / positives if positives else 0.0
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    hours = _negative_hours(scored) if negative_hours is None else negative_hours
    fp_per_hour = (c["fp"] / hours) if hours > 0 else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "recall": round(recall, 4),
        "false_rejection_rate": round(1.0 - recall, 4),
        "precision": round(precision, 4),
        "false_positives": c["fp"],
        "false_positives_per_hour": round(fp_per_hour, 4),
        "true_positives": c["tp"],
        "false_negatives": c["fn"],
        "negative_hours": round(hours, 4),
    }


def threshold_sweep(
    scored: list[dict],
    thresholds: list[float] | None = None,
    negative_hours: float | None = None,
) -> list[dict]:
    thresholds = thresholds or threshold_mod.default_thresholds()
    return [metrics_at_threshold(scored, t, negative_hours) for t in thresholds]


def by_condition(scored: list[dict], threshold: float, conditions: list[str]) -> dict:
    """Per-condition metrics. Positives in a condition measure recall there;
    negatives in a condition measure that condition's false-positive rate."""
    result = {}
    for condition in conditions:
        subset = [r for r in scored if r["condition"] == condition]
        if subset:
            result[condition] = metrics_at_threshold(subset, threshold)
        else:
            result[condition] = {"threshold": round(threshold, 4), "no_data": True}
    return result


# --- performance ------------------------------------------------------------------


def benchmark_inference(runner, frames: list, repeats: int = 1, clock=time.perf_counter) -> dict:
    """Per-frame inference latency stats from ``runner(frame) -> score``."""
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        for frame in frames:
            start = clock()
            runner(frame)
            samples.append((clock() - start) * 1000.0)
    samples.sort()
    n = len(samples)

    def pct(p: float) -> float:
        if not samples:
            return 0.0
        idx = min(n - 1, int(round(p * (n - 1))))
        return round(samples[idx], 3)

    mean = round(sum(samples) / n, 3) if n else 0.0
    return {
        "frames": len(frames),
        "samples": n,
        "latency_ms_mean": mean,
        "latency_ms_p50": pct(0.50),
        "latency_ms_p95": pct(0.95),
        "latency_ms_max": round(samples[-1], 3) if samples else 0.0,
    }


def cpu_usage(func, *, clock=time.perf_counter, cpu_clock=time.process_time) -> dict:
    """Best-effort CPU utilisation of a callable: process_time / wall_time."""
    wall0, cpu0 = clock(), cpu_clock()
    func()
    wall = max(clock() - wall0, 1e-9)
    cpu = max(cpu_clock() - cpu0, 0.0)
    return {
        "wall_seconds": round(wall, 4),
        "cpu_seconds": round(cpu, 4),
        "cpu_fraction": round(cpu / wall, 3),
    }


def model_size_bytes(path: str | Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.is_file() else 0


# --- top-level report + ranking ---------------------------------------------------


def evaluate_scored(
    scored: list[dict],
    *,
    conditions: list[str],
    target_false_positives_per_hour: float,
    min_recall: float,
    thresholds: list[float] | None = None,
    latency: dict | None = None,
    cpu: dict | None = None,
    model_size: int | None = None,
) -> dict:
    """Assemble the full evaluation report and pick an operating threshold."""
    sweep = threshold_sweep(scored, thresholds)
    selection = threshold_mod.select_threshold_from_sweep(
        sweep, target_false_positives_per_hour, min_recall
    )
    chosen = selection["threshold"]
    headline = metrics_at_threshold(scored, chosen)
    return {
        "selected_threshold": chosen,
        "threshold_selection": selection,
        "headline": headline,
        "recall": headline["recall"],
        "false_rejection_rate": headline["false_rejection_rate"],
        "false_positives_per_hour": headline["false_positives_per_hour"],
        "precision": headline["precision"],
        "sweep": sweep,
        "by_condition": by_condition(scored, chosen, conditions),
        "latency": latency or {},
        "cpu": cpu or {},
        "model_size_bytes": model_size or 0,
        "model_size_kb": round((model_size or 0) / 1024, 1),
        "n_positive": sum(1 for r in scored if r["label"] == 1),
        "n_negative": sum(1 for r in scored if r["label"] == 0),
        "targets": {
            "target_false_positives_per_hour": target_false_positives_per_hour,
            "min_recall": min_recall,
        },
    }


def rank_candidates(reports: dict[str, dict]) -> list[dict]:
    """Rank candidates by EVALUATION metrics only (never training accuracy).

    Ordering key, best first:
      1. meets both targets (bool)
      2. higher recall
      3. lower false-positives/hour
      4. lower latency (mean ms)
      5. smaller model
    """
    ranked = []
    for name, report in reports.items():
        selection = report.get("threshold_selection", {})
        ranked.append({
            "candidate": name,
            "met_targets": bool(selection.get("met_targets")),
            "recall": report.get("recall", 0.0),
            "false_positives_per_hour": report.get("false_positives_per_hour", 0.0),
            "false_rejection_rate": report.get("false_rejection_rate", 1.0),
            "latency_ms_mean": report.get("latency", {}).get("latency_ms_mean", 0.0),
            "model_size_bytes": report.get("model_size_bytes", 0),
            "selected_threshold": report.get("selected_threshold"),
        })
    ranked.sort(
        key=lambda r: (
            not r["met_targets"],                 # meeting targets first
            -r["recall"],                          # then higher recall
            r["false_positives_per_hour"],         # then fewer false alarms
            r["latency_ms_mean"],                  # then faster
            r["model_size_bytes"],                 # then smaller
        )
    )
    for i, row in enumerate(ranked):
        row["rank"] = i + 1
    return ranked


# --- the real ONNX scoring runner (lazy) ------------------------------------------


def onnx_score_runner(model_path: str | Path, sample_rate: int = 16000):
    """Return ``runner(wav_path) -> max wake score`` using openWakeWord (LAZY).

    Used by the CLI for real evaluation; the metric math above never needs it.
    """
    from openwakeword.model import Model  # noqa: F401  (heavy; isolated venv only)
    import numpy as np
    import soundfile as sf

    model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def runner(wav_path: str | Path) -> float:
        model.reset()
        audio, sr = sf.read(str(wav_path), dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        best = 0.0
        step = 1280  # 80 ms frames at 16 kHz
        for start in range(0, len(audio) - step + 1, step):
            preds = model.predict(np.asarray(audio[start : start + step], dtype=np.int16))
            if preds:
                best = max(best, max(preds.values()))
        return best

    return runner
