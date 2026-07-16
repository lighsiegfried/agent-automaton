"""Threshold selection from an evaluation sweep — pure, deterministic policy.

Model selection must never be driven by training accuracy; it is driven by these
operating-point choices over the evaluation sweep. Given a sweep of
``{threshold, recall, false_positives_per_hour}`` rows, pick the operating point
that satisfies the false-alarm budget while keeping recall as high as possible.
"""

from __future__ import annotations


def default_thresholds(start: float = 0.1, stop: float = 0.95, step: float = 0.05) -> list[float]:
    """A standard threshold grid for the sweep (inclusive of stop when it lands)."""
    values: list[float] = []
    x = start
    while x <= stop + 1e-9:
        values.append(round(x, 4))
        x += step
    return values


def select_threshold_from_sweep(
    sweep: list[dict],
    target_false_positives_per_hour: float,
    min_recall: float,
) -> dict:
    """Choose an operating threshold from a sweep.

    Priority:
      1. Meet BOTH the FP/hr budget and the recall floor — among those, take the
         highest recall (tie-break: lower FP/hr, then higher/more-conservative
         threshold).
      2. If none meet both, meet the FP/hr budget and take the highest recall.
      3. If none meet the FP/hr budget, take the lowest FP/hr (then highest recall).

    Returns ``{threshold, met_targets, rationale, metrics}``.
    """
    if not sweep:
        raise ValueError("cannot select a threshold from an empty sweep")

    def fp(row: dict) -> float:
        return float(row.get("false_positives_per_hour", float("inf")))

    def rc(row: dict) -> float:
        return float(row.get("recall", 0.0))

    both = [
        r for r in sweep
        if fp(r) <= target_false_positives_per_hour and rc(r) >= min_recall
    ]
    if both:
        best = max(both, key=lambda r: (rc(r), -fp(r), r["threshold"]))
        return _result(best, True, "meets FP/hr budget and recall floor")

    fp_ok = [r for r in sweep if fp(r) <= target_false_positives_per_hour]
    if fp_ok:
        best = max(fp_ok, key=lambda r: (rc(r), r["threshold"]))
        return _result(
            best, False,
            f"meets FP/hr budget but recall {rc(best):.3f} < floor {min_recall:.3f}",
        )

    best = min(sweep, key=lambda r: (fp(r), -rc(r)))
    return _result(
        best, False,
        f"no threshold meets the FP/hr budget ({target_false_positives_per_hour:g}/h); "
        f"chose the lowest achievable ({fp(best):g}/h)",
    )


def _result(row: dict, met: bool, rationale: str) -> dict:
    return {
        "threshold": round(float(row["threshold"]), 4),
        "met_targets": met,
        "rationale": rationale,
        "metrics": row,
    }
