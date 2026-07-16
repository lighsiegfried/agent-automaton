"""Threshold-selection policy over an evaluation sweep."""

import pytest

from trainer import threshold


def test_default_thresholds_span_the_grid():
    grid = threshold.default_thresholds()
    assert grid[0] == 0.1
    assert grid[-1] == 0.95
    assert all(0.0 < t <= 1.0 for t in grid)


def test_selects_point_meeting_both_targets():
    sweep = [
        {"threshold": 0.3, "recall": 0.95, "false_positives_per_hour": 3.0},
        {"threshold": 0.5, "recall": 0.85, "false_positives_per_hour": 0.4},
        {"threshold": 0.7, "recall": 0.60, "false_positives_per_hour": 0.1},
    ]
    result = threshold.select_threshold_from_sweep(sweep, 0.5, 0.7)
    assert result["threshold"] == 0.5   # only 0.5 meets FP/hr<=0.5 AND recall>=0.7
    assert result["met_targets"] is True


def test_prefers_higher_recall_among_qualifying_points():
    sweep = [
        {"threshold": 0.4, "recall": 0.90, "false_positives_per_hour": 0.3},
        {"threshold": 0.5, "recall": 0.80, "false_positives_per_hour": 0.2},
    ]
    result = threshold.select_threshold_from_sweep(sweep, 0.5, 0.7)
    assert result["threshold"] == 0.4   # both qualify; higher recall wins
    assert result["met_targets"] is True


def test_falls_back_to_fp_budget_when_recall_floor_unreachable():
    sweep = [
        {"threshold": 0.6, "recall": 0.55, "false_positives_per_hour": 0.3},
        {"threshold": 0.8, "recall": 0.40, "false_positives_per_hour": 0.1},
    ]
    result = threshold.select_threshold_from_sweep(sweep, 0.5, 0.7)
    assert result["met_targets"] is False
    assert result["threshold"] == 0.6   # meets FP/hr, best achievable recall
    assert "recall" in result["rationale"]


def test_falls_back_to_lowest_fp_when_budget_unreachable():
    sweep = [
        {"threshold": 0.6, "recall": 0.9, "false_positives_per_hour": 5.0},
        {"threshold": 0.9, "recall": 0.5, "false_positives_per_hour": 2.0},
    ]
    result = threshold.select_threshold_from_sweep(sweep, 0.5, 0.7)
    assert result["met_targets"] is False
    assert result["threshold"] == 0.9   # lowest FP/hr when none meet the budget


def test_empty_sweep_raises():
    with pytest.raises(ValueError, match="empty sweep"):
        threshold.select_threshold_from_sweep([], 0.5, 0.7)
