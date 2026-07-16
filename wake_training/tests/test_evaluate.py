"""Evaluation metric math, performance measurement, and candidate ranking."""

from trainer import evaluate


def _scored():
    # 3 positives, 2 negatives (30 min each -> 1.0 negative hour total).
    return [
        {"label": 1, "score": 0.9, "condition": "quiet", "duration_seconds": 1.0},
        {"label": 1, "score": 0.8, "condition": "noisy", "duration_seconds": 1.0},
        {"label": 1, "score": 0.2, "condition": "far_field", "duration_seconds": 1.0},
        {"label": 0, "score": 0.1, "condition": "quiet", "duration_seconds": 1800.0},
        {"label": 0, "score": 0.6, "condition": "noisy", "duration_seconds": 1800.0},
    ]


def test_score_clips_uses_runner():
    clips = [
        {"path": "a.wav", "label": 1, "condition": "quiet"},
        {"path": "b.wav", "label": 0, "condition": "noisy", "duration_seconds": 5.0},
    ]
    scores = {"a.wav": 0.7, "b.wav": 0.2}
    scored = evaluate.score_clips(lambda p: scores[p], clips)
    assert scored[0] == {"label": 1, "condition": "quiet", "duration_seconds": 0.0, "score": 0.7}
    assert scored[1]["score"] == 0.2


def test_metrics_at_threshold():
    m = evaluate.metrics_at_threshold(_scored(), 0.5)
    assert m["recall"] == round(2 / 3, 4)             # 2 of 3 positives fire
    assert m["false_rejection_rate"] == round(1 - 2 / 3, 4)
    assert m["precision"] == round(2 / 3, 4)          # tp=2, fp=1
    assert m["false_positives"] == 1
    assert m["negative_hours"] == 1.0                 # 2 * 1800s
    assert m["false_positives_per_hour"] == 1.0       # 1 fp / 1.0 h


def test_metrics_perfect_threshold():
    m = evaluate.metrics_at_threshold(_scored(), 0.75)
    assert m["recall"] == round(2 / 3, 4)             # 0.9, 0.8 fire; 0.2 does not
    assert m["false_positives"] == 0                  # no negative >= 0.75
    assert m["precision"] == 1.0


def test_threshold_sweep_length_and_shape():
    sweep = evaluate.threshold_sweep(_scored(), thresholds=[0.3, 0.5, 0.7])
    assert [row["threshold"] for row in sweep] == [0.3, 0.5, 0.7]
    assert all("false_positives_per_hour" in row for row in sweep)


def test_by_condition_splits_metrics():
    bc = evaluate.by_condition(_scored(), 0.5, ["quiet", "noisy", "far_field", "music"])
    assert bc["quiet"]["recall"] == 1.0               # the one quiet positive fires
    assert bc["noisy"]["false_positives"] == 1        # 0.6 negative fires at 0.5
    assert bc["far_field"]["recall"] == 0.0           # the 0.2 positive does not fire
    assert bc["music"]["no_data"] is True             # no music clips supplied


def test_benchmark_inference_with_fake_clock():
    times = iter([0.0, 0.001, 0.005, 0.007])  # start_a, end_a, start_b, end_b
    bench = evaluate.benchmark_inference(
        lambda f: None, frames=["a", "b"], repeats=1, clock=lambda: next(times)
    )
    assert bench["frames"] == 2
    assert bench["latency_ms_mean"] == 1.5            # (1.0 + 2.0) / 2
    assert bench["latency_ms_max"] == 2.0


def test_cpu_usage_with_injected_clocks():
    wall = iter([10.0, 12.0])   # wall0, wall1  -> 2.0 s
    cpu = iter([5.0, 6.0])      # cpu0, cpu1    -> 1.0 s
    usage = evaluate.cpu_usage(
        lambda: None, clock=lambda: next(wall), cpu_clock=lambda: next(cpu)
    )
    assert usage["wall_seconds"] == 2.0
    assert usage["cpu_seconds"] == 1.0
    assert usage["cpu_fraction"] == 0.5


def test_model_size_bytes(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x" * 2048)
    assert evaluate.model_size_bytes(f) == 2048
    assert evaluate.model_size_bytes(tmp_path / "missing.onnx") == 0


def test_evaluate_scored_end_to_end():
    report = evaluate.evaluate_scored(
        _scored(),
        conditions=["quiet", "noisy", "far_field"],
        target_false_positives_per_hour=0.5,
        min_recall=0.5,
        thresholds=[0.3, 0.5, 0.75],
        latency={"latency_ms_mean": 3.0},
        cpu={"cpu_fraction": 0.4},
        model_size=200000,
    )
    assert "selected_threshold" in report
    assert report["n_positive"] == 3
    assert report["n_negative"] == 2
    assert report["model_size_kb"] == round(200000 / 1024, 1)
    assert set(report["by_condition"]) == {"quiet", "noisy", "far_field"}
    # 0.75 achieves 0 FP/hr with recall 0.667 -> meets both targets.
    assert report["threshold_selection"]["met_targets"] is True
    assert report["selected_threshold"] == 0.75


def test_rank_candidates_prefers_targets_then_recall_not_training_accuracy():
    reports = {
        "misses_targets": {
            "threshold_selection": {"met_targets": False},
            "recall": 0.99, "false_positives_per_hour": 0.1,
            "false_rejection_rate": 0.01,
            "latency": {"latency_ms_mean": 4.0}, "model_size_bytes": 180000,
        },
        "meets_lower_recall": {
            "threshold_selection": {"met_targets": True},
            "recall": 0.80, "false_positives_per_hour": 0.3,
            "false_rejection_rate": 0.20,
            "latency": {"latency_ms_mean": 5.0}, "model_size_bytes": 200000,
        },
        "meets_higher_recall": {
            "threshold_selection": {"met_targets": True},
            "recall": 0.90, "false_positives_per_hour": 0.2,
            "false_rejection_rate": 0.10,
            "latency": {"latency_ms_mean": 6.0}, "model_size_bytes": 220000,
        },
    }
    ranked = evaluate.rank_candidates(reports)
    order = [r["candidate"] for r in ranked]
    # Both target-meeting candidates rank above the higher-recall miss; among
    # the meeters, higher recall wins.
    assert order == ["meets_higher_recall", "meets_lower_recall", "misses_targets"]
    assert ranked[0]["rank"] == 1
