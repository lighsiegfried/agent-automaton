"""Wake-word calibration core: observations, labels, threshold recommendation,
and the guarantee that calibration NEVER executes a command.
"""

from app.voice import wake_calibration as wc
from app.voice.wake_calibration import CalibrationSession, CalibrationStore


# --- store -------------------------------------------------------------------------


def test_store_append_load_clear(tmp_path):
    store = CalibrationStore(tmp_path / "cal.jsonl")
    assert store.load() == []
    store.append({"score": 0.7, "label": "detection"})
    store.append({"score": 0.2, "label": "false_positive"})
    loaded = store.load()
    assert [o["label"] for o in loaded] == ["detection", "false_positive"]
    store.clear()
    assert store.load() == []


def test_store_skips_corrupt_lines(tmp_path):
    path = tmp_path / "cal.jsonl"
    path.write_text('{"score": 0.5, "label": "detection"}\nnot json\n', encoding="utf-8")
    assert len(CalibrationStore(path).load()) == 1


# --- session -----------------------------------------------------------------------


def test_observe_records_detection_above_threshold_with_speech():
    session = CalibrationSession(threshold=0.5)
    assert session.observe(0.6, is_speech=True) is True
    assert session.observe(0.4, is_speech=True) is False   # below threshold
    assert session.detections == 1


def test_vad_gate_blocks_non_speech_detection():
    session = CalibrationSession(threshold=0.5)
    assert session.observe(0.9, is_speech=False) is False   # high score, but not speech
    assert session.detections == 0


def test_mark_correct_and_false_positive_annotate_last_detection():
    session = CalibrationSession(threshold=0.5)
    session.observe(0.8)                       # a detection
    correct = session.mark_correct()
    assert correct["label"] == "correct" and correct["score"] == 0.8
    session.observe(0.7)                        # another detection
    fp = session.mark_false_positive()
    assert fp["label"] == "false_positive" and fp["score"] == 0.7
    assert session.labels["correct"] == 1 and session.labels["false_positive"] == 1


def test_mark_missed_uses_peak_score():
    session = CalibrationSession(threshold=0.6)
    session.observe(0.3)
    session.observe(0.55)       # peak, but below threshold -> no detection
    session.observe(0.2)
    missed = session.mark_missed()
    assert missed["label"] == "missed"
    assert missed["score"] == 0.55       # the peak the user's word reached
    assert session.peak_score == 0.0     # reset after marking


def test_mark_without_detection_is_noop():
    session = CalibrationSession(threshold=0.5)
    assert session.mark_correct() is None
    assert session.mark_false_positive() is None


def test_session_persists_to_store(tmp_path):
    store = CalibrationStore(tmp_path / "cal.jsonl")
    session = CalibrationSession(threshold=0.5, store=store)
    session.observe(0.9)
    session.mark_correct()
    assert [o["label"] for o in store.load()] == ["detection", "correct"]


def test_session_has_no_command_execution_path():
    """A calibration session cannot submit or execute anything — by construction."""
    session = CalibrationSession(threshold=0.5)
    for attr in ("submit", "client", "handle_utterance", "execute", "command"):
        assert not hasattr(session, attr)
    # Feeding a whole "utterance" of strong detections only ever records data.
    for _ in range(50):
        session.observe(0.95, is_speech=True)
    assert session.detections == 50
    assert all(o["label"] == "detection" for o in session.observations)


# --- recommendation ----------------------------------------------------------------


def test_recommend_separable_picks_midpoint():
    obs = [
        {"score": 0.85, "label": "correct"},
        {"score": 0.80, "label": "correct"},
        {"score": 0.20, "label": "false_positive"},
    ]
    rec = wc.recommend_threshold(obs, current_threshold=0.5)
    assert rec["separable"] is True
    assert 0.2 < rec["recommended"] < 0.8


def test_recommend_only_true_positives_sets_below_min():
    rec = wc.recommend_threshold(
        [{"score": 0.7, "label": "correct"}, {"score": 0.6, "label": "missed"}],
        current_threshold=0.5,
    )
    assert rec["recommended"] < 0.6          # below the weakest true positive
    assert rec["n_true"] == 2 and rec["n_false"] == 0


def test_recommend_only_false_positives_sets_above_max():
    rec = wc.recommend_threshold(
        [{"score": 0.4, "label": "false_positive"}], current_threshold=0.3
    )
    assert rec["recommended"] > 0.4          # above the worst false positive


def test_recommend_overlap_flags_not_separable():
    obs = [
        {"score": 0.5, "label": "correct"},
        {"score": 0.6, "label": "false_positive"},   # false scores higher than a true
    ]
    rec = wc.recommend_threshold(obs, current_threshold=0.5)
    assert rec["separable"] is False
    assert "OVERLAP" in rec["rationale"]


def test_recommend_no_data_keeps_current():
    rec = wc.recommend_threshold([], current_threshold=0.55)
    assert rec["recommended"] == 0.55
    assert rec["separable"] is None


def test_summarize_store_end_to_end(tmp_path):
    store = CalibrationStore(tmp_path / "cal.jsonl")
    for obs in [
        {"score": 0.8, "label": "detection"},
        {"score": 0.8, "label": "correct"},
        {"score": 0.2, "label": "false_positive"},
    ]:
        store.append(obs)
    summary = wc.summarize_store(tmp_path / "cal.jsonl", current_threshold=0.5)
    assert summary["total_observations"] == 3
    assert summary["label_counts"]["correct"] == 1
    assert "recommended" in summary["recommendation"]
