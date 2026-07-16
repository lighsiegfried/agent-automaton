"""Trainer CLI: parsing, dispatch, and command wiring (no heavy training)."""

import json
import sys
from pathlib import Path

import pytest

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKE_TRAINING_ROOT / "scripts"))

import wake_trainer  # noqa: E402
from trainer import paths  # noqa: E402


@pytest.fixture
def reports_under_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(paths, "INSTALL_TARGET", tmp_path / "models" / "fifi.onnx")
    monkeypatch.setattr(paths, "METADATA_TARGET", tmp_path / "models" / "fifi.metadata.json")
    (tmp_path / "reports").mkdir()
    return tmp_path


def _eval_report(met=True, recall=0.9, fp=0.2, latency=5.0, size=200000):
    return {
        "selected_threshold": 0.5,
        "threshold_selection": {"met_targets": met},
        "recall": recall, "false_rejection_rate": round(1 - recall, 4),
        "false_positives_per_hour": fp, "precision": 0.95,
        "latency": {"latency_ms_mean": latency}, "model_size_bytes": size,
    }


# --- parsing / dispatch ------------------------------------------------------------


def test_parser_dispatches_commands():
    parser = wake_trainer.build_parser()
    assert parser.parse_args(["status"]).func is wake_trainer.cmd_status
    args = parser.parse_args(["train", "fifi", "--dry-run"])
    assert args.func is wake_trainer.cmd_train
    assert args.candidate == "fifi" and args.dry_run is True
    assert parser.parse_args(["install", "oye_fifi", "--force"]).force is True


def test_no_command_prints_help_and_returns_1(capsys):
    assert wake_trainer.main([]) == 1


def test_discover_candidates_finds_both():
    assert set(wake_trainer.discover_candidates()) >= {"fifi", "oye_fifi"}


# --- status ------------------------------------------------------------------------


def test_status_lists_candidates(reports_under_tmp, capsys):
    assert wake_trainer.cmd_status(wake_trainer.build_parser().parse_args(["status"])) == 0
    out = capsys.readouterr().out
    assert "fifi" in out and "oye_fifi" in out
    assert "Installed model" in out


# --- train --dry-run ---------------------------------------------------------------


def test_train_dry_run_prepares_without_launch(reports_under_tmp, capsys):
    args = wake_trainer.build_parser().parse_args(["train", "fifi", "--dry-run"])
    assert wake_trainer.cmd_train(args) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    # The generated openWakeWord config was written under the redirected outputs.
    assert (reports_under_tmp / "outputs" / "fifi" / "fifi.training.yaml").is_file()


def test_train_rejects_unknown_candidate(reports_under_tmp, capsys):
    args = wake_trainer.build_parser().parse_args(["train", "nope"])
    assert wake_trainer.cmd_train(args) == 1


# --- evaluate (pre-scored manifest, no model needed) -------------------------------


def test_evaluate_with_prescored_manifest(reports_under_tmp, capsys):
    manifest = reports_under_tmp / "clips.json"
    manifest.write_text(json.dumps([
        {"path": "p1.wav", "label": 1, "condition": "quiet", "score": 0.9},
        {"path": "p2.wav", "label": 1, "condition": "noisy", "score": 0.8},
        {"path": "n1.wav", "label": 0, "condition": "quiet", "score": 0.1,
         "duration_seconds": 3600},
    ]), encoding="utf-8")
    args = wake_trainer.build_parser().parse_args(
        ["evaluate", "fifi", "--manifest", str(manifest)]
    )
    assert wake_trainer.cmd_evaluate(args) == 0
    out = capsys.readouterr().out
    assert "Recall" in out and "By condition" in out
    assert (reports_under_tmp / "reports" / "fifi-eval.json").is_file()


def test_evaluate_without_manifest_errors(reports_under_tmp, capsys):
    args = wake_trainer.build_parser().parse_args(["evaluate", "fifi"])
    assert wake_trainer.cmd_evaluate(args) == 1


# --- compare -----------------------------------------------------------------------


def test_compare_ranks_from_reports(reports_under_tmp, capsys):
    (reports_under_tmp / "reports" / "fifi-eval.json").write_text(
        json.dumps(_eval_report(met=True, recall=0.85)), encoding="utf-8"
    )
    (reports_under_tmp / "reports" / "oye_fifi-eval.json").write_text(
        json.dumps(_eval_report(met=True, recall=0.92)), encoding="utf-8"
    )
    assert wake_trainer.cmd_compare(wake_trainer.build_parser().parse_args(["compare"])) == 0
    out = capsys.readouterr().out
    assert "Recommended" in out
    assert "oye_fifi" in out  # higher recall wins


def test_compare_without_reports_errors(reports_under_tmp, capsys):
    assert wake_trainer.cmd_compare(wake_trainer.build_parser().parse_args(["compare"])) == 1


# --- install (install_model stubbed) -----------------------------------------------


def test_install_refuses_without_eval_or_threshold(reports_under_tmp, capsys):
    model = reports_under_tmp / "outputs" / "fifi" / "fifi.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    args = wake_trainer.build_parser().parse_args(["install", "fifi"])
    assert wake_trainer.cmd_install(args) == 1
    assert "refusing to install" in capsys.readouterr().out


def test_install_wires_through_to_install_model(reports_under_tmp, monkeypatch, capsys):
    model = reports_under_tmp / "outputs" / "fifi" / "fifi.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    (reports_under_tmp / "reports" / "fifi-eval.json").write_text(
        json.dumps(_eval_report()), encoding="utf-8"
    )
    captured = {}

    def fake_install_model(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return {"status": "ok", "target": "T", "hash": "a" * 64, "size_bytes": 4,
                "metadata": "M", "backup": {}}

    monkeypatch.setattr(wake_trainer.install, "install_model", fake_install_model)
    args = wake_trainer.build_parser().parse_args(["install", "fifi", "--force"])
    assert wake_trainer.cmd_install(args) == 0
    assert captured["candidate"] == "fifi"
    assert captured["phrase"] == "fifi"
    assert captured["threshold"] == 0.5   # from the eval report's selected_threshold
    assert captured["force"] is True
    assert "Installed" in capsys.readouterr().out


def test_install_missing_model_errors(reports_under_tmp, capsys):
    args = wake_trainer.build_parser().parse_args(["install", "fifi"])
    assert wake_trainer.cmd_install(args) == 1
    assert "not found" in capsys.readouterr().out


# --- calibrate (summary) -----------------------------------------------------------


def test_calibrate_summarizes_store(reports_under_tmp, tmp_path, capsys):
    store = tmp_path / "cal.jsonl"
    store.write_text("\n".join(json.dumps(o) for o in [
        {"score": 0.8, "label": "correct"},
        {"score": 0.2, "label": "false_positive"},
    ]), encoding="utf-8")
    args = wake_trainer.build_parser().parse_args(
        ["calibrate", "--store", str(store), "--threshold", "0.5"]
    )
    assert wake_trainer.cmd_calibrate(args) == 0
    out = capsys.readouterr().out
    assert "Recommended thr" in out
