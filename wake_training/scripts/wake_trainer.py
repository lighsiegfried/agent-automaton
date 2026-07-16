"""Fifi wake-word trainer CLI (Phase 3D.2) — isolated workspace.

    py=wake_training/.venv/Scripts/python.exe
    $py wake_training/scripts/wake_trainer.py status
    $py wake_training/scripts/wake_trainer.py train fifi        [--dry-run] [--overwrite]
    $py wake_training/scripts/wake_trainer.py train oye_fifi
    $py wake_training/scripts/wake_trainer.py evaluate fifi --manifest <clips.json>
    $py wake_training/scripts/wake_trainer.py compare
    $py wake_training/scripts/wake_trainer.py calibrate [--store <jsonl>] [--threshold X]
    $py wake_training/scripts/wake_trainer.py install fifi      [--force] [--threshold X]

`train`/`evaluate` need the isolated venv (openWakeWord, torch). `status`,
`compare`, `calibrate` (summary), and `install` run in the main venv too.
Python (not PowerShell) on purpose: AllSigned Group Policy blocks unsigned .ps1.
"""

import argparse
import json
import sys
from pathlib import Path

WAKE_TRAINING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WAKE_TRAINING_ROOT.parent
for _p in (str(WAKE_TRAINING_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainer import config as cfg  # noqa: E402
from trainer import datasets, evaluate, install, metadata, paths, pipeline, verifier  # noqa: E402


def discover_candidates() -> list[str]:
    """Candidate config names in config/ (everything except defaults.yaml)."""
    return sorted(
        p.stem for p in paths.CONFIG_DIR.glob("*.yaml") if p.stem != "defaults"
    )


def _load(candidate: str):
    return cfg.load_config(candidate)


def _phrase(config) -> str:
    return config.target_phrase[0] if config.target_phrase else config.model_name


# --- status ------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    print("=== Fifi wake trainer: status ===")
    print(f"Workspace       : {paths.WAKE_TRAINING_ROOT}")
    venv = pipeline.default_venv_python()
    print(f"Isolated venv   : {'present' if venv.exists() else 'MISSING (run setup.py)'}")
    print(f"Piper generator : {'present' if paths.PIPER_DIR.exists() else 'not installed'}")
    print(f"Candidates      : {', '.join(discover_candidates()) or '(none)'}")

    for name in discover_candidates():
        try:
            config = _load(name)
        except cfg.ConfigError as exc:
            print(f"  [{name}] INVALID CONFIG: {exc}")
            continue
        model = paths.candidate_model_path(name)
        eval_report = paths.candidate_report_path(name, "eval")
        ds = datasets.dataset_status(config)
        ver = verifier.verifier_status(config)
        print(
            f"  [{name}] phrase={_phrase(config)!r} trained={model.is_file()} "
            f"evaluated={eval_report.is_file()} data_ready={ds['ready_to_train']} "
            f"bg_hours={ds['background_hours']} verifier={ver['enabled']}"
        )

    # Installed runtime model.
    target = paths.INSTALL_TARGET
    print(f"Installed model : {'present' if target.is_file() else 'MISSING'} ({target})")
    if target.is_file():
        check = metadata.verify_installed(target, paths.METADATA_TARGET)
        doc = check.get("metadata") or {}
        print(
            f"  metadata      : phrase={doc.get('phrase')!r} "
            f"threshold={doc.get('threshold')} version={doc.get('training_version')} "
            f"hash_ok={check['ok']}"
        )
        if not check["ok"]:
            for problem in check["problems"]:
                print(f"    ! {problem}")
    return 0


# --- train -------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    print(f"=== Fifi wake trainer: train {args.candidate} ===")
    try:
        config = _load(args.candidate)
    except cfg.ConfigError as exc:
        print(str(exc))
        return 1

    if args.dry_run:
        prepared = pipeline.prepare_run(config, args.candidate)
        print(f"Generated config: {prepared['generated_config']}")
        print(f"Hard negatives  : {prepared['hard_negative_manifest']}")
        print(f"Output dir      : {prepared['output_dir']}")
        print("Dry run — no training was launched.")
        return 0

    stages = tuple(args.stages) if args.stages else pipeline.STAGES
    result = pipeline.run_training(
        config, args.candidate, stages=stages, overwrite=args.overwrite
    )
    if result["status"] != "ok":
        print(f"Training FAILED : {result.get('message') or result.get('problems') or result}")
        return 1
    print(f"Model           : {result['model_path']}")
    print(f"Seed            : {result['seed']['seed']}")
    print(f"Report          : {result.get('report_path')}")
    print("Next            : evaluate it, then install the better candidate.")
    return 0


# --- evaluate ----------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace, *, score_runner_factory=None) -> int:
    print(f"=== Fifi wake trainer: evaluate {args.candidate} ===")
    try:
        config = _load(args.candidate)
    except cfg.ConfigError as exc:
        print(str(exc))
        return 1
    if not args.manifest:
        print("Provide a labelled validation manifest with --manifest <clips.json>.")
        print("Each entry: {path, label(0|1), condition, duration_seconds[, score]}.")
        return 1
    clips = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not clips:
        print("Manifest is empty.")
        return 1

    if all("score" in c for c in clips):
        scored = [
            {"label": int(c["label"]), "condition": c.get("condition", "unknown"),
             "duration_seconds": float(c.get("duration_seconds", 0.0)),
             "score": float(c["score"])}
            for c in clips
        ]
    else:
        factory = score_runner_factory or evaluate.onnx_score_runner
        runner = factory(paths.candidate_model_path(args.candidate))
        scored = evaluate.score_clips(runner, clips)

    report = evaluate.evaluate_scored(
        scored,
        conditions=config.evaluation.conditions,
        target_false_positives_per_hour=config.evaluation.target_false_positives_per_hour,
        min_recall=config.evaluation.min_recall,
        model_size=evaluate.model_size_bytes(paths.candidate_model_path(args.candidate)),
    )
    report["candidate"] = args.candidate
    out = paths.candidate_report_path(args.candidate, "eval")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Recall          : {report['recall']}  (FRR {report['false_rejection_rate']})")
    print(f"False pos/hour  : {report['false_positives_per_hour']}")
    print(f"Precision       : {report['precision']}")
    print(f"Threshold       : {report['selected_threshold']} "
          f"(meets targets: {report['threshold_selection']['met_targets']})")
    print(f"Model size      : {report['model_size_kb']} KB")
    print("By condition    :")
    for cond, m in report["by_condition"].items():
        if m.get("no_data"):
            print(f"  {cond:<12}: (no data)")
        else:
            print(f"  {cond:<12}: recall={m['recall']} fp/h={m['false_positives_per_hour']}")
    print(f"Report          : {out}")
    print("NOTE: selection uses evaluation metrics, NOT training accuracy.")
    return 0


# --- compare -----------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    print("=== Fifi wake trainer: compare ===")
    reports = {}
    for name in discover_candidates():
        path = paths.candidate_report_path(name, "eval")
        if path.is_file():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if not reports:
        print("No evaluation reports found — run 'evaluate <candidate>' first.")
        return 1
    ranked = evaluate.rank_candidates(reports)
    for row in ranked:
        print(
            f"#{row['rank']} {row['candidate']:<10} "
            f"targets={'yes' if row['met_targets'] else 'no '} "
            f"recall={row['recall']} fp/h={row['false_positives_per_hour']} "
            f"latency={row['latency_ms_mean']}ms size={row['model_size_bytes']}B "
            f"thr={row['selected_threshold']}"
        )
    winner = ranked[0]
    print(f"Recommended     : {winner['candidate']} "
          f"(install with: install {winner['candidate']})")
    return 0


# --- calibrate (summary + recommendation) -----------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    from app.voice import wake_calibration

    print("=== Fifi wake trainer: calibrate ===")
    store_path = args.store or wake_calibration.DEFAULT_STORE_PATH
    threshold = args.threshold
    if threshold is None:
        doc = metadata.read_metadata(paths.METADATA_TARGET) or {}
        threshold = float(doc.get("threshold", 0.5))
    summary = wake_calibration.summarize_store(store_path, threshold)
    print(f"Store           : {summary['path']}")
    print(f"Observations    : {summary['total_observations']} {summary['label_counts']}")
    rec = summary["recommendation"]
    print(f"Current thr     : {rec['current']}")
    print(f"Recommended thr : {rec['recommended']}  ({rec['rationale']})")
    print(f"  true={rec['n_true']} false={rec['n_false']} separable={rec['separable']}")
    print("Live capture    : run 'python scripts/local_runtime.py wake-calibrate' first "
          "(it never executes commands).")
    return 0


# --- install -----------------------------------------------------------------------


def cmd_install(args: argparse.Namespace) -> int:
    print(f"=== Fifi wake trainer: install {args.candidate} ===")
    try:
        config = _load(args.candidate)
    except cfg.ConfigError as exc:
        print(str(exc))
        return 1
    model = paths.candidate_model_path(args.candidate)
    if not model.is_file():
        print(f"Candidate model not found: {model} — train it first.")
        return 1

    eval_report = {}
    report_path = paths.candidate_report_path(args.candidate, "eval")
    if report_path.is_file():
        eval_report = json.loads(report_path.read_text(encoding="utf-8"))
    elif not args.threshold:
        print("No evaluation report and no --threshold: refusing to install a model "
              "chosen without evaluation. Run 'evaluate' first (or pass --threshold).")
        return 1

    threshold = args.threshold if args.threshold is not None else eval_report.get(
        "selected_threshold", config.training.target_recall
    )
    metrics = {
        k: eval_report.get(k)
        for k in ("recall", "false_rejection_rate", "false_positives_per_hour",
                  "precision", "by_condition", "latency", "model_size_bytes")
        if k in eval_report
    }
    ds = datasets.dataset_status(config)
    provenance = {
        "config": args.candidate,
        "seed": config.random_seed,
        "language": config.language,
        "n_samples": config.positive.n_samples,
        "background_hours": ds["background_hours"],
        "evaluation_report": str(report_path) if report_path.is_file() else "",
    }
    result = install.install_model(
        model, candidate=args.candidate, phrase=_phrase(config), threshold=threshold,
        metrics=metrics, provenance=provenance, force=args.force,
    )
    if result["status"] != "ok":
        print(f"Install {result['status'].upper()}: {result.get('message')}")
        if result.get("reason") == "already_installed":
            print("  Re-run with --force to replace it (a versioned backup is kept).")
        if result.get("validation", {}).get("problems"):
            for p in result["validation"]["problems"]:
                print(f"    ! {p}")
        return 1
    print(f"Installed       : {result['target']}")
    print(f"Hash            : {result['hash'][:16]}…  size={result['size_bytes']}B")
    print(f"Metadata        : {result['metadata']}")
    if result["backup"]:
        print(f"Backup          : {result['backup'].get('model')}")
    print("Runtime         : validate with 'python scripts/local_runtime.py wake-doctor'.")
    return 0


# --- parser ------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fifi wake-word trainer CLI.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="workspace, candidates, datasets, install").set_defaults(
        func=cmd_status
    )

    p_train = sub.add_parser("train", help="train a candidate (fifi | oye_fifi)")
    p_train.add_argument("candidate")
    p_train.add_argument("--dry-run", action="store_true", help="prepare config only")
    p_train.add_argument("--overwrite", action="store_true", help="overwrite augmented features")
    p_train.add_argument("--stages", nargs="*", choices=pipeline.STAGES,
                         help="subset of generate/augment/train (default: all)")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a trained candidate")
    p_eval.add_argument("candidate")
    p_eval.add_argument("--manifest", help="labelled validation clips JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    sub.add_parser("compare", help="rank candidates by evaluation metrics").set_defaults(
        func=cmd_compare
    )

    p_cal = sub.add_parser("calibrate", help="summarise calibration + recommend a threshold")
    p_cal.add_argument("--store", help="calibration JSONL (default: storage/runtime/...)")
    p_cal.add_argument("--threshold", type=float, help="current threshold to compare against")
    p_cal.set_defaults(func=cmd_calibrate)

    p_install = sub.add_parser("install", help="validate + install a candidate as the active model")
    p_install.add_argument("candidate")
    p_install.add_argument("--force", action="store_true", help="replace an installed model (backed up)")
    p_install.add_argument("--threshold", type=float, help="override the installed threshold")
    p_install.set_defaults(func=cmd_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
