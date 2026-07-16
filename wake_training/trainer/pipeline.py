"""Training orchestration: seed, generate, augment, train, locate the ONNX.

This drives openWakeWord's official config-driven trainer

    python -m openwakeword.train --training_config <cfg> \
        --generate_clips --augment_clips --train_model

inside the ISOLATED wake_training/.venv. The heavy work happens in a subprocess
(injectable as ``runner`` for tests), so this module itself imports nothing
heavy. Reproducibility is enforced by seeding every RNG we can reach and pinning
PYTHONHASHSEED in the child's environment.

openWakeWord auto-exports ``<output_dir>/<model_name>.onnx`` after training; we
verify that artifact exists and write a training report for provenance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .config import TrainingConfig

STAGES = ("generate", "augment", "train")
_STAGE_FLAG = {
    "generate": "--generate_clips",
    "augment": "--augment_clips",
    "train": "--train_model",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_venv_python() -> Path:
    """The isolated trainer interpreter (wake_training/.venv)."""
    if sys.platform == "win32":
        return paths.VENV_DIR / "Scripts" / "python.exe"
    return paths.VENV_DIR / "bin" / "python"


# --- reproducibility --------------------------------------------------------------


def seed_everything(seed: int) -> dict:
    """Seed random / numpy / torch (best-effort) for a reproducible run.

    Returns a record of what was actually seeded. torch is optional here: the
    real training subprocess seeds itself from the same value via the env, this
    only makes the *orchestrator* deterministic.
    """
    import random

    seeded = {"seed": seed, "python_random": True, "numpy": False, "torch": False}
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    except Exception:
        pass
    return seeded


def subprocess_env(seed: int, base_env: dict | None = None) -> dict:
    """Child environment with the seed pinned (deterministic hashing + seeds)."""
    env = dict(base_env if base_env is not None else os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["WAKE_TRAIN_SEED"] = str(seed)
    return env


# --- locating the openWakeWord trainer --------------------------------------------


def locate_train_entry(venv_python: str | Path, runner=subprocess.run) -> list[str]:
    """Return the argv prefix that runs openWakeWord's trainer in the venv.

    Prefers ``-m openwakeword.train``; falls back to the train.py file inside the
    installed package. Raises RuntimeError if openWakeWord is not importable.
    """
    probe = (
        "import importlib.util, os, sys;"
        "spec = importlib.util.find_spec('openwakeword');"
        "print('' if spec is None else os.path.dirname(spec.origin))"
    )
    result = runner(
        [str(venv_python), "-c", probe], capture_output=True, text=True, timeout=60
    )
    lines = (result.stdout or "").strip().splitlines()
    pkg_dir = lines[-1].strip() if lines else ""
    if not pkg_dir:
        raise RuntimeError(
            "openWakeWord is not installed in the training venv. Run: "
            "python wake_training/scripts/setup.py"
        )
    train_py = Path(pkg_dir) / "train.py"
    if train_py.is_file():
        return [str(venv_python), str(train_py)]
    # Package present but train.py not shipped — use the module runner.
    return [str(venv_python), "-m", "openwakeword.train"]


def build_train_command(
    entry: list[str],
    config_path: str | Path,
    stages: tuple[str, ...] = STAGES,
    overwrite: bool = False,
) -> list[str]:
    """Assemble the full trainer argv for the requested stages."""
    cmd = list(entry) + ["--training_config", str(config_path)]
    for stage in STAGES:  # keep canonical order regardless of caller order
        if stage in stages:
            cmd.append(_STAGE_FLAG[stage])
    if overwrite:
        cmd.append("--overwrite")
    return cmd


# --- the run ----------------------------------------------------------------------


def prepare_run(config: TrainingConfig, name: str) -> dict:
    """Write the generated openWakeWord config + hard-negative manifest.

    Returns the paths involved. Creates outputs/<name>/ and reports/.
    """
    from . import datasets

    paths.ensure_workspace_dirs()
    output_dir = paths.candidate_output_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_config = output_dir / f"{name}.training.yaml"
    config.write_openwakeword_config(
        gen_config,
        output_dir=output_dir,
        piper_sample_generator_path=paths.PIPER_DIR,
    )
    neg_manifest = output_dir / f"{name}.hard_negatives.txt"
    datasets.write_hard_negative_manifest(config, neg_manifest)
    return {
        "output_dir": str(output_dir),
        "generated_config": str(gen_config),
        "hard_negative_manifest": str(neg_manifest),
    }


def run_training(
    config: TrainingConfig,
    name: str,
    *,
    venv_python: str | Path | None = None,
    entry: list[str] | None = None,
    runner=subprocess.run,
    stages: tuple[str, ...] = STAGES,
    overwrite: bool = False,
    write_report: bool = True,
) -> dict:
    """Full training run for a candidate. ``runner`` is injectable for tests.

    Steps: validate -> seed -> prepare -> run trainer subprocess -> verify the
    exported ONNX exists -> write a provenance report. Never raises for a normal
    training failure; returns ``{"status": "error", ...}`` so the CLI can report
    it cleanly.
    """
    problems = config.validate()
    if problems:
        return {"status": "error", "stage": "validate", "problems": problems}

    venv_python = venv_python or default_venv_python()
    seeded = seed_everything(config.random_seed)
    prepared = prepare_run(config, name)

    if entry is None:
        try:
            entry = locate_train_entry(venv_python, runner=runner)
        except RuntimeError as exc:
            return {"status": "error", "stage": "locate", "message": str(exc)}

    command = build_train_command(
        entry, prepared["generated_config"], stages=stages, overwrite=overwrite
    )
    started = _utc()
    completed = runner(
        command,
        cwd=str(paths.WAKE_TRAINING_ROOT),
        env=subprocess_env(config.random_seed),
    )
    return_code = getattr(completed, "returncode", 1)
    model_path = paths.candidate_model_path(name)

    result = {
        "status": "ok" if (return_code == 0 and model_path.is_file()) else "error",
        "candidate": name,
        "seed": seeded,
        "stages": list(stages),
        "command": command,
        "return_code": return_code,
        "model_path": str(model_path),
        "model_present": model_path.is_file(),
        "generated_config": prepared["generated_config"],
        "started_utc": started,
        "finished_utc": _utc(),
    }
    if result["status"] == "error" and return_code == 0 and not model_path.is_file():
        result["message"] = (
            f"trainer exited 0 but {model_path.name} was not produced "
            f"(expected at {model_path})"
        )
    if write_report:
        report_path = paths.candidate_report_path(name, "train")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["report_path"] = str(report_path)
    return result
