"""Candidate training configuration — load, validate, and translate to the
openWakeWord training_config.

Our config (``config/*.yaml``) is a friendly layer over openWakeWord's flat
training config. A candidate file sets ``extends: defaults.yaml`` and overrides
only what differs; ``load_config`` resolves the inheritance (deep-merge maps,
REPLACE lists), validates every field, and can emit the exact flat YAML that
``openwakeword/train.py --training_config`` consumes.

Pure and dependency-light: only PyYAML + stdlib, so config validation is fully
testable in the main project's .venv (no torch / openWakeWord needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths


class ConfigError(ValueError):
    """Raised when a candidate config is missing or fails validation."""


# --- nested config sections --------------------------------------------------------


@dataclass
class PositiveConfig:
    n_samples: int = 20000
    n_samples_val: int = 2000
    piper_voices: list[str] = field(default_factory=list)
    length_scales: list[float] = field(default_factory=lambda: [1.0])
    noise_scales: list[float] = field(default_factory=lambda: [0.667])


@dataclass
class AugmentationConfig:
    rounds: int = 2
    batch_size: int = 16
    pitch_semitones: list[float] = field(default_factory=lambda: [-3.0, 3.0])
    speed: list[float] = field(default_factory=lambda: [0.9, 1.1])
    gain_db: list[float] = field(default_factory=lambda: [-6.0, 6.0])
    noise_snr_db: list[float] = field(default_factory=lambda: [5.0, 25.0])
    reverb: bool = True


@dataclass
class NegativesConfig:
    background_paths: list[str] = field(default_factory=list)
    background_duplication_rate: int = 1
    rir_paths: list[str] = field(default_factory=list)
    false_positive_validation_data: str = ""


@dataclass
class TrainingParams:
    steps: int = 50000
    max_negative_weight: int = 1500
    model_type: str = "dnn"
    layer_size: int = 96
    batch_n_per_class: int = 1024
    target_false_positives_per_hour: float = 0.2
    target_recall: float = 0.6


@dataclass
class EvaluationConfig:
    target_false_positives_per_hour: float = 0.5
    min_recall: float = 0.7
    conditions: list[str] = field(
        default_factory=lambda: ["quiet", "noisy", "far_field"]
    )


@dataclass
class VerifierConfig:
    enabled: bool = False
    threshold: float = 0.3
    reference_dir: str = "datasets/positive/user"


@dataclass
class TrainingConfig:
    model_name: str
    target_phrase: list[str]
    language: str = "es"
    random_seed: int = 1234
    positive: PositiveConfig = field(default_factory=PositiveConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    hard_negative_phrases: list[str] = field(default_factory=list)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    # The raw merged mapping, kept for provenance in reports/metadata.
    source: dict[str, Any] = field(default_factory=dict)

    # -- validation ----------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of human-readable problems ([] means valid)."""
        problems: list[str] = []

        def bad(msg: str) -> None:
            problems.append(msg)

        if not self.model_name or not str(self.model_name).strip():
            bad("model_name is required")
        if not isinstance(self.target_phrase, list) or not self.target_phrase:
            bad("target_phrase must be a non-empty list of phrases")
        elif any(not isinstance(p, str) or not p.strip() for p in self.target_phrase):
            bad("every target_phrase entry must be a non-empty string")
        if not isinstance(self.language, str) or not self.language.strip():
            bad("language is required (this project trains Spanish: 'es')")
        if not isinstance(self.random_seed, int):
            bad("random_seed must be an integer (reproducibility)")

        p = self.positive
        if p.n_samples <= 0:
            bad("positive.n_samples must be > 0")
        if p.n_samples_val <= 0:
            bad("positive.n_samples_val must be > 0")
        if not p.piper_voices:
            bad("positive.piper_voices must list at least one voice (multi-voice)")
        if any(s <= 0 for s in p.length_scales):
            bad("positive.length_scales must all be > 0")
        if any(s <= 0 for s in p.noise_scales):
            bad("positive.noise_scales must all be > 0")

        a = self.augmentation
        if a.rounds < 0:
            bad("augmentation.rounds must be >= 0")
        if a.batch_size <= 0:
            bad("augmentation.batch_size must be > 0")
        for name, rng, positive_only in (
            ("pitch_semitones", a.pitch_semitones, False),
            ("speed", a.speed, True),
            ("gain_db", a.gain_db, False),
            ("noise_snr_db", a.noise_snr_db, True),
        ):
            if len(rng) != 2 or rng[0] > rng[1]:
                bad(f"augmentation.{name} must be [min, max] with min <= max")
            elif positive_only and rng[0] <= 0:
                bad(f"augmentation.{name} must be > 0")

        if not isinstance(self.hard_negative_phrases, list):
            bad("hard_negative_phrases must be a list")
        elif any(
            not isinstance(x, str) or not x.strip() for x in self.hard_negative_phrases
        ):
            bad("every hard_negative_phrases entry must be a non-empty string")

        n = self.negatives
        if n.background_duplication_rate < 1:
            bad("negatives.background_duplication_rate must be >= 1")

        t = self.training
        if t.steps <= 0:
            bad("training.steps must be > 0")
        if t.max_negative_weight < 0:
            bad("training.max_negative_weight must be >= 0")
        if t.model_type not in ("dnn", "rnn"):
            bad("training.model_type must be 'dnn' or 'rnn'")
        if t.layer_size <= 0:
            bad("training.layer_size must be > 0")
        if t.batch_n_per_class <= 0:
            bad("training.batch_n_per_class must be > 0")
        if t.target_false_positives_per_hour < 0:
            bad("training.target_false_positives_per_hour must be >= 0")
        if not 0.0 <= t.target_recall <= 1.0:
            bad("training.target_recall must be within [0, 1]")

        e = self.evaluation
        if e.target_false_positives_per_hour < 0:
            bad("evaluation.target_false_positives_per_hour must be >= 0")
        if not 0.0 <= e.min_recall <= 1.0:
            bad("evaluation.min_recall must be within [0, 1]")
        if not e.conditions:
            bad("evaluation.conditions must list at least one condition")

        v = self.verifier
        if not isinstance(v.enabled, bool):
            bad("verifier.enabled must be a boolean")
        if not 0.0 <= v.threshold <= 1.0:
            bad("verifier.threshold must be within [0, 1]")

        return problems

    def raise_if_invalid(self) -> "TrainingConfig":
        problems = self.validate()
        if problems:
            raise ConfigError(
                "invalid training config:\n  - " + "\n  - ".join(problems)
            )
        return self

    # -- openWakeWord translation --------------------------------------------------

    def to_openwakeword_config(
        self,
        output_dir: str | Path,
        piper_sample_generator_path: str | Path,
    ) -> dict[str, Any]:
        """The flat mapping ``openwakeword/train.py`` reads via --training_config.

        Only keys that openWakeWord's trainer understands are emitted here; our
        extra knobs (pitch/speed/gain ranges, per-voice generation) drive the
        piper generation + our augmentation pass, not this file.
        """
        return {
            "model_name": self.model_name,
            "target_phrase": list(self.target_phrase),
            "custom_negative_phrases": list(self.hard_negative_phrases),
            "n_samples": self.positive.n_samples,
            "n_samples_val": self.positive.n_samples_val,
            "steps": self.training.steps,
            "max_negative_weight": self.training.max_negative_weight,
            "target_false_positives_per_hour": (
                self.training.target_false_positives_per_hour
            ),
            "output_dir": str(output_dir),
            "background_paths": list(self.negatives.background_paths),
            "background_paths_duplication_rate": [
                self.negatives.background_duplication_rate
            ]
            * max(len(self.negatives.background_paths), 1),
            "rir_paths": list(self.negatives.rir_paths),
            "false_positive_validation_data_path": (
                self.negatives.false_positive_validation_data
            ),
            "augmentation_rounds": self.augmentation.rounds,
            "augmentation_batch_size": self.augmentation.batch_size,
            "tts_batch_size": max(1, min(50, self.positive.n_samples)),
            "model_type": self.training.model_type,
            "layer_size": self.training.layer_size,
            "batch_n_per_class": self.training.batch_n_per_class,
            "piper_sample_generator_path": str(piper_sample_generator_path),
        }

    def write_openwakeword_config(
        self,
        path: str | Path,
        output_dir: str | Path,
        piper_sample_generator_path: str | Path,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_openwakeword_config(output_dir, piper_sample_generator_path)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path


# --- loading + merging -------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge maps; a list/scalar in ``override`` REPLACES the base value.

    List-replace (not append) is deliberate: it lets oye_fifi drop "oye" from the
    default hard negatives simply by listing the negatives it wants.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_extends(data: dict, config_dir: Path, _seen: set[str] | None = None) -> dict:
    """Resolve a chain of ``extends:`` references (relative to config_dir)."""
    _seen = _seen or set()
    parent_ref = data.get("extends")
    if not parent_ref:
        return {k: v for k, v in data.items() if k != "extends"}
    if parent_ref in _seen:
        raise ConfigError(f"circular 'extends' reference at {parent_ref!r}")
    _seen.add(parent_ref)
    parent_path = config_dir / parent_ref
    if not parent_path.is_file():
        raise ConfigError(f"extends target not found: {parent_path}")
    parent_data = yaml.safe_load(parent_path.read_text(encoding="utf-8")) or {}
    parent_resolved = _resolve_extends(parent_data, config_dir, _seen)
    child = {k: v for k, v in data.items() if k != "extends"}
    return _deep_merge(parent_resolved, child)


def _section(cls, data: dict, key: str):
    """Build a nested dataclass from ``data[key]``, ignoring unknown fields."""
    raw = data.get(key) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config section {key!r} must be a mapping")
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown keys in {key!r}: {', '.join(sorted(unknown))}")
    return cls(**{k: v for k, v in raw.items() if k in known})


def from_mapping(data: dict) -> TrainingConfig:
    """Build a TrainingConfig from an already-merged mapping (no validation)."""
    return TrainingConfig(
        model_name=data.get("model_name", ""),
        target_phrase=data.get("target_phrase", []),
        language=data.get("language", "es"),
        random_seed=data.get("random_seed", 1234),
        positive=_section(PositiveConfig, data, "positive"),
        augmentation=_section(AugmentationConfig, data, "augmentation"),
        hard_negative_phrases=data.get("hard_negative_phrases", []),
        negatives=_section(NegativesConfig, data, "negatives"),
        training=_section(TrainingParams, data, "training"),
        evaluation=_section(EvaluationConfig, data, "evaluation"),
        verifier=_section(VerifierConfig, data, "verifier"),
        source=data,
    )


def resolve_config_path(name_or_path: str | Path) -> Path:
    """A candidate name ('fifi') -> config/fifi.yaml; a path is used as-is."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.exists():
        return candidate
    return paths.CONFIG_DIR / f"{name_or_path}.yaml"


def load_config(name_or_path: str | Path, *, validate: bool = True) -> TrainingConfig:
    """Load + merge (resolving ``extends``) a candidate config and validate it."""
    path = resolve_config_path(name_or_path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must be a mapping")
    merged = _resolve_extends(raw, path.parent)
    config = from_mapping(merged)
    if validate:
        config.raise_if_invalid()
    return config
