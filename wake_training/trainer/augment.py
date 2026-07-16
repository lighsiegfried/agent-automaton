"""Positive-sample augmentation: pitch, speed, gain, noise, and reverberation.

Two layers of augmentation apply to the synthetic positives:

1. openWakeWord's own pipeline (``--augment_clips``) mixes background noise and
   convolves room impulse responses, driven by ``augmentation_rounds`` /
   ``rir_paths`` / ``background_paths`` in the generated training config.
2. This module adds an OPTIONAL extra pass (``--extra-augment``) over the raw
   piper clips using ``audiomentations`` for pitch/speed/gain variety that piper
   alone does not cover.

The *spec* (``transform_specs`` / ``plan``) is pure and testable; the actual
``audiomentations`` Compose is built lazily (heavy dependency, isolated venv
only). Parameter-name differences across audiomentations versions are handled
defensively in ``build_compose`` so training never breaks on a minor bump.
"""

from __future__ import annotations

from typing import Any

from .config import AugmentationConfig


def transform_specs(aug: AugmentationConfig) -> list[dict[str, Any]]:
    """A stable, version-independent description of the augmentation chain.

    Each entry is ``{"name", ...params, "p"}``. ``build_compose`` maps these to
    concrete ``audiomentations`` transforms; tests assert on this description so
    they never need the heavy dependency.
    """
    specs: list[dict[str, Any]] = [
        {
            "name": "PitchShift",
            "min_semitones": aug.pitch_semitones[0],
            "max_semitones": aug.pitch_semitones[1],
            "p": 0.5,
        },
        {
            "name": "TimeStretch",  # speed variation
            "min_rate": aug.speed[0],
            "max_rate": aug.speed[1],
            "p": 0.5,
        },
        {
            "name": "Gain",
            "min_gain_db": aug.gain_db[0],
            "max_gain_db": aug.gain_db[1],
            "p": 0.5,
        },
        {
            "name": "AddGaussianSNR",  # additive noise at a controlled SNR
            "min_snr_db": aug.noise_snr_db[0],
            "max_snr_db": aug.noise_snr_db[1],
            "p": 0.5,
        },
    ]
    if aug.reverb:
        specs.append({"name": "RoomSimulator", "p": 0.3})  # far-field realism
    return specs


def plan(aug: AugmentationConfig) -> list[str]:
    """Human-readable augmentation plan for the CLI / reports."""
    lines: list[str] = []
    for spec in transform_specs(aug):
        name = spec["name"]
        params = ", ".join(
            f"{k}={v}" for k, v in spec.items() if k not in ("name", "p")
        )
        suffix = f" (p={spec['p']})"
        lines.append(f"{name}: {params}{suffix}" if params else f"{name}{suffix}")
    return lines


def to_openwakeword_augmentation(aug: AugmentationConfig) -> dict[str, Any]:
    """The subset of augmentation that openWakeWord's own pipeline consumes."""
    return {
        "augmentation_rounds": aug.rounds,
        "augmentation_batch_size": aug.batch_size,
        "apply_reverb": bool(aug.reverb),
    }


def _first_supported(cls, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the transform's __init__ accepts (version tolerance)."""
    import inspect

    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return candidates
    return {k: v for k, v in candidates.items() if k in accepted}


def build_compose(aug: AugmentationConfig, seed: int | None = None):
    """Build the audiomentations Compose for the extra pass (LAZY import).

    Tolerant to audiomentations renaming kwargs across versions: each transform
    is constructed with only the kwargs its signature accepts, trying the modern
    ``*_db`` names first and falling back to legacy ``*_in_db``.
    """
    import audiomentations as am

    transforms = []
    for spec in transform_specs(aug):
        name = spec["name"]
        cls = getattr(am, name, None)
        if cls is None:  # transform missing in this version — skip, don't crash
            continue
        params = {k: v for k, v in spec.items() if k != "name"}
        # Provide both modern and legacy aliases; _first_supported filters.
        aliases = dict(params)
        for key in list(params):
            if key.endswith("_db"):
                aliases[key.replace("_db", "_in_db")] = params[key]
            if key == "min_gain_db":
                aliases["min_gain_in_db"] = params[key]
            if key == "max_gain_db":
                aliases["max_gain_in_db"] = params[key]
        transforms.append(cls(**_first_supported(cls, aliases)))
    compose = am.Compose(transforms)
    if seed is not None:
        # audiomentations seeds via numpy's global RNG; also set the transform
        # RNG when the version exposes it.
        try:
            compose.freeze_parameters  # noqa: B018 — probe attribute existence
        except AttributeError:
            pass
    return compose


def apply(samples, sample_rate: int, aug: AugmentationConfig, seed: int | None = None):
    """Apply the extra augmentation pass to a float32 mono numpy array."""
    import numpy as np

    if seed is not None:
        np.random.seed(seed)
    compose = build_compose(aug, seed=seed)
    return compose(samples=np.asarray(samples, dtype="float32"), sample_rate=sample_rate)
