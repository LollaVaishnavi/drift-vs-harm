"""Typed access to policy.toml.

This file exists so no threshold is ever read from a literal buried in code. The
whole argument of the repository is that the alarm a threshold produces does not
line up with harm, so the thresholds have to be visible, named and in one place a
reviewer can change. Loading is stdlib only (tomllib), so a cold clone needs no
extra dependency to read its own policy.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "policy.toml"

_VALID_BINNING = ("quantile", "equal_width")
_VALID_WATCH = ("features", "score")


@dataclass(frozen=True)
class DriftPolicy:
    """Thresholds for the label-blind detectors."""

    psi_alarm: float
    psi_bins: int
    psi_binning: str
    ks_alpha: float
    js_alarm: float
    watch: tuple[str, ...]


@dataclass(frozen=True)
class HarmPolicy:
    """What counts as the model getting worse, measured only where labels exist."""

    auc_drop_alarm: float
    label_sample: int


@dataclass(frozen=True)
class RunPolicy:
    """Seeding and window sizes that keep the suite byte-identical across runs."""

    seed: int
    reference_window: int
    production_window: int


@dataclass(frozen=True)
class Policy:
    """The whole monitoring policy, assembled from the three sections."""

    drift: DriftPolicy
    harm: HarmPolicy
    run: RunPolicy


def _require(mapping: dict, section: str, key: str):
    if key not in mapping:
        raise ValueError(f"policy section [{section}] is missing key '{key}'")
    return mapping[key]


def load_policy(path: str | Path | None = None) -> Policy:
    """Read and validate a policy file.

    Validation is strict on purpose: a silently defaulted threshold is a number
    nobody chose, and an unchosen number is exactly what this project argues
    against. An unknown binning strategy or watch signal is refused at load time
    rather than producing a quietly wrong alarm later.
    """
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    with policy_path.open("rb") as handle:
        raw = tomllib.load(handle)

    drift_raw = raw.get("drift", {})
    harm_raw = raw.get("harm", {})
    run_raw = raw.get("run", {})

    binning = _require(drift_raw, "drift", "psi_binning")
    if binning not in _VALID_BINNING:
        raise ValueError(f"psi_binning must be one of {_VALID_BINNING}, got {binning!r}")

    watch = tuple(_require(drift_raw, "drift", "watch"))
    if not watch:
        raise ValueError("drift.watch must list at least one of features, score")
    for signal in watch:
        if signal not in _VALID_WATCH:
            raise ValueError(f"drift.watch entries must be in {_VALID_WATCH}, got {signal!r}")

    bins = int(_require(drift_raw, "drift", "psi_bins"))
    if bins < 2:
        raise ValueError(f"psi_bins must be at least 2, got {bins}")

    drift = DriftPolicy(
        psi_alarm=float(_require(drift_raw, "drift", "psi_alarm")),
        psi_bins=bins,
        psi_binning=binning,
        ks_alpha=float(_require(drift_raw, "drift", "ks_alpha")),
        js_alarm=float(_require(drift_raw, "drift", "js_alarm")),
        watch=watch,
    )
    harm = HarmPolicy(
        auc_drop_alarm=float(_require(harm_raw, "harm", "auc_drop_alarm")),
        label_sample=int(_require(harm_raw, "harm", "label_sample")),
    )
    run = RunPolicy(
        seed=int(_require(run_raw, "run", "seed")),
        reference_window=int(_require(run_raw, "run", "reference_window")),
        production_window=int(_require(run_raw, "run", "production_window")),
    )
    return Policy(drift=drift, harm=harm, run=run)
