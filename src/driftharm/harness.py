"""The harness: run a scenario, collect what the detector sees and what is true.

This is where the two halves meet for the only time. For each scenario the harness
draws a production window, runs the label-blind detectors against a fixed reference
baseline, and separately measures harm against a labelled reference holdout. It
returns both verdicts side by side without ever letting one inform the other, which
is the experimental discipline the whole argument rests on: the detector's inputs
are features and scores, the harm measurement's inputs are labels, and the harness
keeps them apart.

Seeding is per scenario and derived from the policy master seed, so the suite is
byte-identical across runs and the order of scenarios never changes a number.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from driftharm.config import Policy
from driftharm.drift import SignalDrift, score_signal
from driftharm.generator import (
    N_FEATURES,
    LabelledWindow,
    ScenarioSpec,
    draw_window,
    null_spec,
)
from driftharm.harm import HarmResult, measure_harm
from driftharm.model import DeployedModel, train_reference_model


@dataclass(frozen=True)
class Baseline:
    """The frozen reference the monitor stores: a model and two reference windows.

    ``signal_window`` is the unlabelled baseline the detectors compare against;
    ``holdout`` is the labelled baseline harm is measured against. They are separate
    samples, as they would be in production, but both come from the unshifted world.
    """

    model: DeployedModel
    signal_window: LabelledWindow
    holdout: LabelledWindow


def build_baseline(policy: Policy) -> Baseline:
    """Train the model and capture the reference windows, all from one seed."""
    rng = np.random.default_rng(policy.run.seed)
    model = train_reference_model(rng)
    reference = null_spec("reference")
    signal_window = draw_window(rng, reference, policy.run.reference_window, model.score)
    holdout = draw_window(rng, reference, policy.harm.label_sample, model.score)
    return Baseline(model=model, signal_window=signal_window, holdout=holdout)


@dataclass(frozen=True)
class ScenarioOutcome:
    """Everything one scenario produced: the detector side and the truth side.

    ``drift_alarm`` is the generous reading: any detector on any watched signal.
    The per-detector aggregates below exist because the detectors fail differently.
    PSI and Jensen-Shannon read at their null level under concept drift and so give
    a deterministic structural zero; KS, run across many signals, throws
    multiple-comparison false alarms at the same rate on concept drift as on nothing
    at all. Reporting them apart is what lets the README lead with PSI, the canonical
    population-stability metric, rather than hide behind an aggregate.
    """

    name: str
    kind: str
    signals: tuple[SignalDrift, ...]
    drift_alarm: bool
    psi_alarm: bool
    ks_alarm: bool
    js_alarm: bool
    max_psi: float
    min_ks_p: float
    max_js: float
    harm: HarmResult

    @property
    def harmed(self) -> bool:
        return self.harm.harmed

    @property
    def measurable(self) -> bool:
        return self.harm.measurable

    @property
    def silent_failure(self) -> bool:
        """Harm happened and no detector fired: the blind spot."""
        return self.harmed and not self.drift_alarm


def _scenario_seed(master_seed: int, name: str) -> int:
    """Derive a scenario's seed from its name, not its position in the list.

    Keying on the name means filtering the suite to one family leaves every
    scenario's numbers identical to the full run. Python's built-in hash is salted
    per process, so a stable digest is used instead, which also keeps the suite
    byte-identical across machines.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:8]
    name_int = int.from_bytes(digest, "big")
    return int(np.random.SeedSequence([master_seed, name_int]).generate_state(1)[0])


def _watched_signals(
    policy: Policy,
    baseline: Baseline,
    window: LabelledWindow,
) -> list[SignalDrift]:
    drift = policy.drift
    results: list[SignalDrift] = []
    if "features" in drift.watch:
        for column in range(N_FEATURES):
            results.append(
                score_signal(
                    f"feature[{column}]",
                    baseline.signal_window.features[:, column],
                    window.features[:, column],
                    psi_alarm=drift.psi_alarm,
                    psi_bins=drift.psi_bins,
                    psi_binning=drift.psi_binning,
                    ks_alpha=drift.ks_alpha,
                    js_alarm=drift.js_alarm,
                )
            )
    if "score" in drift.watch:
        results.append(
            score_signal(
                "score",
                baseline.signal_window.scores,
                window.scores,
                psi_alarm=drift.psi_alarm,
                psi_bins=drift.psi_bins,
                psi_binning=drift.psi_binning,
                ks_alpha=drift.ks_alpha,
                js_alarm=drift.js_alarm,
            )
        )
    return results


def run_scenario(
    policy: Policy,
    baseline: Baseline,
    spec: ScenarioSpec,
) -> ScenarioOutcome:
    """Draw one production window under ``spec`` and score both halves."""
    rng = np.random.default_rng(_scenario_seed(policy.run.seed, spec.name))
    window = draw_window(rng, spec, policy.run.production_window, baseline.model.score)

    signals = _watched_signals(policy, baseline, window)
    drift_alarm = any(signal.any_fired for signal in signals)
    psi_alarm = any(s.psi_fired for s in signals)
    ks_alarm = any(s.ks_fired for s in signals)
    js_alarm = any(s.js_fired for s in signals)
    max_psi = max((s.psi for s in signals), default=0.0)
    min_ks_p = min((s.ks_p for s in signals), default=1.0)
    max_js = max((s.js for s in signals), default=0.0)

    harm = measure_harm(
        baseline.holdout.labels,
        baseline.holdout.scores,
        window.labels,
        window.scores,
        auc_drop_alarm=policy.harm.auc_drop_alarm,
    )

    return ScenarioOutcome(
        name=spec.name,
        kind=spec.kind,
        signals=tuple(signals),
        drift_alarm=drift_alarm,
        psi_alarm=psi_alarm,
        ks_alarm=ks_alarm,
        js_alarm=js_alarm,
        max_psi=max_psi,
        min_ks_p=min_ks_p,
        max_js=max_js,
        harm=harm,
    )


def run_suite(
    policy: Policy,
    specs: list[ScenarioSpec],
    baseline: Baseline | None = None,
) -> list[ScenarioOutcome]:
    """Run every scenario against one shared baseline, in order."""
    if baseline is None:
        baseline = build_baseline(policy)
    return [run_scenario(policy, baseline, spec) for spec in specs]
