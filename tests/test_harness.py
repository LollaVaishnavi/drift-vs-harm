"""The harness keeps the detector side and the truth side apart; these pin that."""

from __future__ import annotations

import dataclasses

import numpy as np

from driftharm.harness import (
    _scenario_seed,
    _watched_signals,
    run_scenario,
    run_suite,
)
from driftharm.scenarios import concept_spec, covariate_spec, default_suite, null_spec


def test_a_scenario_seed_is_stable_for_a_name():
    assert _scenario_seed(1, "concept-f0") == _scenario_seed(1, "concept-f0")


def test_different_names_give_different_seeds():
    assert _scenario_seed(1, "a") != _scenario_seed(1, "b")


def test_the_master_seed_changes_every_scenario_seed():
    assert _scenario_seed(1, "a") != _scenario_seed(2, "a")


def test_build_baseline_returns_matching_window_sizes(policy, baseline):
    assert baseline.signal_window.features.shape[0] == policy.run.reference_window
    assert baseline.holdout.labels.shape[0] == policy.harm.label_sample


def test_a_concept_scenario_is_a_silent_failure(policy, baseline):
    spec = concept_spec("concept-probe", feature=0, delta=-2.5)
    outcome = run_scenario(policy, baseline, spec)
    assert outcome.harmed
    assert not outcome.psi_alarm
    assert outcome.silent_failure


def test_a_covariate_scenario_fires_psi_without_harm(policy, baseline):
    spec = covariate_spec("cov-probe", feature=4, shift=2.5)
    outcome = run_scenario(policy, baseline, spec)
    assert outcome.psi_alarm
    assert not outcome.harmed
    assert not outcome.silent_failure


def test_watching_only_the_score_yields_one_signal(policy, baseline):
    score_only = dataclasses.replace(policy.drift, watch=("score",))
    pol = dataclasses.replace(policy, drift=score_only)
    window = baseline.signal_window
    signals = _watched_signals(pol, baseline, window)
    assert [s.signal for s in signals] == ["score"]


def test_watching_only_features_yields_one_signal_per_column(policy, baseline):
    feat_only = dataclasses.replace(policy.drift, watch=("features",))
    pol = dataclasses.replace(policy, drift=feat_only)
    signals = _watched_signals(pol, baseline, baseline.signal_window)
    assert len(signals) == baseline.signal_window.features.shape[1]


def test_run_suite_builds_its_own_baseline_when_none_is_given(policy):
    outcomes = run_suite(policy, [null_spec("solo")])
    assert len(outcomes) == 1


def test_running_a_family_alone_matches_the_full_suite(policy, baseline, suite_outcomes):
    from driftharm.scenarios import suite_by_family

    concept_specs = suite_by_family(default_suite())["concept"]
    alone = run_suite(policy, concept_specs, baseline=baseline)
    full = {o.name: o for o in suite_outcomes if o.kind == "concept"}
    for outcome in alone:
        twin = full[outcome.name]
        assert np.isclose(outcome.harm.auc_drop, twin.harm.auc_drop)
        assert outcome.psi_alarm == twin.psi_alarm
        assert outcome.drift_alarm == twin.drift_alarm
