"""The claims the README leads with, pinned as tests so they cannot quietly rot.

These are the properties a reviewer is really auditing: that a population-drift
statistic is blind to concept drift, that it fires on harmless covariate shift, and
that the PSI alarm can flip on nothing but a binning choice. If any of these breaks,
the repository's headline is false and the build must go red.
"""

from __future__ import annotations

import numpy as np

from driftharm.config import load_policy
from driftharm.confusion import (
    confusion_matrix,
    generous_alarm,
    psi_only,
    recall_on_family,
)
from driftharm.drift import psi
from driftharm.generator import draw_window
from driftharm.harness import _scenario_seed, run_scenario, run_suite
from driftharm.scenarios import (
    WAR_BINNING_SCALE,
    default_suite,
    suite_by_family,
    war_binning_spec,
)


def test_psi_is_blind_to_every_concept_drift_scenario(suite_outcomes):
    # The structural zero: no concept-drift scenario moves P(X), so PSI cannot fire
    # on one, however large the AUC drop underneath it.
    concept = [o for o in suite_outcomes if o.kind == "concept"]
    assert concept, "the suite must contain concept-drift scenarios"
    assert all(o.harmed for o in concept)
    assert recall_on_family(suite_outcomes, "concept", psi_only).recall == 0.0


def test_the_concept_family_carries_real_harm(suite_outcomes):
    drops = [o.harm.auc_drop for o in suite_outcomes if o.kind == "concept"]
    assert min(drops) > 0.02  # every one is a genuine, measurable degradation
    assert max(drops) > 0.2  # and the worst is large


def test_psi_recall_on_concept_equals_its_alarm_rate_on_null(suite_outcomes):
    # A label-blind detector cannot tell concept drift from nothing, so its recall on
    # concept drift is bounded by its false-positive rate on null. For PSI both are
    # exactly zero, which is the cleanest possible statement of the blindness.
    concept = [o for o in suite_outcomes if o.kind == "concept"]
    null = [o for o in suite_outcomes if o.kind == "null"]
    psi_recall_concept = sum(o.psi_alarm for o in concept) / len(concept)
    psi_rate_null = sum(o.psi_alarm for o in null) / len(null)
    assert psi_recall_concept == psi_rate_null == 0.0


def test_covariate_shift_fires_psi_without_causing_harm(suite_outcomes):
    covariate = [o for o in suite_outcomes if o.kind == "covariate"]
    assert all(o.psi_alarm for o in covariate)  # the loud false alarms
    assert not any(o.harmed for o in covariate)  # that cost the model nothing


def test_a_drift_alarm_is_no_better_than_the_base_rate_at_predicting_harm(suite_outcomes):
    cm = confusion_matrix(suite_outcomes, generous_alarm)
    # Precision at or below prevalence means the alarm carries no information about
    # harm: knowing it fired does not raise your estimate that harm occurred.
    assert cm.precision is not None and cm.harm_prevalence is not None
    assert cm.precision <= cm.harm_prevalence + 1e-9


def test_the_psi_alarm_flips_on_the_binning_choice_alone():
    # ADR-005. One benign variance shift, two textbook binnings, opposite verdicts.
    policy = load_policy()
    from driftharm.harness import build_baseline

    baseline = build_baseline(policy)
    spec = war_binning_spec()
    rng = np.random.default_rng(_scenario_seed(policy.run.seed, spec.name))
    window = draw_window(rng, spec, policy.run.production_window, baseline.model.score)
    reference = baseline.signal_window.features[:, 0]
    production = window.features[:, 0]

    quantile = psi(reference, production, policy.drift.psi_bins, "quantile")
    equal_width = psi(reference, production, policy.drift.psi_bins, "equal_width")

    assert quantile < policy.drift.psi_alarm  # quiet under the default
    assert equal_width >= policy.drift.psi_alarm  # alarm under equal width
    # And the scenario is harmless, so the equal-width alarm is a pure artifact.
    outcome = run_scenario(policy, baseline, spec)
    assert not outcome.harmed


def test_the_war_story_scale_is_the_committed_value():
    # Guards against someone quietly changing the shift until the flip disappears.
    assert WAR_BINNING_SCALE == 1.4


def test_the_whole_suite_is_byte_identical_across_runs(policy, baseline):
    first = run_suite(policy, default_suite(), baseline=baseline)
    second = run_suite(policy, default_suite(), baseline=baseline)
    for a, b in zip(first, second, strict=True):
        assert a.max_psi == b.max_psi
        assert a.harm.auc_drop == b.harm.auc_drop
        assert a.drift_alarm == b.drift_alarm


def test_the_suite_covers_the_four_families(suite_outcomes):
    kinds = {o.kind for o in suite_outcomes}
    assert kinds == set(suite_by_family(default_suite()))
