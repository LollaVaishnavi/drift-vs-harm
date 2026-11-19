"""Scoring the drift alarm as a classifier of harm, with every zero priced."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from driftharm.confusion import (
    Confusion,
    confusion_matrix,
    false_positive_rate,
    generous_alarm,
    js_only,
    ks_only,
    psi_harm_correlation,
    psi_only,
    recall_on_family,
    resolution_floor,
    silent_failures,
)


@dataclass
class FakeHarm:
    auc_drop: float | None


@dataclass
class FakeOutcome:
    """A minimal stand-in so confusion logic is tested without running the model."""

    name: str
    kind: str
    harmed: bool
    drift_alarm: bool
    psi_alarm: bool
    measurable: bool
    max_psi: float
    harm: FakeHarm

    @property
    def silent_failure(self) -> bool:
        return self.harmed and not self.drift_alarm

    @property
    def js_alarm(self) -> bool:
        return self.drift_alarm

    @property
    def ks_alarm(self) -> bool:
        return self.drift_alarm


def _o(kind, harmed, alarm, psi_alarm, measurable=True, psi=0.0, drop=0.0):
    return FakeOutcome(
        name=f"{kind}-{psi}",
        kind=kind,
        harmed=harmed,
        drift_alarm=alarm,
        psi_alarm=psi_alarm,
        measurable=measurable,
        max_psi=psi,
        harm=FakeHarm(auc_drop=drop),
    )


def test_resolution_floor_is_one_over_n():
    assert resolution_floor(15) == pytest.approx(1 / 15)


def test_resolution_floor_is_none_for_a_zero_sample():
    assert resolution_floor(0) is None


def test_precision_and_recall_on_a_known_matrix():
    outcomes = [
        _o("concept", harmed=True, alarm=False, psi_alarm=False),
        _o("concept", harmed=True, alarm=False, psi_alarm=False),
        _o("covariate", harmed=False, alarm=True, psi_alarm=True),
        _o("null", harmed=False, alarm=False, psi_alarm=False),
    ]
    cm = confusion_matrix(outcomes)
    assert (cm.true_positive, cm.false_positive, cm.false_negative, cm.true_negative) == (
        0, 1, 2, 1,
    )
    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.harm_prevalence == pytest.approx(0.5)


def test_precision_is_none_when_nothing_alarms():
    cm = confusion_matrix([_o("null", harmed=False, alarm=False, psi_alarm=False)])
    assert cm.precision is None


def test_recall_is_none_when_nothing_is_harmed():
    cm = confusion_matrix([_o("null", harmed=False, alarm=True, psi_alarm=True)])
    assert cm.recall is None
    assert cm.alarms == 1


def test_prevalence_is_none_for_an_empty_matrix():
    cm = confusion_matrix([])
    assert cm.harm_prevalence is None
    assert isinstance(cm, Confusion)


def test_not_measurable_scenarios_are_counted_apart():
    outcomes = [
        _o("concept", harmed=False, alarm=False, psi_alarm=False, measurable=False),
        _o("concept", harmed=True, alarm=True, psi_alarm=True),
    ]
    cm = confusion_matrix(outcomes)
    assert cm.not_measurable == 1
    assert cm.total == 1


def test_psi_predictor_gives_zero_recall_where_generous_gives_more():
    outcomes = [
        _o("concept", harmed=True, alarm=True, psi_alarm=False),  # ks fired, not psi
        _o("concept", harmed=True, alarm=False, psi_alarm=False),
    ]
    assert recall_on_family(outcomes, "concept", psi_only).recall == 0.0
    assert recall_on_family(outcomes, "concept", generous_alarm).recall == 0.5


def test_each_detector_predictor_reads_its_own_flag():
    harmed_js = _o("concept", harmed=True, alarm=True, psi_alarm=False)
    assert js_only(harmed_js) is True  # FakeOutcome maps js_alarm to drift_alarm
    quiet = _o("null", harmed=False, alarm=False, psi_alarm=False)
    assert js_only(quiet) is False


def test_false_positive_rate_counts_alarms_on_unharmed_scenarios():
    outcomes = [
        _o("null", harmed=False, alarm=True, psi_alarm=False),
        _o("null", harmed=False, alarm=False, psi_alarm=False),
        _o("concept", harmed=True, alarm=True, psi_alarm=False),
    ]
    assert false_positive_rate(outcomes, generous_alarm) == 0.5
    assert false_positive_rate(outcomes, ks_only) == 0.5


def test_false_positive_rate_is_none_when_all_scenarios_are_harmful():
    outcomes = [_o("concept", harmed=True, alarm=False, psi_alarm=False)]
    assert false_positive_rate(outcomes) is None


def test_silent_failures_lists_harm_with_no_alarm():
    outcomes = [
        _o("concept", harmed=True, alarm=False, psi_alarm=False),
        _o("mixed", harmed=True, alarm=True, psi_alarm=True),
    ]
    silent = silent_failures(outcomes)
    assert len(silent) == 1
    assert silent[0].kind == "concept"


def test_spearman_is_none_with_too_few_points():
    outcomes = [_o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.1, drop=0.1)]
    assert psi_harm_correlation(outcomes) is None


def test_spearman_is_none_when_psi_has_no_variance():
    outcomes = [
        _o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.01, drop=0.1),
        _o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.01, drop=0.2),
        _o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.01, drop=0.3),
    ]
    assert psi_harm_correlation(outcomes) is None


def test_spearman_is_negative_when_more_drift_means_less_harm():
    outcomes = [
        _o("covariate", harmed=False, alarm=True, psi_alarm=True, psi=3.0, drop=0.0),
        _o("mixed", harmed=True, alarm=True, psi_alarm=True, psi=1.0, drop=0.1),
        _o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.01, drop=0.3),
    ]
    assert psi_harm_correlation(outcomes) < 0


def test_spearman_skips_a_measurable_scenario_whose_drop_is_none():
    # A measurable scenario can still carry a None drop if AUC was undefined on one
    # side; the correlation must skip it rather than crash. Only two usable points
    # remain here, so the result is None.
    middle = _o("concept", harmed=False, alarm=False, psi_alarm=False, psi=0.02)
    middle.harm.auc_drop = None
    outcomes = [
        _o("concept", harmed=True, alarm=False, psi_alarm=False, psi=0.01, drop=0.3),
        middle,
        _o("covariate", harmed=False, alarm=True, psi_alarm=True, psi=3.0, drop=0.0),
    ]
    assert psi_harm_correlation(outcomes) is None
