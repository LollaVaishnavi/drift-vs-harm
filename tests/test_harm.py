"""Harm is the labelled truth; these tests pin when it is measurable and when it is not."""

from __future__ import annotations

import numpy as np

from driftharm.harm import expected_calibration_error, measure_harm

RNG = np.random.default_rng(200)


def _well_ranked(n, rng):
    labels = rng.integers(0, 2, n)
    # Scores correlated with the label, plus noise, so AUC is high but below 1.
    scores = np.clip(0.5 + 0.3 * (labels - 0.5) * 2 + rng.normal(0, 0.15, n), 0.01, 0.99)
    return labels, scores


def test_a_perfectly_calibrated_constant_has_small_ece():
    labels = (RNG.random(4000) < 0.3).astype(int)
    scores = np.full(4000, 0.3)
    assert expected_calibration_error(labels, scores) < 0.05


def test_ece_is_zero_for_an_empty_window():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_ece_sees_a_confident_but_wrong_model():
    labels = np.zeros(1000, dtype=int)
    scores = np.full(1000, 0.95)
    assert expected_calibration_error(labels, scores) > 0.9


def test_harm_is_flagged_when_auc_falls_past_the_threshold():
    ref_labels, ref_scores = _well_ranked(4000, np.random.default_rng(201))
    # Production: shuffle scores so ranking quality collapses to chance.
    prod_labels = np.random.default_rng(202).integers(0, 2, 4000)
    prod_scores = np.random.default_rng(203).random(4000)
    result = measure_harm(
        ref_labels, ref_scores, prod_labels, prod_scores, auc_drop_alarm=0.02
    )
    assert result.measurable
    assert result.harmed
    assert result.auc_drop > 0.02


def test_no_harm_is_flagged_when_ranking_is_preserved():
    ref_labels, ref_scores = _well_ranked(4000, np.random.default_rng(204))
    prod_labels, prod_scores = _well_ranked(4000, np.random.default_rng(205))
    result = measure_harm(
        ref_labels, ref_scores, prod_labels, prod_scores, auc_drop_alarm=0.02
    )
    assert result.measurable
    assert not result.harmed


def test_harm_is_not_measurable_when_a_window_has_one_class_only():
    ref_labels, ref_scores = _well_ranked(1000, np.random.default_rng(206))
    prod_labels = np.ones(1000, dtype=int)  # one class, AUC undefined
    prod_scores = np.random.default_rng(207).random(1000)
    result = measure_harm(
        ref_labels, ref_scores, prod_labels, prod_scores, auc_drop_alarm=0.02
    )
    assert not result.measurable
    assert result.auc_drop is None
    assert not result.harmed  # 'could not answer' is never reported as harm


def test_not_measurable_when_the_reference_window_has_one_class():
    ref_labels = np.zeros(1000, dtype=int)
    ref_scores = np.random.default_rng(208).random(1000)
    prod_labels, prod_scores = _well_ranked(1000, np.random.default_rng(209))
    result = measure_harm(
        ref_labels, ref_scores, prod_labels, prod_scores, auc_drop_alarm=0.02
    )
    assert not result.measurable
    assert result.reference_auc is None


def test_logloss_and_ece_are_reported_even_when_auc_is_undefined():
    ref_labels = np.ones(500, dtype=int)
    ref_scores = np.full(500, 0.6)
    prod_labels, prod_scores = _well_ranked(500, np.random.default_rng(210))
    result = measure_harm(
        ref_labels, ref_scores, prod_labels, prod_scores, auc_drop_alarm=0.02
    )
    assert result.reference_logloss > 0.0
    assert result.production_ece >= 0.0
