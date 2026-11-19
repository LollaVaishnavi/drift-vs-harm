"""The detectors are standard; these tests pin their behaviour and their blind spot."""

from __future__ import annotations

import numpy as np
import pytest

from driftharm.drift import (
    bin_edges,
    js_divergence,
    ks_pvalue,
    psi,
    score_signal,
)

RNG = np.random.default_rng(100)


def test_psi_is_near_zero_for_two_samples_of_the_same_distribution():
    a = RNG.standard_normal(5000)
    b = np.random.default_rng(101).standard_normal(5000)
    assert psi(a, b, bins=10, strategy="quantile") < 0.05


def test_psi_grows_when_the_production_mean_moves():
    a = RNG.standard_normal(5000)
    b = np.random.default_rng(102).standard_normal(5000) + 2.0
    assert psi(a, b, bins=10, strategy="quantile") > 1.0


def test_quantile_and_equal_width_give_different_bin_edges():
    reference = np.concatenate([RNG.standard_normal(4900), np.full(100, 50.0)])
    q = bin_edges(reference, 10, "quantile")
    e = bin_edges(reference, 10, "equal_width")
    # The outlier stretches equal-width bins far wider than quantile bins.
    assert e[-2] > q[-2]


def test_an_unknown_binning_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown binning strategy"):
        bin_edges(RNG.standard_normal(100), 10, "octiles")


def test_bin_edges_handle_a_near_constant_signal():
    constant = np.full(1000, 3.0)
    edges = bin_edges(constant, 10, "quantile")
    assert edges.shape[0] >= 2
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_bin_edges_span_to_infinity_so_production_tails_are_counted():
    edges = bin_edges(RNG.standard_normal(1000), 8, "equal_width")
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_ks_pvalue_is_small_for_a_clear_shift():
    a = RNG.standard_normal(3000)
    b = np.random.default_rng(103).standard_normal(3000) + 1.5
    assert ks_pvalue(a, b) < 1e-6


def test_ks_pvalue_is_not_small_for_the_same_distribution():
    a = RNG.standard_normal(3000)
    b = np.random.default_rng(104).standard_normal(3000)
    assert ks_pvalue(a, b) > 0.01


def test_js_divergence_is_zero_for_identical_samples():
    a = RNG.standard_normal(2000)
    assert js_divergence(a, a, bins=10, strategy="quantile") < 1e-9


def test_js_divergence_is_bounded_by_one_bit():
    a = RNG.standard_normal(2000)
    b = np.random.default_rng(105).standard_normal(2000) + 6.0
    value = js_divergence(a, b, bins=10, strategy="quantile")
    assert 0.0 <= value <= 1.0


def test_js_divergence_is_zero_for_empty_samples():
    # An empty reference yields a single all-capturing bin, so there is nothing to
    # diverge: the result is a well-defined zero rather than a crash.
    empty = np.array([])
    assert js_divergence(empty, empty, bins=4, strategy="equal_width") == 0.0


def test_psi_is_zero_for_empty_samples():
    empty = np.array([])
    assert psi(empty, empty, bins=4, strategy="quantile") == 0.0


def test_score_signal_fires_every_detector_on_a_large_shift():
    ref = RNG.standard_normal(4000)
    prod = np.random.default_rng(106).standard_normal(4000) + 3.0
    result = score_signal(
        "x", ref, prod,
        psi_alarm=0.2, psi_bins=10, psi_binning="quantile", ks_alpha=0.05, js_alarm=0.1,
    )
    assert result.psi_fired and result.ks_fired and result.js_fired
    assert result.any_fired


def test_score_signal_stays_quiet_on_an_unchanged_signal():
    ref = RNG.standard_normal(4000)
    prod = np.random.default_rng(107).standard_normal(4000)
    result = score_signal(
        "x", ref, prod,
        psi_alarm=0.2, psi_bins=10, psi_binning="quantile", ks_alpha=0.001, js_alarm=0.1,
    )
    assert not result.psi_fired
    assert not result.js_fired
    assert not result.any_fired
