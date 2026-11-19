"""The generator owns the ground truth; these tests pin what each shift does to it."""

from __future__ import annotations

import numpy as np

from driftharm.generator import (
    N_FEATURES,
    TRUE_WEIGHTS,
    ScenarioSpec,
    draw_window,
    null_spec,
    sigmoid,
)
from driftharm.scenarios import concept_spec, covariate_spec


def test_sigmoid_is_stable_for_large_positive_and_negative_inputs():
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.isclose(values[0], 0.0)
    assert np.isclose(values[1], 0.5)
    assert np.isclose(values[2], 1.0)
    assert np.all(np.isfinite(values))


def test_sigmoid_is_monotonic():
    xs = np.linspace(-6, 6, 50)
    ys = sigmoid(xs)
    assert np.all(np.diff(ys) > 0)


def test_null_spec_leaves_inputs_and_concept_at_identity():
    spec = null_spec()
    assert np.all(spec.mean_shift == 0)
    assert np.all(spec.scale == 1)
    assert np.all(spec.concept_delta == 0)
    assert np.all(spec.production_weights() == TRUE_WEIGHTS)


def test_concept_delta_changes_only_the_labelling_weights():
    spec = concept_spec("c", feature=1, delta=2.0)
    weights = spec.production_weights()
    assert weights[1] == TRUE_WEIGHTS[1] + 2.0
    assert np.all(spec.mean_shift == 0)
    assert np.all(spec.scale == 1)


def test_a_concept_window_keeps_the_input_distribution_of_reference():
    # Same seed, so the base draw is identical; a concept shift must not touch X.
    identity = lambda x: np.zeros(x.shape[0])  # noqa: E731 - trivial score stub
    ref = draw_window(np.random.default_rng(7), null_spec(), 2000, identity)
    con = draw_window(
        np.random.default_rng(7), concept_spec("c", feature=0, delta=-2.0), 2000, identity
    )
    assert np.allclose(ref.features, con.features)
    # The labels, which the detector never sees, are what moved.
    assert not np.array_equal(ref.labels, con.labels)


def test_a_covariate_window_moves_the_requested_feature(rng):
    identity = lambda x: np.zeros(x.shape[0])  # noqa: E731
    spec = covariate_spec("cov", feature=2, shift=3.0)
    window = draw_window(rng, spec, 5000, identity)
    assert window.features[:, 2].mean() > 2.5
    # A feature that was not shifted stays centred near zero.
    assert abs(window.features[:, 3].mean()) < 0.2


def test_scale_widens_the_requested_feature(rng):
    identity = lambda x: np.zeros(x.shape[0])  # noqa: E731
    spec = covariate_spec("cov", feature=0, shift=0.0, scale=3.0)
    window = draw_window(rng, spec, 5000, identity)
    assert window.features[:, 0].std() > 2.5


def test_draw_window_returns_labels_and_scores_of_matching_length(rng):
    score_fn = lambda x: np.full(x.shape[0], 0.5)  # noqa: E731
    window = draw_window(rng, null_spec(), 128, score_fn)
    assert window.features.shape == (128, N_FEATURES)
    assert window.labels.shape == (128,)
    assert window.scores.shape == (128,)
    assert set(np.unique(window.labels)).issubset({0, 1})


def test_scenario_spec_is_frozen():
    spec = null_spec()
    assert isinstance(spec, ScenarioSpec)
    try:
        spec.name = "x"
    except Exception as exc:  # noqa: BLE001 - we only assert it is refused
        assert "cannot assign" in str(exc) or "frozen" in str(exc)
    else:  # pragma: no cover - a frozen dataclass must raise
        raise AssertionError("ScenarioSpec should be immutable")
