"""The deployed model is frozen and deterministic; these tests pin both."""

from __future__ import annotations

import numpy as np
import pytest

from driftharm.generator import N_FEATURES
from driftharm.model import train_reference_model


def test_the_model_scores_are_probabilities(rng):
    model = train_reference_model(rng, n=2000)
    features = np.random.default_rng(1).standard_normal((100, N_FEATURES))
    scores = model.score(features)
    assert scores.shape == (100,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_training_is_deterministic_for_a_fixed_seed():
    a = train_reference_model(np.random.default_rng(3), n=2000)
    b = train_reference_model(np.random.default_rng(3), n=2000)
    probe = np.random.default_rng(9).standard_normal((50, N_FEATURES))
    assert np.allclose(a.score(probe), b.score(probe))


def test_scoring_rejects_the_wrong_number_of_features(rng):
    model = train_reference_model(rng, n=1000)
    with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
        model.score(np.zeros((10, 3)))


def test_scoring_rejects_a_one_dimensional_input(rng):
    model = train_reference_model(rng, n=1000)
    with pytest.raises(ValueError, match="expected features"):
        model.score(np.zeros(6))


def test_the_model_ranks_better_than_chance_on_its_own_world(rng):
    from sklearn.metrics import roc_auc_score

    from driftharm.generator import TRUE_BIAS, TRUE_WEIGHTS, sigmoid

    model = train_reference_model(rng, n=4000)
    probe = np.random.default_rng(2).standard_normal((4000, N_FEATURES))
    labels = (
        np.random.default_rng(4).random(4000) < sigmoid(probe @ TRUE_WEIGHTS + TRUE_BIAS)
    ).astype(int)
    assert roc_auc_score(labels, model.score(probe)) > 0.8
