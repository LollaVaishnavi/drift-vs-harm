"""The deployed model: fit once on reference data, then frozen.

This file exists to give every scenario a single, fixed scorer, so that any change
in ranking quality is caused by the scenario and not by a retrain. The model is an
ordinary logistic regression. It is deliberately unremarkable: the repository is
not about the model, it is about what a monitor can and cannot learn from watching
the model's inputs and outputs without labels.

The model is a deterministic function of its inputs once fit. That determinism is
load-bearing for the concept-drift argument in generator.py: equal inputs give equal
scores, so a concept-only shift is invisible in the score distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from driftharm.generator import N_FEATURES, TRUE_BIAS, TRUE_WEIGHTS, sigmoid


@dataclass(frozen=True)
class DeployedModel:
    """A frozen scorer. ``score`` returns P(Y=1|X) for each row."""

    estimator: LogisticRegression

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(
                f"expected features with shape (n, {N_FEATURES}), got {features.shape}"
            )
        return self.estimator.predict_proba(features)[:, 1]


def train_reference_model(rng: np.random.Generator, n: int = 8000) -> DeployedModel:
    """Fit the deployed model on a sample from the reference concept.

    lbfgs on a fixed sample is deterministic, so the returned scorer is identical
    across runs given the same seed. The sample is drawn from the unshifted world:
    the model never sees any production scenario, exactly as a real deployed model
    is frozen before the shifts it will later face.
    """
    features = rng.standard_normal((n, N_FEATURES))
    probs = sigmoid(features @ TRUE_WEIGHTS + TRUE_BIAS)
    labels = (rng.random(n) < probs).astype(int)
    estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
    estimator.fit(features, labels)
    return DeployedModel(estimator=estimator)
