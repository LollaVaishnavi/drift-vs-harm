"""The world: a seeded data generator where the true cause of a shift is known.

This file is the reason the whole repository can make its claim at all. On a real
production log you cannot label a shift as covariate or concept, because you never
observe the counterfactual, so the central question ("did the model actually get
worse, and would drift have told me?") is unfalsifiable. Here the generator owns
the ground truth: it decides whether a production window differs from reference by
its inputs P(X), by its labelling rule P(Y|X), by both, or by neither. ADR-001
records why this is synthetic and what that costs.

The concept-drift construction is the delicate one. A concept-drift window is
drawn from the SAME input distribution as reference (identity covariate transform),
and only the labelling weights move. Because the deployed model is a deterministic
function of X, its scores on that window are distributed exactly as on reference up
to sampling noise. That is the point: the detector sees an unchanged world while the
labels, which it never reads, have moved underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The true labelling rule of the reference world. Features 0..3 carry signal with
# decreasing weight; features 4 and 5 are pure noise the ideal model would ignore.
# Fixed as data, never fit, so a scenario's cause is unambiguous.
TRUE_WEIGHTS = np.array([1.4, -1.1, 0.9, 0.6, 0.0, 0.0])
TRUE_BIAS = 0.2
N_FEATURES = TRUE_WEIGHTS.shape[0]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass(frozen=True)
class ScenarioSpec:
    """A named shift from the reference world.

    kind is the ground-truth cause. ``mean_shift`` and ``scale`` act on the input
    distribution P(X); ``concept_delta`` is added to the labelling weights to move
    P(Y|X). A null scenario leaves all three at their identity. These vectors are
    what the generator knows and the detector does not.
    """

    name: str
    kind: str  # one of: null, covariate, concept, mixed
    mean_shift: np.ndarray
    scale: np.ndarray
    concept_delta: np.ndarray

    def production_weights(self) -> np.ndarray:
        return TRUE_WEIGHTS + self.concept_delta


def null_spec(name: str = "null") -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        kind="null",
        mean_shift=np.zeros(N_FEATURES),
        scale=np.ones(N_FEATURES),
        concept_delta=np.zeros(N_FEATURES),
    )


def _draw_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Reference input distribution: independent standard normals."""
    return rng.standard_normal((n, N_FEATURES))


def _apply_covariate(base: np.ndarray, spec: ScenarioSpec) -> np.ndarray:
    """Transform reference inputs into a scenario's input distribution."""
    return base * spec.scale + spec.mean_shift


def _draw_labels(rng: np.random.Generator, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    probs = sigmoid(features @ weights + TRUE_BIAS)
    return (rng.random(features.shape[0]) < probs).astype(int)


@dataclass(frozen=True)
class LabelledWindow:
    """A window of inputs with their true labels and the model's scores."""

    features: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def draw_window(
    rng: np.random.Generator,
    spec: ScenarioSpec,
    n: int,
    score_fn,
) -> LabelledWindow:
    """Draw one window under a scenario.

    ``score_fn`` maps features to the deployed model's probabilities. Labels follow
    the scenario's (possibly shifted) concept weights; scores follow the frozen
    model. Harm is the gap between the two, and the detector gets only features and
    scores, never labels.
    """
    base = _draw_features(rng, n)
    features = _apply_covariate(base, spec)
    labels = _draw_labels(rng, features, spec.production_weights())
    scores = score_fn(features)
    return LabelledWindow(features=features, labels=labels, scores=scores)
