"""The four shift families, parameterised by magnitude.

Each family isolates one ground-truth cause so the confusion between drift and harm
can be read off cleanly:

- null:       P(X) and P(Y|X) both unchanged. Neither should fire.
- covariate:  P(X) moves, P(Y|X) fixed. Drift should fire; a well-fit model keeps
              its ranking, so harm should be absent. The false-alarm class.
- concept:    P(X) fixed, P(Y|X) moves. Drift cannot see it; harm is real. The
              silent-failure class, and the heart of the repository.
- mixed:      both move, at varied ratios. Used to show drift magnitude does not
              track harm magnitude.

Magnitudes are chosen so each family spans a range rather than a single point; the
exact numbers a reviewer cares about are measured by the experiments, not asserted
here.
"""

from __future__ import annotations

import numpy as np

from driftharm.generator import N_FEATURES, ScenarioSpec, null_spec


def _zeros() -> np.ndarray:
    return np.zeros(N_FEATURES)


def _ones() -> np.ndarray:
    return np.ones(N_FEATURES)


def covariate_spec(name: str, feature: int, shift: float, scale: float = 1.0) -> ScenarioSpec:
    """Shift one feature's mean (and optionally its spread), concept untouched."""
    mean = _zeros()
    mean[feature] = shift
    scales = _ones()
    scales[feature] = scale
    return ScenarioSpec(
        name=name,
        kind="covariate",
        mean_shift=mean,
        scale=scales,
        concept_delta=_zeros(),
    )


def concept_spec(name: str, feature: int, delta: float) -> ScenarioSpec:
    """Move one labelling weight, inputs untouched (identity covariate transform)."""
    concept = _zeros()
    concept[feature] = delta
    return ScenarioSpec(
        name=name,
        kind="concept",
        mean_shift=_zeros(),
        scale=_ones(),
        concept_delta=concept,
    )


def mixed_spec(
    name: str, cov_feature: int, shift: float, concept_feature: int, delta: float
) -> ScenarioSpec:
    """A covariate shift and a concept shift at the same time."""
    mean = _zeros()
    mean[cov_feature] = shift
    concept = _zeros()
    concept[concept_feature] = delta
    return ScenarioSpec(
        name=name,
        kind="mixed",
        mean_shift=mean,
        scale=_ones(),
        concept_delta=concept,
    )


def default_suite() -> list[ScenarioSpec]:
    """The standard scenario set the CLI and most experiments run.

    Sizes of each family are balanced enough that the confusion matrix has a usable
    resolution floor (one over the family size), which the receipts price every zero
    against.
    """
    specs: list[ScenarioSpec] = []

    # Null (10): independent draws of the unchanged world. These estimate the
    # detectors' own false-positive rate, which is the ceiling on their recall for
    # concept drift, because concept drift is indistinguishable from null in the
    # space a label-blind detector observes.
    specs += [null_spec(f"null-{i:02d}") for i in range(10)]

    # Covariate (10): P(X) moves, the labelling rule does not. Shifts land on the
    # noise features the model ignores and on informative features where a well-fit
    # logistic model keeps its ranking, so drift fires and harm stays absent. The
    # false-alarm class.
    specs += [
        covariate_spec("cov-noise4-1.5", feature=4, shift=1.5),
        covariate_spec("cov-noise4-2.5", feature=4, shift=2.5),
        covariate_spec("cov-noise5-2.0", feature=5, shift=2.0),
        covariate_spec("cov-noise5-var", feature=5, shift=0.0, scale=2.0),
        covariate_spec("cov-info0-1.0", feature=0, shift=1.0),
        covariate_spec("cov-info0-1.5", feature=0, shift=1.5),
        covariate_spec("cov-info2-1.5", feature=2, shift=1.5),
        covariate_spec("cov-info3-2.0", feature=3, shift=2.0),
        covariate_spec("cov-info3-var", feature=3, shift=0.0, scale=1.8),
        covariate_spec("cov-info2-var", feature=2, shift=0.0, scale=1.7),
    ]

    # Concept (15): the labelling weight on one feature moves while inputs do not.
    # Each carries a real AUC drop; none of them moves P(X), so PSI and JS read at
    # their null level by construction. The silent-failure class.
    specs += [
        concept_spec("concept-f0-d-2.5", feature=0, delta=-2.5),
        concept_spec("concept-f0-d-2.0", feature=0, delta=-2.0),
        concept_spec("concept-f0-d-1.5", feature=0, delta=-1.5),
        concept_spec("concept-f0-d-1.0", feature=0, delta=-1.0),
        concept_spec("concept-f1-d+1.5", feature=1, delta=1.5),
        concept_spec("concept-f1-d+2.0", feature=1, delta=2.0),
        concept_spec("concept-f1-d+2.5", feature=1, delta=2.5),
        concept_spec("concept-f2-d-2.5", feature=2, delta=-2.5),
        concept_spec("concept-f2-d-1.5", feature=2, delta=-1.5),
        concept_spec("concept-f3-d-2.5", feature=3, delta=-2.5),
        concept_spec("concept-f3-d-2.0", feature=3, delta=-2.0),
        concept_spec("concept-f4-d-2.5", feature=4, delta=-2.5),
        concept_spec("concept-f4-d+2.5", feature=4, delta=2.5),
        concept_spec("concept-f5-d-2.5", feature=5, delta=-2.5),
        concept_spec("concept-f5-d+2.0", feature=5, delta=2.0),
    ]

    # Mixed (8): a visible covariate shift and a concept shift at the same time, at
    # varied ratios. Drift fires on the covariate part while the concept part drives
    # the harm, so the alarm is right for the wrong reason. Used to show drift
    # magnitude does not track harm magnitude.
    specs += [
        mixed_spec("mix-small-cov", cov_feature=4, shift=0.6, concept_feature=1, delta=2.0),
        mixed_spec("mix-big-cov", cov_feature=4, shift=2.5, concept_feature=0, delta=-0.8),
        mixed_spec("mix-even-0-2", cov_feature=0, shift=1.0, concept_feature=2, delta=-1.5),
        mixed_spec("mix-cov-heavy", cov_feature=0, shift=2.0, concept_feature=3, delta=-1.2),
        mixed_spec("mix-concept-heavy", cov_feature=4, shift=0.4, concept_feature=0, delta=-2.0),
        mixed_spec("mix-balanced-2-1", cov_feature=2, shift=1.2, concept_feature=1, delta=1.6),
        mixed_spec("mix-5-0", cov_feature=5, shift=1.5, concept_feature=0, delta=-1.4),
        mixed_spec("mix-3-2", cov_feature=3, shift=1.8, concept_feature=2, delta=-2.0),
    ]

    return specs


# The war-story scenario (ADR-005): a benign variance increase on feature 0 that
# carries no harm, yet whose PSI crosses the 0.2 alarm under equal-width binning and
# stays quiet under quantile binning. Defined here so the experiment that measures
# the flip and the regression test that pins it use exactly the same shift.
WAR_BINNING_FEATURE = 0
WAR_BINNING_SCALE = 1.4


def war_binning_spec() -> ScenarioSpec:
    return covariate_spec("war-binning-f0-var", feature=WAR_BINNING_FEATURE, shift=0.0,
                          scale=WAR_BINNING_SCALE)


def suite_by_family(specs: list[ScenarioSpec]) -> dict[str, list[ScenarioSpec]]:
    """Group a spec list by its ground-truth cause."""
    families: dict[str, list[ScenarioSpec]] = {}
    for spec in specs:
        families.setdefault(spec.kind, []).append(spec)
    return families
