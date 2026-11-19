"""The scenario families are the experiment's design; these pin their shape."""

from __future__ import annotations

import numpy as np

from driftharm.scenarios import (
    concept_spec,
    covariate_spec,
    default_suite,
    mixed_spec,
    suite_by_family,
)


def test_covariate_spec_shifts_one_feature_and_leaves_the_concept_alone():
    spec = covariate_spec("c", feature=2, shift=1.5, scale=2.0)
    assert spec.kind == "covariate"
    assert spec.mean_shift[2] == 1.5
    assert spec.scale[2] == 2.0
    assert np.all(spec.concept_delta == 0)


def test_concept_spec_moves_one_weight_and_leaves_inputs_alone():
    spec = concept_spec("c", feature=1, delta=-1.0)
    assert spec.kind == "concept"
    assert spec.concept_delta[1] == -1.0
    assert np.all(spec.mean_shift == 0)
    assert np.all(spec.scale == 1)


def test_mixed_spec_moves_both_an_input_and_a_weight():
    spec = mixed_spec("m", cov_feature=0, shift=1.0, concept_feature=3, delta=2.0)
    assert spec.kind == "mixed"
    assert spec.mean_shift[0] == 1.0
    assert spec.concept_delta[3] == 2.0


def test_the_default_suite_has_all_four_families():
    families = suite_by_family(default_suite())
    assert set(families) == {"null", "covariate", "concept", "mixed"}


def test_the_default_suite_family_sizes_are_fixed():
    families = suite_by_family(default_suite())
    assert len(families["null"]) == 10
    assert len(families["covariate"]) == 10
    assert len(families["concept"]) == 15
    assert len(families["mixed"]) == 8


def test_every_scenario_name_is_unique():
    names = [spec.name for spec in default_suite()]
    assert len(names) == len(set(names))
