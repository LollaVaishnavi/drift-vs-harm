"""Shared fixtures.

The baseline (model plus reference windows) and the default-suite run are built once
per session and reused, because each is deterministic and rebuilding them per test
would add nothing but seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftharm.config import load_policy
from driftharm.harness import build_baseline, run_suite
from driftharm.scenarios import default_suite


@pytest.fixture(scope="session")
def policy():
    return load_policy()


@pytest.fixture(scope="session")
def baseline(policy):
    return build_baseline(policy)


@pytest.fixture(scope="session")
def suite_outcomes(policy, baseline):
    return run_suite(policy, default_suite(), baseline=baseline)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
