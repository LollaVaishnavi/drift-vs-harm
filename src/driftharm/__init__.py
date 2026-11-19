"""drift-vs-harm: does a population-drift alarm actually predict model harm?

The package exists to answer one question that a monitoring dashboard cannot:
when a drift detector fires (or stays quiet), was the model's quality actually
worse? A drift statistic is a function of the input distribution P(X) and of the
model's own score f(X). Harm is a function of the joint with the label, P(Y|X).
Nothing in the detector's inputs carries a label, so the detector is structurally
blind to a change in P(Y|X). This library makes that blindness measurable by
replaying scenarios whose true cause is known.

Public surface is intentionally small; the modules below each own one idea.
"""

from driftharm.config import Policy, load_policy

__all__ = ["Policy", "load_policy"]
__version__ = "0.1.0"
