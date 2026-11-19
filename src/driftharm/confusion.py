"""Treat the drift alarm as a classifier of harm, and score it honestly.

A monitoring policy that alarms on drift is, implicitly, predicting harm. So the
fair way to judge it is a confusion matrix: over a scenario set, how often does a
drift alarm coincide with real harm (precision), and how often does harm occur with
no alarm (the miss, which is recall's complement). The two numbers that matter are
recall on the concept family (structurally near zero) and precision at the realised
harm base rate.

Every zero here is priced, not celebrated. A measured recall of zero over k harmed
scenarios is "no higher than one in k could show", and the module reports that
resolution floor next to the estimate, because a bare zero would approve every claim
in the repository. That discipline is the point of receipts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scipy import stats

from driftharm.harness import ScenarioOutcome

# A predictor maps a scenario outcome to "did this monitor fire?". The default is
# the generous aggregate; experiments pass a per-detector predictor to show that
# PSI and JS give a clean structural zero while the aggregate is muddied by KS.
Predictor = Callable[[ScenarioOutcome], bool]


def generous_alarm(outcome: ScenarioOutcome) -> bool:
    return outcome.drift_alarm


def psi_only(outcome: ScenarioOutcome) -> bool:
    return outcome.psi_alarm


def js_only(outcome: ScenarioOutcome) -> bool:
    return outcome.js_alarm


def ks_only(outcome: ScenarioOutcome) -> bool:
    return outcome.ks_alarm


@dataclass(frozen=True)
class Confusion:
    """Drift alarm (predictor) against harm (truth), over measurable scenarios."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    not_measurable: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def harmed(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def alarms(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float | None:
        """P(harm | alarm). None when nothing alarmed."""
        if self.alarms == 0:
            return None
        return self.true_positive / self.alarms

    @property
    def recall(self) -> float | None:
        """P(alarm | harm). None when nothing was harmed."""
        if self.harmed == 0:
            return None
        return self.true_positive / self.harmed

    @property
    def harm_prevalence(self) -> float | None:
        if self.total == 0:
            return None
        return self.harmed / self.total


def confusion_matrix(
    outcomes: list[ScenarioOutcome], predictor: Predictor = generous_alarm
) -> Confusion:
    """Build the 2x2 of a monitor's alarm against harm, over measurable scenarios.

    ``predictor`` selects which alarm to judge: the generous aggregate by default,
    or a single detector. Scenarios whose harm could not be measured (a degenerate
    one-class window) are counted separately and kept out of the rates, so a "could
    not answer" never inflates or deflates precision.
    """
    tp = fp = fn = tn = nm = 0
    for outcome in outcomes:
        if not outcome.measurable:
            nm += 1
            continue
        fired = predictor(outcome)
        if outcome.harmed and fired:
            tp += 1
        elif outcome.harmed and not fired:
            fn += 1
        elif not outcome.harmed and fired:
            fp += 1
        else:
            tn += 1
    return Confusion(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        not_measurable=nm,
    )


def resolution_floor(sample_size: int) -> float | None:
    """One over the sample size: the smallest rate a sample of this size can resolve."""
    if sample_size <= 0:
        return None
    return 1.0 / sample_size


def recall_on_family(
    outcomes: list[ScenarioOutcome], family: str, predictor: Predictor = generous_alarm
) -> Confusion:
    """Confusion restricted to one ground-truth family (for example, concept)."""
    return confusion_matrix([o for o in outcomes if o.kind == family], predictor)


def false_positive_rate(
    outcomes: list[ScenarioOutcome], predictor: Predictor = generous_alarm
) -> float | None:
    """Alarm rate on scenarios that carried no harm: the detector's own noise floor.

    The structural punchline lives here: a detector blind to labels cannot tell a
    concept-drift window from a no-change window, so its recall on concept drift is
    bounded by this rate. If the two are equal, the detector's "detections" of harm
    are indistinguishable from its false alarms.
    """
    unharmed = [o for o in outcomes if o.measurable and not o.harmed]
    if not unharmed:
        return None
    fired = sum(1 for o in unharmed if predictor(o))
    return fired / len(unharmed)


def silent_failures(outcomes: list[ScenarioOutcome]) -> list[ScenarioOutcome]:
    """Scenarios where harm was real and no detector fired (generous aggregate)."""
    return [o for o in outcomes if o.silent_failure]


def psi_harm_correlation(outcomes: list[ScenarioOutcome]) -> float | None:
    """Spearman correlation between drift magnitude (max PSI) and realised AUC drop.

    Computed over measurable scenarios. A value near zero is the magnitude-decoupling
    result: a bigger drift reading does not mean a bigger loss. None when there are
    too few points or no variance to correlate.
    """
    pairs = [
        (o.max_psi, o.harm.auc_drop)
        for o in outcomes
        if o.measurable and o.harm.auc_drop is not None
    ]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    result = stats.spearmanr(xs, ys)
    return float(result.statistic)
