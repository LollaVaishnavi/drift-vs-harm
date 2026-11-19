"""Harm: the model getting worse, measurable only where labels exist.

This file is the half of the world the detector never sees. Harm is a change in the
model's ranking quality (AUC) between a reference holdout and a production window,
corroborated by log loss and calibration error. It is kept deliberately robust:
large labelled samples, a modest threshold, a threshold-free primary metric. That is
a choice, recorded in ADR-003, and it is what keeps the repository's headline about
blindness rather than about noise. A sister project (rag-regression-gate) is about
the case where the metric itself is too noisy to trust; this one assumes harm is
cleanly measurable once you have the labels, and asks whether a label-blind monitor
can stand in for that measurement. It cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Mean gap between confidence and accuracy across equal-width score bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.shape[0]
    if total == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # The final bin includes its right edge so a score of exactly 1.0 is counted.
        upper = (scores <= hi) if hi == 1.0 else (scores < hi)
        in_bin = (scores >= lo) & upper
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        confidence = float(np.mean(scores[in_bin]))
        accuracy = float(np.mean(labels[in_bin]))
        ece += (count / total) * abs(confidence - accuracy)
    return ece


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUC, or None when it is undefined because a window holds one class only."""
    if np.unique(labels).shape[0] < 2:
        return None
    return float(roc_auc_score(labels, scores))


@dataclass(frozen=True)
class HarmResult:
    """The labelled-truth verdict for one production window."""

    reference_auc: float | None
    production_auc: float | None
    auc_drop: float | None
    reference_logloss: float
    production_logloss: float
    reference_ece: float
    production_ece: float
    measurable: bool
    harmed: bool


def measure_harm(
    reference_labels: np.ndarray,
    reference_scores: np.ndarray,
    production_labels: np.ndarray,
    production_scores: np.ndarray,
    *,
    auc_drop_alarm: float,
) -> HarmResult:
    """Compare production ranking quality to the reference holdout.

    When AUC is undefined on either side (a degenerate one-class window) the result
    is marked not measurable, and ``harmed`` is False rather than a guess: the honest
    answer there is that labels could not settle it. The CLI maps that to a distinct
    exit code so "could not answer" never reads as "answer is no".
    """
    reference_auc = _safe_auc(reference_labels, reference_scores)
    production_auc = _safe_auc(production_labels, production_scores)

    reference_logloss = float(log_loss(reference_labels, reference_scores, labels=[0, 1]))
    production_logloss = float(log_loss(production_labels, production_scores, labels=[0, 1]))
    reference_ece = expected_calibration_error(reference_labels, reference_scores)
    production_ece = expected_calibration_error(production_labels, production_scores)

    if reference_auc is None or production_auc is None:
        return HarmResult(
            reference_auc=reference_auc,
            production_auc=production_auc,
            auc_drop=None,
            reference_logloss=reference_logloss,
            production_logloss=production_logloss,
            reference_ece=reference_ece,
            production_ece=production_ece,
            measurable=False,
            harmed=False,
        )

    auc_drop = reference_auc - production_auc
    return HarmResult(
        reference_auc=reference_auc,
        production_auc=production_auc,
        auc_drop=auc_drop,
        reference_logloss=reference_logloss,
        production_logloss=production_logloss,
        reference_ece=reference_ece,
        production_ece=production_ece,
        measurable=True,
        harmed=auc_drop >= auc_drop_alarm,
    )
