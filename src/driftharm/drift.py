"""The label-blind detectors: PSI, KS, Jensen-Shannon.

These are the standard monitoring statistics, implemented plainly because the
repository's claim is about what they structurally cannot do, not about any
weakness in a particular implementation. Each is a function of a reference sample
and a production sample of one signal (a feature column, or the model's score).
None of them takes a label, which is the entire point: whatever threshold you set,
the function has no argument through which a change in P(Y|X) could enter.

Binning is where PSI hides an arbitrary choice. Both strategies are implemented so
the sensitivity is measurable rather than asserted; ADR-005 records the war story
that made quantile the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

_EPS = 1e-6


def bin_edges(reference: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Edges frozen on the reference sample.

    quantile gives each reference bin roughly equal mass; equal_width splits the
    reference range into equal intervals and is therefore hostage to the reference
    tails. Edges always come from reference, never from production, so the baseline
    is fixed before any production data arrives, as a stored monitoring baseline is.
    """
    if reference.size == 0:
        # Nothing to bin against: one all-capturing bin, so downstream histograms of
        # any production sample are well defined and the divergence is zero.
        return np.array([-np.inf, np.inf])
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(reference, quantiles)
    elif strategy == "equal_width":
        lo = float(np.min(reference))
        hi = float(np.max(reference))
        edges = np.linspace(lo, hi, bins + 1)
    else:
        raise ValueError(f"unknown binning strategy: {strategy!r}")
    edges = np.asarray(edges, dtype=float)
    # Collapse duplicate edges (a near-constant signal yields repeated quantiles)
    # and widen the outer edges so production values beyond the reference range are
    # captured rather than silently dropped.
    edges = np.unique(edges)
    if edges.shape[0] < 2:
        # Degenerate signal: one value. Make a trivial two-edge bin around it.
        centre = edges[0] if edges.shape[0] else 0.0
        edges = np.array([centre - 0.5, centre + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(sample: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(sample, bins=edges)
    proportions = counts.astype(float) / max(sample.shape[0], 1)
    return np.clip(proportions, _EPS, None)


def psi(reference: np.ndarray, production: np.ndarray, bins: int, strategy: str) -> float:
    """Population Stability Index of production against reference.

    PSI = sum over bins of (p_prod - p_ref) * ln(p_prod / p_ref). Symmetric-ish and
    always non-negative. Values near zero mean the production histogram matches the
    reference histogram; the common rule of thumb calls 0.1 minor and 0.2 significant.
    """
    edges = bin_edges(reference, bins, strategy)
    p_ref = _proportions(reference, edges)
    p_prod = _proportions(production, edges)
    return float(np.sum((p_prod - p_ref) * np.log(p_prod / p_ref)))


def ks_pvalue(reference: np.ndarray, production: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov p-value. Small means distinguishable."""
    result = stats.ks_2samp(reference, production)
    return float(result.pvalue)


def js_divergence(reference: np.ndarray, production: np.ndarray, bins: int, strategy: str) -> float:
    """Jensen-Shannon divergence in bits (0 to 1), from histograms on reference edges."""
    edges = bin_edges(reference, bins, strategy)
    p_ref = _proportions(reference, edges)
    p_prod = _proportions(production, edges)
    distance = jensenshannon(p_ref, p_prod, base=2.0)
    return float(distance**2)


@dataclass(frozen=True)
class SignalDrift:
    """Every detector's reading on one signal, plus whether each fired."""

    signal: str
    psi: float
    ks_p: float
    js: float
    psi_fired: bool
    ks_fired: bool
    js_fired: bool

    @property
    def any_fired(self) -> bool:
        return self.psi_fired or self.ks_fired or self.js_fired


def score_signal(
    signal: str,
    reference: np.ndarray,
    production: np.ndarray,
    *,
    psi_alarm: float,
    psi_bins: int,
    psi_binning: str,
    ks_alpha: float,
    js_alarm: float,
) -> SignalDrift:
    """Run all three detectors on one signal and apply the thresholds."""
    psi_value = psi(reference, production, psi_bins, psi_binning)
    ks_p = ks_pvalue(reference, production)
    js_value = js_divergence(reference, production, psi_bins, psi_binning)
    return SignalDrift(
        signal=signal,
        psi=psi_value,
        ks_p=ks_p,
        js=js_value,
        psi_fired=psi_value >= psi_alarm,
        ks_fired=ks_p < ks_alpha,
        js_fired=js_value >= js_alarm,
    )
