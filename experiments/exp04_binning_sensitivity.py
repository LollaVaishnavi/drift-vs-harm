"""Experiment 4: does the PSI alarm survive an arbitrary binning choice?

The war story behind ADR-005. One benign variance shift that causes no harm is
scored with the two textbook PSI binnings. Under quantile bins it stays below the
0.2 alarm; under equal-width bins it crosses it. The alarm is an artifact of the
binning, not of anything that mattered. This is why the default freezes quantile
edges on the reference window, and why a regression test pins the flip.
"""

from __future__ import annotations

import numpy as np

from driftharm.drift import psi
from driftharm.generator import draw_window
from driftharm.harness import _scenario_seed, run_scenario
from driftharm.scenarios import war_binning_spec
from harness import baseline, policy, print_table, suite_outcomes, write_json


def compute(_outcomes) -> dict:
    pol = policy()
    base = baseline()
    spec = war_binning_spec()
    rng = np.random.default_rng(_scenario_seed(pol.run.seed, spec.name))
    window = draw_window(rng, spec, pol.run.production_window, base.model.score)
    feature = spec.mean_shift.argmax() if spec.scale.max() == 1.0 else int(np.argmax(spec.scale))
    reference = base.signal_window.features[:, feature]
    production = window.features[:, feature]
    quantile = psi(reference, production, pol.drift.psi_bins, "quantile")
    equal_width = psi(reference, production, pol.drift.psi_bins, "equal_width")
    outcome = run_scenario(pol, base, spec)
    return {
        "psi_alarm_threshold": pol.drift.psi_alarm,
        "war_psi_quantile": round(quantile, 4),
        "war_psi_equal_width": round(equal_width, 4),
        "war_quantile_alarms": bool(quantile >= pol.drift.psi_alarm),
        "war_equal_width_alarms": bool(equal_width >= pol.drift.psi_alarm),
        "war_harmed": bool(outcome.harmed),
        "war_auc_drop": round(outcome.harm.auc_drop, 4),
    }


def main() -> None:
    result = compute(suite_outcomes())
    rows = [
        ("quantile (default)", f"{result['war_psi_quantile']:.3f}",
         "ALARM" if result["war_quantile_alarms"] else "quiet"),
        ("equal_width", f"{result['war_psi_equal_width']:.3f}",
         "ALARM" if result["war_equal_width_alarms"] else "quiet"),
    ]
    print_table(
        f"Experiment 4: one benign shift, two binnings, threshold {result['psi_alarm_threshold']}",
        rows,
        ("binning", "psi", "verdict"),
    )
    print()
    print(f"The scenario is harmless (AUC drop {result['war_auc_drop']:+.3f}), "
          f"so the equal-width alarm is a pure artifact of the binning choice.")
    write_json("exp04_binning_sensitivity", result)


if __name__ == "__main__":
    main()
