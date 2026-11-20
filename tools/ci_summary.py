"""Write the awkward table to the CI job summary.

A human reading the Actions UI should see the uncomfortable numbers, not have to dig
them out of a log. Reads docs/metrics.json (already re-measured by collect_metrics in
the receipts step) and prints a markdown table to stdout, which the workflow appends
to $GITHUB_STEP_SUMMARY.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    m = json.loads((ROOT / "docs" / "metrics.json").read_text())["metrics"]
    harmed = m["concept_harmed"]
    worst = m["worst_silent_auc_drop"]
    silent = f"{m['silent_harm_count']} of {m['harmed_scenarios']} harmed, invisible to drift"
    binning = f"{m['war_psi_quantile']:.2f} vs {m['war_psi_equal_width']:.2f}"
    threshold = m["psi_alarm_threshold"]
    binning_note = f"same benign shift, quantile vs equal-width, threshold {threshold}"
    rows = [
        ("PSI recall on concept drift", f"{m['concept_psi_recall_pct']:.0f}%",
         f"0 of {harmed} harmed, worst AUC drop {worst:.2f}"),
        ("Drift alarm precision", f"{m['drift_precision_pct']:.0f}%",
         f"at harm prevalence {m['harm_prevalence_pct']:.0f}%"),
        ("Drift vs harm correlation", f"{m['spearman_across_mix']:.2f}",
         "Spearman across the mix (negative: more drift, less harm)"),
        ("Covariate false alarms", f"{m['covariate_psi_alarm_pct']:.0f}%",
         "PSI fires, harm absent"),
        ("Silent share of harm", f"{m['silent_harm_share_pct']:.0f}%", silent),
        ("Binning artifact", binning, binning_note),
    ]
    print("### drift-vs-harm: the numbers that matter\n")
    print("| finding | value | note |")
    print("|---|---|---|")
    for name, value, note in rows:
        print(f"| {name} | {value} | {note} |")
    print(f"\n{m['tests_total']} tests, {m['coverage_line_pct']:.0f}% line coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
