"""Experiment 1: does a drift detector catch concept drift?

Question the README asks: when the labelling rule P(Y|X) changes but the inputs do
not, does a population-drift alarm fire? The concept family is built exactly this
way, and every scenario in it carries a real AUC drop. If PSI fired on any of them,
the repository's central claim would be false.
"""

from __future__ import annotations

from driftharm.confusion import generous_alarm, js_only, psi_only, recall_on_family
from harness import pct, print_table, suite_outcomes, write_json


def compute(outcomes) -> dict:
    concept = [o for o in outcomes if o.kind == "concept"]
    drops = [o.harm.auc_drop for o in concept]
    psi_cm = recall_on_family(outcomes, "concept", psi_only)
    js_cm = recall_on_family(outcomes, "concept", js_only)
    any_cm = recall_on_family(outcomes, "concept", generous_alarm)
    harmed = psi_cm.harmed
    return {
        "concept_scenarios": len(concept),
        "concept_harmed": harmed,
        "concept_psi_recall_pct": pct(psi_cm.recall),
        "concept_js_recall_pct": pct(js_cm.recall),
        "concept_generous_recall_pct": pct(any_cm.recall),
        "worst_silent_auc_drop": round(max(drops), 4),
        "min_silent_auc_drop": round(min(drops), 4),
        "resolution_floor_pct": round(100.0 / harmed, 2) if harmed else None,
    }


def main() -> None:
    outcomes = suite_outcomes()
    result = compute(outcomes)
    rows = [
        (o.name, f"{o.harm.auc_drop:+.3f}", f"{o.max_psi:.3f}",
         "yes" if o.psi_alarm else "no", "yes" if o.harmed else "no")
        for o in outcomes if o.kind == "concept"
    ]
    print_table(
        "Experiment 1: concept drift is a silent failure",
        rows,
        ("scenario", "auc_drop", "max_psi", "psi_alarm", "harmed"),
    )
    print()
    print(f"PSI recall on concept drift: {result['concept_psi_recall_pct']}% "
          f"(0 of {result['concept_harmed']}, floor {result['resolution_floor_pct']}%)")
    print(f"JS recall on concept drift:  {result['concept_js_recall_pct']}%")
    print(f"Worst silent AUC drop:       {result['worst_silent_auc_drop']}")
    write_json("exp01_silent_failure", result)


if __name__ == "__main__":
    main()
