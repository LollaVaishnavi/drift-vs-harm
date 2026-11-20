"""Experiment 3: does a bigger drift reading mean a bigger loss?

Question the README asks: if I rank my production windows by PSI, do I rank them by
harm? This measures the Spearman correlation between the drift magnitude (max PSI)
and the realised AUC drop across the whole mix, and again within the harmed
scenarios only. The across-mix value is negative precisely because of the decoupling:
the loudest drift (covariate) is harmless and the real harm (concept) is silent.
"""

from __future__ import annotations

from driftharm.confusion import psi_harm_correlation
from harness import print_table, suite_outcomes, write_json


def compute(outcomes) -> dict:
    measurable = [o for o in outcomes if o.measurable and o.harm.auc_drop is not None]
    harmed = [o for o in measurable if o.harmed]
    across = psi_harm_correlation(measurable)
    within_harmed = psi_harm_correlation(harmed)
    concept = [o for o in outcomes if o.kind == "concept"]
    concept_psi = [o.max_psi for o in concept]
    concept_drops = [o.harm.auc_drop for o in concept]
    return {
        "spearman_across_mix": round(across, 3) if across is not None else None,
        "spearman_within_harmed": round(within_harmed, 3) if within_harmed is not None else None,
        "concept_psi_spread": round(max(concept_psi) - min(concept_psi), 4),
        "concept_auc_drop_spread": round(max(concept_drops) - min(concept_drops), 4),
    }


def main() -> None:
    outcomes = suite_outcomes()
    result = compute(outcomes)
    rows = []
    for kind in ("null", "covariate", "concept", "mixed"):
        fam = [o for o in outcomes if o.kind == kind and o.measurable]
        psis = [o.max_psi for o in fam]
        drops = [o.harm.auc_drop for o in fam if o.harm.auc_drop is not None]
        rows.append((
            kind,
            f"{min(psis):.2f}-{max(psis):.2f}",
            f"{min(drops):+.3f}..{max(drops):+.3f}",
            sum(o.harmed for o in fam),
        ))
    print_table(
        "Experiment 3: drift magnitude does not track harm magnitude",
        rows,
        ("family", "psi_range", "auc_drop_range", "harmed"),
    )
    print()
    print(f"Spearman(max PSI, AUC drop) across the mix: {result['spearman_across_mix']}")
    print(f"Spearman within harmed scenarios only:      {result['spearman_within_harmed']}")
    print(f"Concept family: PSI spans only {result['concept_psi_spread']} "
          f"while AUC drop spans {result['concept_auc_drop_spread']}")
    write_json("exp03_magnitude_decoupling", result)


if __name__ == "__main__":
    main()
