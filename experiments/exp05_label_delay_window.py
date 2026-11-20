"""Experiment 5: what does drift buy you in the label-blind window?

Harm can only be measured once labels arrive, after some delay. Drift is available
at once. So the fair question is what a drift alarm actually buys you during that
label-blind window. This partitions the realised harm into the part a drift alarm
would have caught (harm that came with a covariate shift) and the part it is blind
to (concept-driven harm), and reports the silent share: the fraction of real harm
that is invisible to a population-drift monitor until the labels come in.
"""

from __future__ import annotations

from harness import pct, print_table, suite_outcomes, write_json


def compute(outcomes) -> dict:
    harmed = [o for o in outcomes if o.measurable and o.harmed]
    psi_visible = [o for o in harmed if o.psi_alarm]
    psi_silent = [o for o in harmed if not o.psi_alarm]
    # The silent harm, broken out by family, so it is clear the blindness is concept.
    silent_by_family: dict[str, int] = {}
    for o in psi_silent:
        silent_by_family[o.kind] = silent_by_family.get(o.kind, 0) + 1
    return {
        "harmed_scenarios": len(harmed),
        "psi_visible_harm": len(psi_visible),
        "psi_silent_harm": len(psi_silent),
        "silent_harm_share_pct": pct(len(psi_silent) / len(harmed)) if harmed else None,
        "silent_harm_by_family": silent_by_family,
    }


def main() -> None:
    outcomes = suite_outcomes()
    result = compute(outcomes)
    rows = [
        (kind,
         sum(1 for o in outcomes if o.kind == kind and o.harmed),
         sum(1 for o in outcomes if o.kind == kind and o.harmed and o.psi_alarm),
         sum(1 for o in outcomes if o.kind == kind and o.harmed and not o.psi_alarm))
        for kind in ("covariate", "concept", "mixed")
    ]
    print_table(
        "Experiment 5: how much realised harm is invisible to drift until labels arrive",
        rows,
        ("family", "harmed", "psi_visible", "psi_silent"),
    )
    print()
    print(f"Silent share of realised harm: {result['silent_harm_share_pct']}% "
          f"({result['psi_silent_harm']} of {result['harmed_scenarios']})")
    print(f"Silent harm by family: {result['silent_harm_by_family']}")
    write_json("exp05_label_delay_window", result)


if __name__ == "__main__":
    main()
