"""Experiment 2: does a drift alarm mean harm?

Question the README asks: when PSI fires, did the model actually get worse? The
covariate family moves the inputs while a well-fit model keeps its ranking, so a
fired alarm there is a false alarm about harm. This experiment reports the precision
of a drift alarm as a harm predictor at the suite's real base rate, and separates
the clean PSI alarms on covariate shift from the noisy KS alarms on pure null.
"""

from __future__ import annotations

from driftharm.confusion import confusion_matrix, generous_alarm
from harness import pct, print_table, suite_outcomes, write_json


def _bonferroni_null_rate(outcomes) -> float | None:
    """Null alarm rate if KS were Bonferroni-corrected across the watched signals.

    Shows that correcting the threshold cuts the false alarms but, as experiment 1
    already proved, does nothing for the misses: a tighter threshold cannot give a
    detector an input it never had.
    """
    null = [o for o in outcomes if o.kind == "null"]
    if not null:
        return None
    n_signals = len(null[0].signals)
    corrected_alpha = 0.05 / n_signals
    fired = sum(1 for o in null if o.min_ks_p < corrected_alpha)
    return fired / len(null)


def compute(outcomes) -> dict:
    whole = confusion_matrix(outcomes, generous_alarm)
    covariate = [o for o in outcomes if o.kind == "covariate"]
    null = [o for o in outcomes if o.kind == "null"]
    cov_psi_alarms = sum(1 for o in covariate if o.psi_alarm)
    cov_harmed = sum(1 for o in covariate if o.harmed)
    null_psi_alarms = sum(1 for o in null if o.psi_alarm)
    null_ks_alarms = sum(1 for o in null if o.ks_alarm)
    return {
        "harm_prevalence_pct": pct(whole.harm_prevalence),
        "drift_precision_pct": pct(whole.precision),
        "covariate_scenarios": len(covariate),
        "covariate_psi_alarm_pct": pct(cov_psi_alarms / len(covariate)) if covariate else None,
        "covariate_harm_pct": pct(cov_harmed / len(covariate)) if covariate else None,
        "null_psi_alarm_pct": pct(null_psi_alarms / len(null)) if null else None,
        "null_ks_alarm_pct": pct(null_ks_alarms / len(null)) if null else None,
        "null_ks_bonferroni_pct": pct(_bonferroni_null_rate(outcomes)),
    }


def main() -> None:
    outcomes = suite_outcomes()
    result = compute(outcomes)
    rows = [
        (o.name, f"{o.max_psi:.2f}", "yes" if o.psi_alarm else "no",
         f"{o.harm.auc_drop:+.3f}", "yes" if o.harmed else "no")
        for o in outcomes if o.kind == "covariate"
    ]
    print_table(
        "Experiment 2: covariate shift is a false alarm",
        rows,
        ("scenario", "max_psi", "psi_alarm", "auc_drop", "harmed"),
    )
    print()
    print(f"Covariate PSI alarm rate: {result['covariate_psi_alarm_pct']}% "
          f"with harm rate {result['covariate_harm_pct']}%")
    print(f"A drift alarm's precision as a harm predictor: {result['drift_precision_pct']}% "
          f"at harm prevalence {result['harm_prevalence_pct']}%")
    print(f"Null PSI alarm rate {result['null_psi_alarm_pct']}%, "
          f"null KS alarm rate {result['null_ks_alarm_pct']}%, "
          f"KS with Bonferroni {result['null_ks_bonferroni_pct']}%")
    write_json("exp02_false_alarm", result)


if __name__ == "__main__":
    main()
