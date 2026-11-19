"""Render a suite run as a table or as JSON.

Kept apart from the CLI so the same rendering is reachable from experiments and
tests without going through argument parsing. The table is written to be legible in
a terminal and in the replay demo; the JSON is what the receipts tooling and the CI
summary consume.
"""

from __future__ import annotations

import json

from driftharm.confusion import (
    confusion_matrix,
    false_positive_rate,
    generous_alarm,
    psi_harm_correlation,
    psi_only,
    recall_on_family,
)
from driftharm.harness import ScenarioOutcome


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def summarise(outcomes: list[ScenarioOutcome]) -> dict:
    """The numbers the whole project turns on, as a plain dict."""
    whole = confusion_matrix(outcomes, generous_alarm)
    concept_psi = recall_on_family(outcomes, "concept", psi_only)
    concept_any = recall_on_family(outcomes, "concept", generous_alarm)
    null = [o for o in outcomes if o.kind == "null"]
    null_psi_alarm_rate = (
        sum(1 for o in null if o.psi_alarm) / len(null) if null else None
    )
    concept_drops = [
        o.harm.auc_drop
        for o in outcomes
        if o.kind == "concept" and o.harm.auc_drop is not None
    ]
    return {
        "scenarios": len(outcomes),
        "harm_prevalence": whole.harm_prevalence,
        "generous_precision": whole.precision,
        "generous_recall": whole.recall,
        "concept_harmed": concept_psi.harmed,
        "concept_psi_recall": concept_psi.recall,
        "concept_any_recall": concept_any.recall,
        "null_psi_alarm_rate": null_psi_alarm_rate,
        "worst_silent_auc_drop": max(concept_drops) if concept_drops else None,
        "psi_false_positive_rate": false_positive_rate(outcomes, psi_only),
        "psi_harm_spearman": psi_harm_correlation(outcomes),
        "not_measurable": whole.not_measurable,
    }


def render_table(outcomes: list[ScenarioOutcome]) -> str:
    """A fixed-width table of every scenario, grouped visually by family."""
    lines = []
    header = (
        f"{'scenario':20} {'family':10} {'psi':>6} {'ks_p':>6} "
        f"{'alarm':>5} {'aucdrop':>8} {'harm':>5} {'silent':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for o in outcomes:
        alarm = "PSI" if o.psi_alarm else ("ks" if o.drift_alarm else ".")
        lines.append(
            f"{o.name:20} {o.kind:10} {o.max_psi:6.2f} {o.min_ks_p:6.3f} "
            f"{alarm:>5} {_fmt(o.harm.auc_drop, '+.3f'):>8} "
            f"{('yes' if o.harmed else 'no'):>5} {('YES' if o.silent_failure else ''):>6}"
        )
    return "\n".join(lines)


def render_summary(outcomes: list[ScenarioOutcome]) -> str:
    """The paragraph a human reads after the table."""
    s = summarise(outcomes)
    silent = sum(1 for o in outcomes if o.silent_failure)
    return (
        f"\nScenarios: {s['scenarios']}  harm prevalence: {_fmt(s['harm_prevalence'])}\n"
        f"Concept-drift scenarios harmed: {s['concept_harmed']} "
        f"(AUC drop up to {_fmt(s['worst_silent_auc_drop'])})\n"
        f"PSI recall on concept drift: {_fmt(s['concept_psi_recall'])} "
        f"(resolution floor 1/{s['concept_harmed']})\n"
        f"PSI alarm rate on pure-null windows: {_fmt(s['null_psi_alarm_rate'])}\n"
        f"Generous-aggregate precision as a harm predictor: {_fmt(s['generous_precision'])}\n"
        f"Spearman(PSI, AUC drop) across the mix: {_fmt(s['psi_harm_spearman'])}\n"
        f"Silent failures (harm, no alarm): {silent}\n"
    )


def render_json(outcomes: list[ScenarioOutcome]) -> str:
    payload = {
        "summary": summarise(outcomes),
        "scenarios": [
            {
                "name": o.name,
                "kind": o.kind,
                "psi_alarm": o.psi_alarm,
                "drift_alarm": o.drift_alarm,
                "max_psi": o.max_psi,
                "min_ks_p": o.min_ks_p,
                "auc_drop": o.harm.auc_drop,
                "harmed": o.harmed,
                "measurable": o.measurable,
                "silent_failure": o.silent_failure,
            }
            for o in outcomes
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
