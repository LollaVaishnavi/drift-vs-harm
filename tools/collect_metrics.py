"""Re-measure every published figure and write docs/metrics.json.

Nothing here is typed by hand. The test count and line coverage are read from the
machine-readable reports pytest already wrote (junit.xml, coverage.json), not parsed
from a progress line. Every other figure is produced by running the seeded suite and
the five experiments in this process. The output pairs each figure with the anchor
phrases a document must use for it, so check_numbers.py can fail the build when a
document drifts from the code.

Run `--skip-tests` in CI after the test step has already produced the reports; run it
without that flag locally to produce them first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from driftharm.config import load_policy  # noqa: E402
from driftharm.harness import build_baseline, run_suite  # noqa: E402
from driftharm.scenarios import default_suite  # noqa: E402

REPORTS = ROOT / "reports"
METRICS_PATH = ROOT / "docs" / "metrics.json"

CHECKED_DOCUMENTS = [
    "README.md",
    "docs/adr/ADR-001-synthetic-generator-over-real-data.md",
    "docs/adr/ADR-002-watch-many-signals-generously.md",
    "docs/adr/ADR-003-harm-kept-robust-not-the-subject.md",
    "docs/adr/ADR-004-alarm-as-a-classifier-of-harm.md",
    "docs/adr/ADR-005-freeze-quantile-bin-edges.md",
]


def _run_tests() -> None:
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests",
            "--junitxml", str(REPORTS / "junit.xml"),
            "--cov=driftharm", "--cov-report", f"json:{REPORTS / 'coverage.json'}",
            "-q",
        ],
        cwd=ROOT,
        check=True,
    )


def _tests_total() -> int:
    tree = ET.parse(REPORTS / "junit.xml")
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    return int(suite.attrib["tests"])


def _coverage_pct() -> float:
    data = json.loads((REPORTS / "coverage.json").read_text())
    return round(float(data["totals"]["percent_covered"]), 2)


def _experiment_metrics(outcomes) -> dict:
    """Import the five experiments and run their compute functions on one suite."""
    import exp01_silent_failure
    import exp02_false_alarm
    import exp03_magnitude_decoupling
    import exp04_binning_sensitivity
    import exp05_label_delay_window

    merged: dict = {}
    merged.update(exp01_silent_failure.compute(outcomes))
    merged.update(exp02_false_alarm.compute(outcomes))
    merged.update(exp03_magnitude_decoupling.compute(outcomes))
    merged.update(exp04_binning_sensitivity.compute(outcomes))
    merged.update(exp05_label_delay_window.compute(outcomes))
    return merged


def build_payload(skip_tests: bool) -> dict:
    if not skip_tests:
        _run_tests()

    policy = load_policy()
    baseline = build_baseline(policy)
    outcomes = run_suite(policy, default_suite(), baseline=baseline)
    exp = _experiment_metrics(outcomes)

    # The curated set of figures that appear in prose, each with raw value and the
    # anchor phrases (template rendered with the value) a document must use.
    metrics = {
        "tests_total": _tests_total(),
        "coverage_line_pct": _coverage_pct(),
        "scenarios_total": len(outcomes),
        "concept_harmed": exp["concept_harmed"],
        "concept_psi_recall_pct": exp["concept_psi_recall_pct"],
        "concept_js_recall_pct": exp["concept_js_recall_pct"],
        "worst_silent_auc_drop": exp["worst_silent_auc_drop"],
        "min_silent_auc_drop": exp["min_silent_auc_drop"],
        "covariate_scenarios": exp["covariate_scenarios"],
        "covariate_psi_alarm_pct": exp["covariate_psi_alarm_pct"],
        "drift_precision_pct": exp["drift_precision_pct"],
        "harm_prevalence_pct": exp["harm_prevalence_pct"],
        "null_psi_alarm_pct": exp["null_psi_alarm_pct"],
        "null_ks_alarm_pct": exp["null_ks_alarm_pct"],
        "spearman_across_mix": exp["spearman_across_mix"],
        "psi_alarm_threshold": exp["psi_alarm_threshold"],
        "war_psi_quantile": exp["war_psi_quantile"],
        "war_psi_equal_width": exp["war_psi_equal_width"],
        "silent_harm_count": exp["psi_silent_harm"],
        "harmed_scenarios": exp["harmed_scenarios"],
        "silent_harm_share_pct": exp["silent_harm_share_pct"],
    }

    anchors = {
        "tests_total": ["{} tests"],
        "coverage_line_pct": ["{:.0f}% line coverage", "{:.0f}% coverage"],
        "scenarios_total": ["{} labelled scenarios", "{} scenarios"],
        "concept_harmed": ["0 of {}", "{} concept-drift scenarios", "one in {}"],
        "concept_psi_recall_pct": ["recall of {:.0f}%", "fires on {:.0f} of", "{:.0f}% recall"],
        "concept_js_recall_pct": ["JS on {:.0f}", "{:.0f}% for JS"],
        "worst_silent_auc_drop": ["AUC falls {:.2f}", "falls by {:.2f}", "up to {:.2f}"],
        "min_silent_auc_drop": ["{:.2f} to", "from {:.2f}"],
        "covariate_scenarios": ["{} covariate", "all {} covariate"],
        "covariate_psi_alarm_pct": ["{:.0f}% of covariate", "on {:.0f}% of the covariate"],
        "drift_precision_pct": ["precision of {:.0f}%", "right {:.0f}% of", "{:.0f}% of the time"],
        "harm_prevalence_pct": ["prevalence of {:.0f}%", "{:.0f}% of the suite"],
        "null_psi_alarm_pct": ["{:.0f}% of null", "null windows {:.0f}%"],
        "null_ks_alarm_pct": ["KS on {:.0f}%", "{:.0f}% of null windows"],
        "spearman_across_mix": ["Spearman of {:.2f}", "correlation of {:.2f}"],
        "psi_alarm_threshold": ["threshold of {}", "{} alarm level", "above {}"],
        "war_psi_quantile": ["{:.2f} under quantile", "quantile reads {:.2f}"],
        "war_psi_equal_width": ["{:.2f} under equal-width", "equal-width reads {:.2f}"],
        "silent_harm_count": ["{} of the 23", "{} of 23"],
        "harmed_scenarios": ["{} harmed scenarios", "of {} harmed"],
        "silent_harm_share_pct": ["{:.0f}% of realised harm", "nearly {:.0f}%"],
    }

    rendered = {
        key: [template.format(metrics[key]) for template in templates]
        for key, templates in anchors.items()
    }

    return {
        "metrics": metrics,
        "anchors": anchors,
        "rendered_anchors": rendered,
        "checked_documents": CHECKED_DOCUMENTS,
        "note": "every value here is produced by running the suite, the experiments and the tests",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="measure every published figure")
    parser.add_argument("--skip-tests", action="store_true", help="read existing test reports")
    args = parser.parse_args(argv)

    payload = build_payload(args.skip_tests)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)} with {len(payload['metrics'])} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
