"""Rendering is shared by the CLI, experiments and tests; these pin its output."""

from __future__ import annotations

import json

from driftharm.report import (
    _fmt,
    render_json,
    render_summary,
    render_table,
    summarise,
)


def test_fmt_renders_none_as_na():
    assert _fmt(None) == "n/a"
    assert _fmt(0.5) == "0.500"


def test_summary_reports_the_headline_fields(suite_outcomes):
    summary = summarise(suite_outcomes)
    for key in (
        "concept_psi_recall",
        "null_psi_alarm_rate",
        "worst_silent_auc_drop",
        "psi_harm_spearman",
        "harm_prevalence",
    ):
        assert key in summary


def test_the_table_has_one_row_per_scenario_plus_a_header(suite_outcomes):
    table = render_table(suite_outcomes)
    lines = table.splitlines()
    # header + rule + one line per scenario
    assert len(lines) == len(suite_outcomes) + 2
    assert "scenario" in lines[0]


def test_the_summary_text_names_the_structural_zero(suite_outcomes):
    text = render_summary(suite_outcomes)
    assert "PSI recall on concept drift" in text
    assert "resolution floor" in text


def test_the_json_round_trips_and_lists_every_scenario(suite_outcomes):
    payload = json.loads(render_json(suite_outcomes))
    assert len(payload["scenarios"]) == len(suite_outcomes)
    assert "summary" in payload
    assert payload["summary"]["concept_psi_recall"] == 0.0
