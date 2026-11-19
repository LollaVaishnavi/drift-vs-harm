"""The exit codes are the interface; these pin each one and the output formats."""

from __future__ import annotations

import io
import json

import pytest

from driftharm.cli import (
    EXIT_NOT_MEASURABLE,
    EXIT_SILENT_FAILURE,
    EXIT_TRACKED,
    audit_exit_code,
    main,
)


class _Outcome:
    def __init__(self, measurable, silent):
        self.measurable = measurable
        self.silent_failure = silent


def test_exit_code_is_not_measurable_when_nothing_can_be_judged():
    assert audit_exit_code([_Outcome(False, False)]) == EXIT_NOT_MEASURABLE
    assert audit_exit_code([]) == EXIT_NOT_MEASURABLE


def test_exit_code_is_silent_failure_when_harm_went_unalarmed():
    outcomes = [_Outcome(True, True), _Outcome(True, False)]
    assert audit_exit_code(outcomes) == EXIT_SILENT_FAILURE


def test_exit_code_is_tracked_when_every_harm_alarmed():
    outcomes = [_Outcome(True, False), _Outcome(True, False)]
    assert audit_exit_code(outcomes) == EXIT_TRACKED


def test_version_prints_and_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_no_command_prints_help(capsys):
    code = main([])
    out = capsys.readouterr().out
    assert code == EXIT_TRACKED
    assert "usage" in out.lower()


def test_audit_on_the_full_suite_reports_a_silent_failure():
    stream = io.StringIO()
    code = main(["audit"], stream=stream)
    assert code == EXIT_SILENT_FAILURE
    assert "PSI recall on concept drift" in stream.getvalue()


def test_audit_on_the_covariate_family_tracks_cleanly():
    # Covariate scenarios carry no harm, so there is nothing to miss: TRACKED.
    stream = io.StringIO()
    code = main(["audit", "--family", "covariate"], stream=stream)
    assert code == EXIT_TRACKED


def test_audit_json_format_is_machine_readable():
    stream = io.StringIO()
    code = main(["audit", "--family", "concept", "--format", "json"], stream=stream)
    payload = json.loads(stream.getvalue())
    assert code == EXIT_SILENT_FAILURE
    assert payload["summary"]["concept_psi_recall"] == 0.0


def test_audit_accepts_an_explicit_policy_path():
    from driftharm.config import DEFAULT_POLICY_PATH

    stream = io.StringIO()
    code = main(["audit", "--policy", str(DEFAULT_POLICY_PATH), "--format", "json"], stream=stream)
    assert code in (EXIT_SILENT_FAILURE, EXIT_TRACKED)
