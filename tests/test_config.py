"""Policy loading is strict: an unchosen threshold is the thing this project warns about."""

from __future__ import annotations

import textwrap

import pytest

from driftharm.config import DEFAULT_POLICY_PATH, load_policy


def test_the_bundled_policy_loads_with_all_three_sections():
    policy = load_policy()
    assert policy.drift.psi_alarm == 0.2
    assert policy.harm.auc_drop_alarm == 0.02
    assert policy.run.seed == 20260211


def test_an_explicit_path_overrides_the_bundled_default():
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert policy.drift.psi_bins == 10


def test_a_missing_policy_file_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="policy file not found"):
        load_policy(tmp_path / "nope.toml")


def _write(tmp_path, body):
    path = tmp_path / "policy.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_a_missing_key_names_its_section(tmp_path):
    path = _write(
        tmp_path,
        """
        [drift]
        psi_bins = 10
        psi_binning = "quantile"
        ks_alpha = 0.05
        js_alarm = 0.1
        watch = ["features"]
        [harm]
        auc_drop_alarm = 0.02
        label_sample = 10
        [run]
        seed = 1
        reference_window = 10
        production_window = 10
        """,
    )
    with pytest.raises(ValueError, match=r"\[drift\] is missing key 'psi_alarm'"):
        load_policy(path)


def test_an_unknown_binning_strategy_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [drift]
        psi_alarm = 0.2
        psi_bins = 10
        psi_binning = "octiles"
        ks_alpha = 0.05
        js_alarm = 0.1
        watch = ["features"]
        [harm]
        auc_drop_alarm = 0.02
        label_sample = 10
        [run]
        seed = 1
        reference_window = 10
        production_window = 10
        """,
    )
    with pytest.raises(ValueError, match="psi_binning must be one of"):
        load_policy(path)


def test_an_empty_watch_list_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [drift]
        psi_alarm = 0.2
        psi_bins = 10
        psi_binning = "quantile"
        ks_alpha = 0.05
        js_alarm = 0.1
        watch = []
        [harm]
        auc_drop_alarm = 0.02
        label_sample = 10
        [run]
        seed = 1
        reference_window = 10
        production_window = 10
        """,
    )
    with pytest.raises(ValueError, match="watch must list at least one"):
        load_policy(path)


def test_an_unknown_watch_signal_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [drift]
        psi_alarm = 0.2
        psi_bins = 10
        psi_binning = "quantile"
        ks_alpha = 0.05
        js_alarm = 0.1
        watch = ["gradients"]
        [harm]
        auc_drop_alarm = 0.02
        label_sample = 10
        [run]
        seed = 1
        reference_window = 10
        production_window = 10
        """,
    )
    with pytest.raises(ValueError, match="watch entries must be in"):
        load_policy(path)


def test_fewer_than_two_bins_is_refused(tmp_path):
    path = _write(
        tmp_path,
        """
        [drift]
        psi_alarm = 0.2
        psi_bins = 1
        psi_binning = "quantile"
        ks_alpha = 0.05
        js_alarm = 0.1
        watch = ["features"]
        [harm]
        auc_drop_alarm = 0.02
        label_sample = 10
        [run]
        seed = 1
        reference_window = 10
        production_window = 10
        """,
    )
    with pytest.raises(ValueError, match="psi_bins must be at least 2"):
        load_policy(path)
