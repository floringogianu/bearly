"""Tests for bearly.get_performance_profiles against rliable rreferences."""

import numpy as np
import pytest

import bearly

from .reference import rliable_ref as rr
from .test_core import one_group_df

TAUS = np.linspace(0.0, 2.0, 9)


def test_point_profiles_match_rliable_run_score_default():
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0, 2, size=(6, 5))
    df = one_group_df(matrix)

    out = bearly.get_performance_profiles(
        df,
        TAUS,
        metric_col="score",
        task_col="task",
        group_cols=["agent"],
        n_samples=50,
    )
    expected = rr.rliable_perf_profile(matrix, TAUS)
    actual = out.sort_values("tau")["y"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_mean_score_deviation_matches_rliable():
    rng = np.random.default_rng(1)
    matrix = rng.uniform(0, 2, size=(5, 4))
    df = one_group_df(matrix)

    out = bearly.get_performance_profiles(
        df,
        TAUS,
        metric_col="score",
        task_col="task",
        group_cols=["agent"],
        deviation="mean_score",
        n_samples=50,
    )
    expected = rr.rliable_avg_perf_profile(matrix, TAUS)
    actual = out.sort_values("tau")["y"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_output_shape_and_dtypes():
    df = one_group_df(np.random.default_rng(2).uniform(0, 2, size=(4, 3)))
    out = bearly.get_performance_profiles(
        df, TAUS, "score", "task", ["agent"], n_samples=10
    )
    assert len(out) == len(TAUS)
    assert list(out.columns) == ["agent", "y", "ymin", "ymax", "stat_fn", "tau"]
    assert out["stat_fn"].iloc[0] == "pp"
    for col in ["y", "ymin", "ymax", "tau"]:
        assert np.issubdtype(out[col].dtype, np.floating), col
    assert np.isin(out["tau"].to_numpy(), TAUS).all()


def test_monotone_decreasing_in_tau():
    df = one_group_df(np.random.default_rng(3).uniform(0, 2, size=(5, 6)))
    out = bearly.get_performance_profiles(
        df, TAUS, "score", "task", ["agent"], n_samples=10
    )
    y = out.sort_values("tau")["y"].to_numpy()
    assert np.all(np.diff(y) <= 1e-12)
    assert y[0] == pytest.approx(1.0)  # tau=0, scores > 0
    assert y[-1] == pytest.approx(0.0)  # tau=2 == max score (strict >)


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(4)
    matrix = rng.uniform(0, 2, size=(6, 6))
    df = one_group_df(matrix)
    out = bearly.get_performance_profiles(
        df, TAUS, "score", "task", ["agent"], n_samples=200
    )
    assert (out["ymin"] <= out["y"] + 1e-12).all()
    assert (out["y"] <= out["ymax"] + 1e-12).all()


def test_grouping_interleaves_taus_per_group(bench_df):
    out = bearly.get_performance_profiles(
        bench_df, TAUS, "score", "task", ["agent"], n_samples=20
    )
    assert set(out["agent"]) == {"A", "B"}
    for agent in ["A", "B"]:
        sub = out[out["agent"] == agent].sort_values("tau")
        np.testing.assert_allclose(sub["tau"].to_numpy(), TAUS, atol=1e-12)


def test_grouping_profiles_are_independent(bench_df):
    """B is constant 4.0; its profile is a step independent of A."""
    out = bearly.get_performance_profiles(
        bench_df,
        np.array([0.0, 1.0, 3.9, 4.0, 4.1]),
        "score",
        "task",
        ["agent"],
        n_samples=10,
    )
    b = out[out["agent"] == "B"].sort_values("tau")["y"].to_numpy()
    # constant 4.0: > 0/1/3.9 true, > 4.0/4.1 false  (strict greater)
    np.testing.assert_allclose(b, [1, 1, 1, 0, 0], atol=1e-12)


def test_tau_values_none_defaults_to_unit_interval():
    df = one_group_df(np.random.default_rng(5).uniform(0, 2, size=(3, 3)))
    out = bearly.get_performance_profiles(
        df, None, "score", "task", ["agent"], n_samples=10
    )
    assert len(out) == 101
    np.testing.assert_allclose(
        out.sort_values("tau")["tau"].to_numpy(), np.linspace(0, 1.0, 101)
    )


def test_invalid_deviation_raises():
    df = one_group_df(np.random.default_rng(6).uniform(0, 1, size=(3, 3)))
    # set_performance_at_tau_stat has no else-branch: returns None
    with pytest.raises((TypeError, AttributeError, ValueError)):
        bearly.get_performance_profiles(
            df, TAUS, "score", "task", ["agent"], deviation="bogus", n_samples=10
        )


def test_nan_metric_rows_are_dropped(nan_df):
    out = bearly.get_performance_profiles(
        nan_df, TAUS, "score", "task", ["agent"], n_samples=20
    )
    assert not out["y"].isna().any()


def test_unsorted_group_keys_match_each_agent(bench_df):
    """Exploded rows must keep their agent label aligned with profile values."""
    out = bearly.get_performance_profiles(
        bench_df, np.array([0.0, 4.0]), "score", "task", ["agent"], n_samples=10
    )
    a = out[out["agent"] == "A"].sort_values("tau")["y"].to_numpy()
    b = out[out["agent"] == "B"].sort_values("tau")["y"].to_numpy()
    # A has scores 1..7 -> both taus pass some; B is 4.0 -> tau=0 yes, tau=4 no
    assert a[0] > a[1]
    np.testing.assert_allclose(b, [1.0, 0.0], atol=1e-12)
