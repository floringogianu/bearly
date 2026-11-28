"""Tests for bearly.get_interval_estimates adherence to rliable semantics."""

import numpy as np
import pandas as pd
import pytest

import bearly
from bearly.core import _generate_stratified_bootstrap

from .reference import rliable_ref as rr


def one_group_df(scores_matrix: np.ndarray, group_value: str = "G") -> pd.DataFrame:
    """(num_runs, num_tasks) matrix -> melted frame with a single group."""
    rows = []
    for r in range(scores_matrix.shape[0]):
        for t in range(scores_matrix.shape[1]):
            rows.append((group_value, f"task_{t}", r, scores_matrix[r, t]))
    return pd.DataFrame(rows, columns=["agent", "task", "run", "score"])


def test_stratified_bootstrap_resamples_per_task_independently():
    """Runs are resampled with replacement *within* each task."""
    rng = np.random.default_rng(0)
    data = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
    sample = _generate_stratified_bootstrap(data, rng)
    assert len(sample) == 2
    for original, resampled in zip(data, sample):
        assert len(resampled) == len(original)
        assert set(resampled).issubset(set(original))


def test_point_estimates_match_rliable():
    rng = np.random.default_rng(1)
    matrix = rng.uniform(0, 5, size=(8, 6))
    df = one_group_df(matrix)

    specs = [
        ("iqm", bearly.iqm, rr.rliable_iqm),
        ("mean", bearly.mean, rr.rliable_mean),
        ("median", bearly.median, rr.rliable_median),
        ("optimality_gap", bearly.optimality_gap, rr.rliable_optimality_gap),
    ]
    for name, b_fn, ref_fn in specs:
        out = bearly.get_interval_estimates(
            df, b_fn, "score", "task", ["agent"], n_samples=50
        )
        row = out.loc[out["agent"] == "G"].iloc[0]
        assert row["y"] == pytest.approx(ref_fn(matrix)), name
        assert row["stat_fn"] == name  # bare function -> __name__


def test_ci_contains_point_estimate_and_is_orderable():
    df = one_group_df(np.random.default_rng(2).uniform(0, 1, size=(5, 4)))
    out = bearly.get_interval_estimates(
        df, bearly.mean, "score", "task", ["agent"], n_samples=200
    )
    row = out.iloc[0]
    assert row["ymin"] <= row["y"] <= row["ymax"]


def test_tuple_name_is_used():
    df = one_group_df(np.random.default_rng(3).uniform(0, 1, size=(4, 3)))
    out = bearly.get_interval_estimates(
        df, ("custom name", bearly.mean), "score", "task", ["agent"], n_samples=50
    )
    assert out.loc[0, "stat_fn"] == "custom name"


class TestGrouping:
    def test_one_row_per_group_and_stat_runs_per_group_independently(self, bench_df):
        out = bearly.get_interval_estimates(
            bench_df,
            bearly.mean,
            "score",
            "task",
            ["agent", "step"],
            n_samples=50,
        )
        assert set(out.columns) == {"agent", "step", "y", "ymin", "ymax", "stat_fn"}
        assert len(out) == bench_df.groupby(["agent", "step"]).ngroups

        # point estimate equality per group, computed by hand
        grouped = bench_df.groupby(["agent", "step"])
        for keys, sub in grouped:
            sub_by_task = sub.groupby("task")["score"].mean()
            expected = sub_by_task.mean()
            actual = out.query("agent == @keys[0] and step == @keys[1]")["y"].iloc[0]
            assert actual == pytest.approx(expected)

    def test_runs_are_not_required_to_be_constant_across_tasks(self, bench_df):
        """Uneven run counts per task must not affect mean/median results."""
        df = bench_df.drop(index=[4])  # drop (A, t2, s0, r1): now uneven
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=50
        )
        sub = df[df["agent"] == "A"]
        expected = np.mean(sub.groupby("task")["score"].mean().to_numpy())
        assert out.loc[0, "y"] == pytest.approx(expected)

    def test_deterministic_against_seeded_rliable_bootstrap(self):
        """bearly CIs must fall inside rliable's seeded 99% interval: slack
        accounts for different RNG streams; they can never be wildly off."""
        rng = np.random.default_rng(11)
        matrix = rng.uniform(0, 4, size=(7, 5))
        df = one_group_df(matrix)

        _, (lo99, hi99) = rr.rliable_boot_ci(
            matrix, rr.rliable_mean, reps=2_000, confidence=0.99, seed=5
        )
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=2_000
        )
        row = out.iloc[0]
        assert lo99 - 0.05 <= row["ymin"] <= row["ymax"] <= hi99 + 0.05

class TestNullHandling:
    def test_nan_rows_are_dropped_before_grouping(self, nan_df):
        out = bearly.get_interval_estimates(
            nan_df, bearly.mean, "score", "task", ["agent"], n_samples=50
        )
        # (A, s0): t1 lost run 2 (NaN), t2 intact -> mean(t means)
        # (A, s1): t1 intact, t2 row had NaN task -> only t1 remains
        a0 = out.query("agent == 'A'")
        assert not a0["y"].isna().any()

    def test_all_nan_metric_yields_empty_result_or_nan(self):
        df = pd.DataFrame(
            {
                "agent": ["A", "A"],
                "task": ["t1", "t2"],
                "run": [0, 0],
                "score": [np.nan, np.nan],
            }
        )
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=10
        )
        assert out.empty

    def test_group_cols_with_none_create_nan_group(self):
        """ntp-style: rows with no algorithm label form their own group."""
        df = pd.DataFrame(
            {
                "agent": ["A", "A", None, None],
                "task": ["t1", "t2", "t1", "t2"],
                "score": [1.0, 2.0, 10.0, 20.0],
            }
        )
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=10
        )
        assert len(out) == 2
        assert out["agent"].isna().sum() == 1
        ungroup = out[out["agent"].isna()].iloc[0]
        assert ungroup["y"] == pytest.approx(15.0)

    def test_nan_group_col_does_not_meet_named_groups(self):
        df = pd.DataFrame(
            {
                "agent": ["A", "A", None, "A"],
                "task": ["t1", "t2", "t1", "t2"],
                "score": [1.0, 2.0, 5.0, 3.0],
            }
        )
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=10
        )
        a_row = out[out["agent"] == "A"].iloc[0]
        # A rows: (t1, 1), (t2, 2), (t2, 3); the None-group's 5 must not leak in
        assert a_row["y"] == pytest.approx(np.mean([1.0, 2.5]))


class TestDegenerateGeometry:
    def test_single_task(self):
        df = one_group_df(np.random.default_rng(0).uniform(size=(5, 1)))
        out = bearly.get_interval_estimates(
            df, bearly.mean, "score", "task", ["agent"], n_samples=20
        )
        assert len(out) == 1
        assert out["y"].iloc[0] == pytest.approx(df["score"].mean())

    def test_single_run_per_task(self, bench_df):
        """One run per task per group: the stratified bootstrap has nothing to
        resample, so every CI collapses to the point estimate."""
        df = bench_df.groupby(["agent", "task"]).head(1)  # keep step 0 only
        out = bearly.get_interval_estimates(
            df, bearly.median, "score", "task", ["agent"], n_samples=50
        )
        assert not out.empty

        a = out[out["agent"] == "A"].iloc[0]
        assert a["ymin"] == a["y"] == a["ymax"]

        b = out[out["agent"] == "B"].iloc[0]
        assert b["ymin"] == b["y"] == b["ymax"] == pytest.approx(4.0)

def test_output_dtypes():
    df = one_group_df(np.random.default_rng(0).uniform(size=(4, 3)))
    out = bearly.get_interval_estimates(
        df, bearly.mean, "score", "task", ["agent"], n_samples=10
    )
    assert pd.api.types.is_float_dtype(out["y"])
    assert pd.api.types.is_float_dtype(out["ymin"])
    assert all(isinstance(s, str) for s in out["stat_fn"])
