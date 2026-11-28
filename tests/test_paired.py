"""Tests for bearly.get_paired_interval_estimates and PoI integration."""

import numpy as np
import pandas as pd
import pytest

import bearly

from .reference import rliable_ref as rr


def paired_df(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    x_name: str = "base",
    y_name: str = "other",
) -> pd.DataFrame:
    """Build a long frame; columns are tasks, rows are runs (per algorithm)."""
    frames = []
    for name, m in [(x_name, x_matrix), (y_name, y_matrix)]:
        rows = [
            (name, f"task_{t}", r, m[r, t])
            for r in range(m.shape[0])
            for t in range(m.shape[1])
        ]
        frames.append(pd.DataFrame(rows, columns=["agent", "task", "run", "score"]))
    return pd.concat(frames, ignore_index=True)


class TestContract:
    def test_column_layout_and_row_shape(self):
        df = paired_df(np.ones((3, 4)), np.full((5, 4), 2.0))
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=20,
        )
        assert list(out.columns) == ["X", "Y", "y", "ymin", "ymax", "stat_fn"]
        assert len(out) == 1
        assert out.loc[0, "X"] == "base"
        assert out.loc[0, "Y"] == "other"

    def test_stat_name_default_and_tuple(self):
        df = paired_df(np.ones((2, 3)) + 1, np.zeros((2, 3)))
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            ("win-rate", bearly.probability_of_improvement),
            n_samples=20,
        )
        assert out.loc[0, "stat_fn"] == "win-rate"

        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=20,
        )
        assert out.loc[0, "stat_fn"] == "probability_of_improvement"


class TestSemantics:
    def test_point_estimate_matches_rliable_poi(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 2, size=(10, 6))
        y = rng.uniform(0, 2, size=(8, 6))  # different run count
        df = paired_df(x, y)

        # bearly(compared=(base, other)) = P(other > base) == rliable_poi(other, base)
        expected = rr.rliable_poi(y, x)
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=100,
        )
        assert out.loc[0, "y"] == pytest.approx(expected)

    def test_ci_bounds_point_estimate_for_nontrivial_input(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, size=(10, 8))
        y = x + rng.uniform(-0.2, 0.5, size=x.shape)
        df = paired_df(x, y)
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=500,
        )
        row = out.iloc[0]
        assert row["ymin"] <= row["y"] <= row["ymax"]
        assert 0.0 <= row["ymin"] <= 1.0 and 0.0 <= row["ymax"] <= 1.0

    def test_deterministic_strength_relationship(self):
        """Clearly-better Y should estimate near 1; clearly-worse Y near 0."""
        x_small = np.full((5, 5), 0.0)
        y_big = np.full((5, 5), 1.0)
        df = paired_df(x_small, y_big)
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=100,
        )
        assert out.loc[0, "y"] == pytest.approx(1.0)

        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "other", "base"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=100,
        )
        assert out.loc[0, "y"] == pytest.approx(0.0)

    def test_identical_scores_is_half(self):
        df = paired_df(np.full((4, 3), 7.0), np.full((4, 3), 7.0))
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=20,
        )
        assert out.loc[0, "y"] == pytest.approx(0.5)


class TestAlignment:
    def test_partial_task_overlap_aligns_on_intersection(self):
        # X has tasks {t1, t2}, Y has {t2, t3}: only t2 is used.
        rows = []
        for name, tasks, scores in [
            ("X", ["t1", "t2"], [[3.0], [1.0]]),
            ("Y", ["t2", "t3"], [[2.0], [99.0]]),
        ]:
            for task, col in zip(tasks, scores):
                rows.extend([(name, task, r, s) for r, s in enumerate(col)])
        df = pd.DataFrame(rows, columns=["agent", "task", "run", "score"])

        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "X", "Y"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        # aligned on t2: base=1, other=2 -> P(Y>X)=1 (no ties possible)
        assert out.loc[0, "y"] == pytest.approx(1.0)

    def test_no_common_tasks_raises(self):
        df = pd.DataFrame(
            {
                "agent": ["X", "Y"],
                "task": ["t1", "t2"],
                "run": [0, 0],
                "score": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="No common tasks"):
            bearly.get_paired_interval_estimates(
                df,
                ("agent", "X", "Y"),
                "task",
                "score",
                bearly.probability_of_improvement,
                n_samples=10,
            )

    def test_row_ordering_does_not_change_point_estimate(self):
        """Pairing is label-based; row order must not move the estimate."""
        rng = np.random.default_rng(4)
        x = rng.uniform(size=(6, 4))
        y = rng.uniform(size=(6, 4))
        df1 = paired_df(x, y)
        out1 = bearly.get_paired_interval_estimates(
            df1,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        df2 = df1.sample(frac=1.0, random_state=9).reset_index(drop=True)
        out2 = bearly.get_paired_interval_estimates(
            df2,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        assert out1.loc[0, "y"] == pytest.approx(out2.loc[0, "y"])

    def test_task_pairing_is_label_based_not_positional(self):
        """Rotating X's labels changes pairing and therefore the estimate."""
        rng = np.random.default_rng(4)
        x = rng.uniform(size=(6, 4))
        y = rng.uniform(size=(6, 4))
        df1 = paired_df(x, y)  # X column i -> task_i
        df2 = paired_df(np.roll(x, 1, axis=1), y)  # X column i -> task_{i-1}
        out1 = bearly.get_paired_interval_estimates(
            df1,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        out2 = bearly.get_paired_interval_estimates(
            df2,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        assert out1.loc[0, "y"] != pytest.approx(out2.loc[0, "y"])
        assert 0.0 <= out1.loc[0, "y"] <= 1.0 and 0.0 <= out2.loc[0, "y"] <= 1.0


class TestNullHandling:
    def test_nan_in_metric_does_not_silently_return_a_number(self):
        """rliable-compatible: NaN in MWU -> per-task NaN -> NaN overall."""
        rng = np.random.default_rng(2)
        x = rng.uniform(size=(5, 3))
        y = rng.uniform(size=(5, 3))
        y[0, 0] = np.nan
        df = paired_df(x, y)
        out = bearly.get_paired_interval_estimates(
            df,
            ("agent", "base", "other"),
            "task",
            "score",
            bearly.probability_of_improvement,
            n_samples=10,
        )
        assert np.isnan(out.loc[0, "y"])

    def test_empty_task_in_one_side_warns_and_skips(self):
        """probability_of_improvement's zero-run branch only triggers at the
        function level (a task label with no rows cannot survive grouping)."""
        with pytest.warns(RuntimeWarning, match="0 runs"):
            bearly.probability_of_improvement(
                [np.array([1.0]), np.array([])],
                [np.array([1.0]), np.array([2.0])],
            )

    def test_nan_in_alignment_column_raises_or_drops(self):
        """NaN *task* labels must never silently match other NaN labels."""
        df = pd.DataFrame(
            {
                "agent": ["X", "X", "Y", "Y"],
                "task": ["t1", None, "t1", None],
                "run": [0, 0, 0, 0],
                "score": [1.0, 9.0, 2.0, 8.0],
            }
        )
        # np.unique keeps NaN in the task set; sorting the mixed
        # {None : str} intersection blows up inside the library.
        with pytest.raises(TypeError):
            bearly.get_paired_interval_estimates(
                df,
                ("agent", "X", "Y"),
                "task",
                "score",
                bearly.probability_of_improvement,
                n_samples=10,
            )
