"""Unit tests for bearly's aggregate stats: hand-computed oracles only.

rliable parity is exercised separately in test_stat_parity.py.
"""

import numpy as np
import pytest
from scipy.stats import trim_mean

from bearly import iqm, mean, median, optimality_gap, probability_of_improvement


def tasks(*arrays) -> list[np.ndarray]:
    return [np.asarray(a, dtype=float) for a in arrays]


class TestIQM:
    def test_hand_computed(self):
        # pooled: [1]*8 + [5]*8, trim 25% of 16 = 4 off each end -> mean = 3.0
        data = tasks([1.0] * 8, [5.0] * 8)
        assert iqm(data) == pytest.approx(3.0)

    def test_known_value(self):
        # concat = [0,1,2,3,10,11,12,100]; trim 2 each end -> mean([2,3,10,11])
        data = tasks([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 100.0])
        assert iqm(data) == pytest.approx(6.5)

    def test_uneven_task_sizes_pool_all_points(self):
        data = tasks([1.0, 2.0, 3.0], [10.0])
        expected = trim_mean([1.0, 2.0, 3.0, 10.0], 0.25)
        assert iqm(data) == pytest.approx(expected)

    def test_nan_propagates(self):
        data = tasks([1.0, np.nan], [3.0, 4.0])
        assert np.isnan(iqm(data))

    def test_single_task_single_run_is_that_value(self):
        assert iqm(tasks([7.0])) == pytest.approx(7.0)


class TestMeanAndMedian:
    @pytest.mark.parametrize(
        ("fn", "expected"),
        [(mean, (2.0 + 20.0) / 2), (median, np.median([2.0, 20.0]))],
        ids=["mean", "median"],
    )
    def test_mean_of_task_means(self, fn, expected):
        # t1 mean 2.0, t2 mean 20.0
        data = tasks([1.0, 3.0], [10.0, 30.0])
        assert fn(data) == pytest.approx(expected)

    def test_mean_ignores_task_sizes_when_averaging(self):
        # t1 has 3 runs, t2 has 1: pooled mean would be (2+2+2+9)/4 = 3.75,
        # mean-of-task-means is (2 + 9) / 2 = 5.5
        data = tasks([2.0, 2.0, 2.0], [9.0])
        assert mean(data) == pytest.approx(5.5)

    def test_median_odd_task_count(self):
        data = tasks([1.0, 1.0], [5.0, 5.0], [9.0, 9.0])
        assert median(data) == pytest.approx(5.0)

    def test_nan_propagates(self):
        data = tasks([1.0, np.nan], [3.0, 4.0])
        assert np.isnan(mean(data))
        assert np.isnan(median(data))


class TestOptimalityGap:
    def test_computation(self):
        data = tasks([0.5, 2.0, 0.0], [1.0, 0.75])
        # clip to gamma=1: [0.5, 1, 0, 1, 0.75] -> 1 - 3.25/5
        assert optimality_gap(data) == pytest.approx(1 - 3.25 / 5)

    def test_scores_above_gamma_clipped_down(self):
        data = tasks([2.0, 1.0])
        # clip -> [1, 1]: gap = 0 despite a score of 2.0
        assert optimality_gap(data, gamma=1.0) == pytest.approx(0.0)

    def test_custom_gamma(self):
        data = tasks([10.0, 2.0])
        assert optimality_gap(data, gamma=10.0) == pytest.approx(10 - 6.0)

    def test_nan_propagates(self):
        data = tasks([np.nan, 0.5])
        assert np.isnan(optimality_gap(data))


class TestProbabilityOfImprovement:
    def test_identical_tasks_is_half(self):
        base = tasks([1.0, 2.0], [3.0])
        other = tasks([1.0, 2.0], [3.0])
        assert probability_of_improvement(base, other) == pytest.approx(0.5)

    def test_strict_greater_than_base_is_one(self):
        base = tasks([1.0, 1.0])
        other = tasks([2.0, 2.0, 3.0])
        assert probability_of_improvement(base, other) == pytest.approx(1.0)

    def test_strict_less_than_base_is_zero(self):
        base = tasks([2.0, 3.0])
        other = tasks([1.0])
        assert probability_of_improvement(base, other) == pytest.approx(0.0)

    def test_argument_order_drives_direction(self):
        """stat_fn(baseline, Y) asks 'is Y better' -> MWU(Y, baseline, greater).

        Note on ties: scipy's MWU U-statistic counts only strict wins; equal
        pairs receive 0 credit (the 0.5/tie correction affects ranks, not U).
        rliable's probability_of_improvement inherits this exact behaviour.
        """
        base = tasks([1.0, 2.0])
        other = tasks([2.0, 2.0])
        # strict wins: 2+1 = 3 of 4 pairs; the tie (2,2) counts for 0
        assert probability_of_improvement(base, other) == pytest.approx(0.75)
        # mirrored question must mirror back: P(X > Y) = 1 - (3 wins + 0 ties)
        assert probability_of_improvement(other, base) == pytest.approx(0.25)

    def test_ties_contribute_half_symmetric_case(self):
        base = tasks([1.0, 2.0])
        other = tasks([2.0, 1.0])
        assert probability_of_improvement(base, other) == pytest.approx(0.5)

    def test_averaged_across_tasks(self):
        # t1: Y wins for sure (1.0), t2: identical runs (0.5)
        base = tasks([1.0, 1.0], [7.0, 7.0])
        other = tasks([2.0], [7.0, 7.0])
        assert probability_of_improvement(base, other) == pytest.approx(0.75)

    def test_mismatched_task_count_raises(self):
        with pytest.raises(ValueError, match="same number of Tasks"):
            probability_of_improvement(tasks([1.0]), tasks([1.0], [2.0]))

    def test_nan_behaves_like_real_data(self, recwarn):
        """NaN breaks exact equality and yields a NaN MWU statistic.

        rliable semantics: no warning, silent NaN in the average.
        """
        base = tasks([1.0, np.nan])
        other = tasks([1.0, 2.0])
        assert np.isnan(probability_of_improvement(base, other))

    def test_incomparable_dtype_raises(self):
        """Metric arrays with non-numeric dtype cannot be ranked."""
        base = [np.array(["a", "b"], dtype=object)]
        other = [np.array([1.0, 2.0])]
        with pytest.raises(TypeError):
            probability_of_improvement(base, other)
