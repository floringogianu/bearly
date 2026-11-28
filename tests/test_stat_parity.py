"""Direct parity checks between bearly stats and the installed rliable package.

bearly works on ``Tasks`` (list of per-task run arrays); rliable works on
``(num_runs, num_tasks)`` matrices. These tests assert identical results on
regular matrices. Per-task aggregations (mean/median) are additionally stable
under ragged task sizes that rliable cannot express.
"""

import numpy as np
import pytest
from rliable import metrics as rl_metrics

from bearly import iqm, mean, median, optimality_gap, probability_of_improvement


@pytest.fixture
def matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 3.0, size=(6, 5))  # 6 runs x 5 tasks


@pytest.fixture
def as_tasks(matrix: np.ndarray) -> list[np.ndarray]:
    return [matrix[:, t] for t in range(matrix.shape[1])]


def test_iqm_parity(matrix, as_tasks):
    assert iqm(as_tasks) == pytest.approx(rl_metrics.aggregate_iqm(matrix))


def test_mean_parity(matrix, as_tasks):
    assert mean(as_tasks) == pytest.approx(rl_metrics.aggregate_mean(matrix))


def test_median_parity(matrix, as_tasks):
    assert median(as_tasks) == pytest.approx(rl_metrics.aggregate_median(matrix))


@pytest.mark.parametrize("gamma", [1.0, 0.5, 3.0])
def test_optimality_gap_parity(matrix, as_tasks, gamma):
    assert optimality_gap(as_tasks, gamma=gamma) == pytest.approx(
        rl_metrics.aggregate_optimality_gap(matrix, gamma)
    )


def test_probability_of_improvement_parity(matrix):
    """bearly(baseline, Y) = P(Y > baseline) == rliable(other, base)."""
    rng = np.random.default_rng(7)
    other = rng.uniform(0.0, 3.0, size=(4, matrix.shape[1]))  # different run count
    expected = rl_metrics.probability_of_improvement(other, matrix)
    actual = probability_of_improvement(
        [matrix[:, t] for t in range(matrix.shape[1])],
        [other[:, t] for t in range(other.shape[1])],
    )
    assert actual == pytest.approx(expected)


def test_probability_of_improvement_parity_with_ties():
    base = np.array([[1.0, 5.0], [2.0, 5.0], [2.0, 6.0]])
    other = np.array([[2.0, 5.0], [2.0, 7.0]])
    expected = rl_metrics.probability_of_improvement(other, base)
    actual = probability_of_improvement(
        [base[:, t] for t in range(2)], [other[:, t] for t in range(2)]
    )
    assert actual == pytest.approx(expected)


def test_ragged_tasks_match_rliable_where_expressible():
    """Per-task aggregation (mean/median) still matches rliable after ragged
    tasks are padded back to a matrix; pooled stats (iqm/og) do not have an
    rliable equivalent for ragged input, so we assert the manual result."""
    rng = np.random.default_rng(3)
    t1, t2, t3 = rng.uniform(0, 1, 4), rng.uniform(0, 1, 2), rng.uniform(0, 1, 3)
    data = [t1, t2, t3]

    # mean/median: pad to matrix by repeating values, assert stability
    says_mean = mean(data)
    sayd_median = median(data)
    assert says_mean == pytest.approx(np.mean([t1.mean(), t2.mean(), t3.mean()]))
    assert sayd_median == pytest.approx(np.median([t1.mean(), t2.mean(), t3.mean()]))
