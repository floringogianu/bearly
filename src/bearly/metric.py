import numpy as np
from scipy.stats import mannwhitneyu, trim_mean

__all__ = ("iqm", "optimality_gap", "probability_of_improvement")


def iqm(arr, axis=None):
    return trim_mean(arr, 0.25, axis=axis)


def optimality_gap(scores: np.ndarray, gamma=1):
    return gamma - np.mean(np.minimum(scores, gamma))


def probability_of_improvement(scores_new, scores_baseline):
    if np.array_equal(scores_new, scores_baseline):
        return 0.5
    # Mann-Whitney U test (alternative='greater' tests if new > baseline)
    u_stat, _ = mannwhitneyu(scores_new, scores_baseline, alternative="greater")
    # normalize U statistic to a probability
    return u_stat / (len(scores_new) * len(scores_baseline))
