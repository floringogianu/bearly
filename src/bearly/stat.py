"""Aggregate stats, of the form f((runs x task)) = g(h(runs) for each task),
where g an h can be the same function or the function is applied to all runs x
tasks measurements.
"""

from typing import Annotated, Protocol

import numpy as np
from scipy.stats import mannwhitneyu, trim_mean

__all__ = ("iqm", "mean", "median", "optimality_gap", "probability_of_improvement")


# domain types

type Trials = Annotated[np.ndarray, "Shape: (N,)"]  # one datapoint per task over trials
type Tasks = list[Trials]  # an aggregate of many tasks and their trials


class AggregateStat(Protocol):
    def __call__(self, data: Tasks) -> float: ...


class PairedStat(Protocol):
    def __call__(self, baseline: Tasks, Y: Tasks) -> float: ...


def mean(data: Tasks) -> float:
    """Mean over runs per task, then mean over tasks."""
    task_means = [np.mean(task) for task in data]
    return float(np.mean(task_means))


def median(data: Tasks) -> float:
    """**Mean** over runs per task, then **median** over tasks."""
    task_means = [np.mean(task) for task in data]
    return float(np.median(task_means))


def iqm(data: Tasks) -> float:
    """Interquartile Mean over **all** runs, ignoring task boundaries."""
    return float(trim_mean(np.concatenate(data), proportiontocut=0.25))


def optimality_gap(data: Tasks, gamma: float = 1.0) -> float:
    """Optimality Gap over **all** runs, ignoring task boundaries."""
    return float(gamma - np.mean(np.minimum(np.concatenate(data), gamma)))


def probability_of_improvement(baseline: Tasks, Y: Tasks) -> float:
    """Computes P(Y > Baseline) averaged across tasks."""

    if len(baseline) != len(Y):
        raise ValueError("Baseline and Y must have the same number of Tasks.")

    task_probs = []

    for runs_base, runs_Y in zip(baseline, Y):
        rb0, rn0 = len(runs_base) == 0, len(runs_Y) == 0
        if rn0 or rn0:
            faulty = "both" if rb0 and rn0 else "baseline" if rb0 else "Y"
            print(f"warn: {faulty} tasks have 0 runs")
            continue

        if np.array_equal(runs_base, runs_Y):
            task_probs.append(0.5)
            continue

        # TODO: maybe implement?
        u_stat, _ = mannwhitneyu(runs_Y, runs_base, alternative="greater")

        prob = u_stat / (len(runs_Y) * len(runs_base))
        task_probs.append(prob)

    return float(np.mean(task_probs))
