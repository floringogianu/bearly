from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from .stat import (
    AggregateStat,
    BootAggregate,
    PairedStat,
    R,
    Tasks,
    set_performance_at_tau_stat,
)

__all__ = (
    "get_interval_estimates",
    "get_paired_interval_estimates",
    "get_performance_profiles",
)


def _generate_stratified_bootstrap(data: Tasks, rng: np.random.Generator) -> Tasks:
    """Resamples runs within each Task with replacement."""
    resampled_data = []
    for task_runs in data:
        n = len(task_runs)
        indices = rng.integers(0, n, size=n)
        resampled_data.append(task_runs[indices])
    return resampled_data


def _compute_interval_estimates(
    data: Tasks,
    metric_fn: AggregateStat[R],
    n_samples: int = 2000,
    confidence: float = 0.95,
) -> BootAggregate:
    rng = np.random.default_rng()

    # TODO: can this be faster?
    boot_estimates = np.array(
        [metric_fn(_generate_stratified_bootstrap(data, rng)) for _ in range(n_samples)]
    )

    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(boot_estimates, alpha, axis=0)
    upper = np.quantile(boot_estimates, 1.0 - alpha, axis=0)

    point_estimate = metric_fn(data)

    return BootAggregate(point_estimate, lower, upper)


def get_interval_estimates(
    df: pd.DataFrame,
    stat: AggregateStat[R] | tuple["str", AggregateStat[R]],
    metric_col: str,
    task_col: str,
    group_cols: list[str],
    n_samples: int = 2000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Uses stratified sampling over tasks to compute aggregates and confidence
    intervals using variouse statistical estimates.

    How it works:
        1. groups by `group_cols`.
        2. converts each group into `Tasks` (`list[np.ndarray]`).
        3. runs the bootstrap.
        4. returns a DataFrame with CIs.

    Args:
        df: tabular data with columns containing:

                - algorithms such as agents or models,
                - tasks such as games or datasets,
                - normalised_values such as accuracy, returns, etc.

            Optionaly, it can also contain columns such as:

                - ticks, suchs as steps or epochs
                - hyperparameters over which to group by

            !! We assume at least a column to be grouped over (such as
            algorithms or ticks).
            TODO: Maybe relax on this assumption?

        stat:       function (and its name) used for aggregating over runs and tasks.
        metric_col: column which contains the evaluations, eg.: `hns`, `val_acc`, etc.
        task_col:   column which contains the tasks, eg.: `game`, `dataset`, etc.
        group_cols: list of columns over which we want to apply this function
            independently, eg.: [`agent`, `step`] or [`model`, `epoch`].
        n_samples:  how many times we apply the procedure of sampling measurements of tasks.
        confidence: confidence interval value

    Returns:
        DataFrame: containing y, ymin, ymax, stat_fn
    """
    fname, stat = stat if isinstance(stat, tuple) else (stat.__name__, stat)
    results = []

    # group by the high-level identifiers (e.g., algorithm, checkpoint)
    grouped = df.groupby(group_cols)

    for group_keys, sub_df in tqdm(grouped, f"{fname:<16s}"):
        # handle single vs multiple group columns
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        # convert DataFrame to Tasks
        # sort by task to ensure deterministic ordering
        sub_df = sub_df.sort_values(task_col)

        # result is [Array(Run1, Run2...), Array(Run1...), ...]
        task_groups = sub_df.groupby(task_col, sort=False)[metric_col]
        benchmark_data: Tasks = [vals.to_numpy() for _, vals in task_groups]

        # 3. Call the Fast Pure Function
        stats = _compute_interval_estimates(
            benchmark_data, stat, n_samples=n_samples, confidence=confidence
        )

        # 4. Reconstruct Result Row
        row = dict(zip(group_cols, group_keys))
        row.update(
            {
                "y": stats.point_estimate,
                "ymin": stats.ci_lower,
                "ymax": stats.ci_upper,
                "stat_fn": fname,
            }
        )
        results.append(row)

    return pd.DataFrame(results)


def _compute_paired_estimates(
    baseline,
    other,
    stat_fn: PairedStat,
    n_samples: int = 2000,
    confidence: float = 0.95,
    pair_name: str = "",
) -> BootAggregate:
    """Compute estimate on paired identifiers, such as two algorithms using a
    paired aggregation function. For example, the probability of improvement.
    """
    rng = np.random.default_rng()

    estimates = []

    for _ in tqdm(range(n_samples), f"{pair_name:<16s}"):
        # generate independent new bootstrap samples for both
        # TODO: think wether we would want to have the same indices for paired
        # seeds.
        boot_base = _generate_stratified_bootstrap(baseline, rng)
        boot_new = _generate_stratified_bootstrap(other, rng)

        # compute the stat on the resampled pair
        estimates.append(stat_fn(boot_base, boot_new))

    # percentiles
    boot_estimates = np.array(estimates)
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(boot_estimates, alpha)
    upper = np.quantile(boot_estimates, 1.0 - alpha)
    # point estimate
    point_estimate = stat_fn(baseline, other)

    return BootAggregate(point_estimate, lower, upper)


def get_paired_interval_estimates(
    df: pd.DataFrame,
    compared: tuple[str, str, str],
    task_col: str,
    metric_col: str,
    stat: PairedStat | tuple["str", PairedStat],
    n_samples: int = 2000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Uses stratified sampling to compute probability of improvement of two
    algorithms.

    Args:
        df: tabular data with columns containing:

            - algorithms such as agents or models,
            - tasks such as games or datasets,
            - normalised_values such as accuracy, returns, etc.

        compared: a tuple specifying the column and the
            algorithms compared, eg.: (algo, quicksort, bubblesort)
        task_col: column of tasks
        metric_col: column of values for which we compute the stats

    Returns:
        DataFrame: containing columns [pair, y, ymin, ymax]
    """

    algo_col, baseline_algo, other_algo = compared
    fname, stat = stat if isinstance(stat, tuple) else (stat.__name__, stat)

    # filter the two algorithms
    df_base = df[df[algo_col] == baseline_algo]
    df_other = df[df[algo_col] == other_algo]

    # intersection of tasks (Alignment)
    # TODO: we need some asserts or warnings here!!
    base_task_set = np.unique(df_base[task_col])
    other_task_set = np.unique(df_other[task_col])
    common_tasks = sorted(list(set(base_task_set) & set(other_task_set)))

    if not common_tasks:
        raise ValueError(
            f"No common tasks found between {baseline_algo} and {other_algo}"
        )

    # convert to Tasks -> (list[Trials])
    base_tasks: Tasks = []
    other_tasks: Tasks = []
    grouped_base = df_base.groupby(task_col)
    grouped_new = df_other.groupby(task_col)

    for task in common_tasks:
        base_tasks.append(grouped_base.get_group(task)[metric_col].to_numpy())
        other_tasks.append(grouped_new.get_group(task)[metric_col].to_numpy())

    # compute the bootstrapped paired stats
    result: BootAggregate = _compute_paired_estimates(
        baseline=base_tasks,
        other=other_tasks,
        stat_fn=stat,
        n_samples=n_samples,
        confidence=confidence,
        pair_name=f"p({other_algo} > {baseline_algo})",
    )

    return pd.DataFrame(
        {
            "X": [baseline_algo],
            "Y": [other_algo],
            "y": [result.point_estimate],
            "ymin": [result.ci_lower],
            "ymax": [result.ci_upper],
            "stat_fn": [fname],
        }
    )


def get_performance_profiles(
    df: pd.DataFrame,
    tau_values: np.ndarray | None,
    metric_col: str,
    task_col: str,
    group_cols: list[str],
    deviation: Literal["run_score", "mean_score"] = "run_score",
    n_samples: int = 2000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute performance profiles over aggregates of tasks for each group.

    Performance profiles allows for visualising the fraction of tasks for which
    an algorithm achieves a normalized score at least equal to a threshold τ.
    This provides a comprehensive view of algorithm performance across the
    entire range of possible scores, capturing both the average performance and
    the robustness of an algorithm across tasks.

    Args:
        df: tabular data containing benchmark results with columns:
            - algorithms such as agents or models,
            - tasks such as games or datasets,
            - normalized values such as accuracy, returns, etc.
            Optionally, it can also contain columns such as:
            - ticks, such as steps or epochs
            - hyperparameters over which to group by
        tau_values: array of normalized score thresholds (τ) to evaluate performance
            at. If None, defaults to 101 evenly spaced values from 0 to 1.
        metric_col: column name containing the normalized scores (e.g., hns,
            val_acc, etc.).
        task_col: column name containing the task identifiers (e.g., game,
            dataset, etc.).
        group_cols: list of column names over which we want to apply this function
            independently, e.g., [agent, step] or [model, epoch].
        n_samples: number of bootstrap samples to use for confidence interval
            estimation (default: 2000).
        confidence: confidence level for the intervals (default: 0.95).

    Returns:
        DataFrame: containing columns [group_cols..., tau, y, ymin, ymax]
            - group_cols: the grouping columns from the input
            - tau: the normalized score threshold
            - y: the fraction of tasks achieving at least τ score
            - ymin, ymax: lower and upper confidence bounds for y
    """
    tau_values = np.linspace(0, 1.0, 101) if tau_values is None else tau_values
    stat = set_performance_at_tau_stat(tau_values, deviation)

    # y, ymin, ymax are lists now, one value for each tau
    pp = get_interval_estimates(
        df,
        ("pp", stat),
        metric_col,
        task_col,
        group_cols,
        n_samples,
        confidence,
    )
    # add tau as column
    pp["tau"] = [tau_values] * len(pp)
    # explode the lists to one value per row
    pp = pp.explode(["y", "ymin", "ymax", "tau"])
    # somehow these columns end up with dtype object (because of the numpy arrays)
    pp[["y", "ymin", "ymax", "tau"]] = pp[["y", "ymin", "ymax", "tau"]].astype(float)

    return pp
