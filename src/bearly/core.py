from typing import Callable

import numpy as np
from pandas import DataFrame, concat

from .metric import probability_of_improvement

__all__ = (
    "get_interval_estimates",
    "get_probability_of_improvement",
    "stratified_sampling_with_replacement",
)


def stratified_sampling_with_replacement(
    df: DataFrame,
    measurement_col: str,
    strata_col: str,
    group_cols: list[str],
    repeats: int,
) -> DataFrame:
    """Stratified sampling. Allows for resampling even when the number of
    measurements for each `strata_col` (task) varies.

    Args:
        df (DataFrame): ...
        measurement_col (str): the column containing the signal we want to
            resample.
        strata_col (str): for each value in this column we sample with
            replacement N times, where N is the number of measurements for each
            `strata_col`.
        group_cols (list[str]): columns for which we *independently* apply this
            sampling procedure.
        repeats (int): number of times the sampling procedure is repeated. In
            total `repeats x len(measurements)` will be drawn.

    Returns:
        DataFrame: with a `sample` column in addition to the existing columns.

    Example:
        `metric, task, [checkpoint]`,
        `acc, dataset, [epoch]`,
        `hns, game, [step]`
    """
    _df = df.sort_values([strata_col, *group_cols])
    # _df = _df[[strata_col, *group_cols, measurement_col]].copy()

    grouped = _df.groupby([strata_col, *group_cols], sort=False)
    group_counts = grouped.size().to_numpy()
    group_starts = np.insert(np.cumsum(group_counts)[:-1], 0, 0)
    indices_list = []
    sample_ids_list = []

    for start, count in zip(group_starts, group_counts):
        if count == 0:
            continue
        rand_offsets = np.random.randint(0, count, size=count * repeats)
        indices_list.append(start + rand_offsets)
        ids = np.repeat(np.arange(repeats), count)
        sample_ids_list.append(ids)

    all_indices = np.concatenate(indices_list)
    all_sample_ids = np.concatenate(sample_ids_list)

    smpls = _df.iloc[all_indices].copy()
    smpls["sample"] = all_sample_ids
    return smpls


def get_interval_estimates(
    df: DataFrame,
    stat_fn: Callable | dict[str, Callable],
    metric: str,
    strata: str,
    group: list[str],
    n_samples: int = 1_000,
):
    """Uses stratified sampling to compute aggregate statistics and confidence
    intervals.
    """

    samples = stratified_sampling_with_replacement(df, metric, strata, group, n_samples)

    stat_fns = stat_fn if isinstance(stat_fn, dict) else {"stat_fn": stat_fn}

    res = []
    for stat_name, f in stat_fns.items():
        # compute bootstrap statistics
        ci = (
            samples.groupby([*group, "sample"])[metric]
            .apply(f)
            .reset_index()
            .groupby(group)[metric]
            .agg([lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
        )

        # rename columns
        ci.columns = ["ymin", "ymax"]
        ci = ci.reset_index()

        # add point estimates
        pe = df.groupby(group)[metric].apply(f).rename("y").reset_index()
        ci = ci.merge(pe, on=group, how="left")

        # add aggregation statistic name
        ci["stat_fn"] = stat_name
        res.append(ci)

    return concat(res, ignore_index=True)


def get_probability_of_improvement(
    df: DataFrame,
    compared: tuple[str, str, str],
    strata: str,
    metric: str,
    n_samples: int = 1_000,
) -> DataFrame:
    """Uses stratified sampling to compute probability of improvement of two
    algorithms.

    Args:
        df (DataFrame): a DataFrame ...
        compared (tuple[str,str,str]): a tuple specifying the column and the
            algorithms compared, eg.: (algo, quicksort, bubblesort)
        strata (str): column of tasks
        metric (str): column of values for which we compute the stats

    Returns:
        DataFrame: containing columns [pair, y, ymin, ymax]
    """

    col, x, y = compared

    # filter
    df = df.loc[df[col].isin([x, y])].reset_index(drop=True)

    # stratified sampling
    samples = stratified_sampling_with_replacement(df, metric, strata, [col], n_samples)

    # algo A and algo B scores in separate columns
    pairs = samples.pivot_table(
        index=["sample", strata],
        columns=col,
        values=metric,
        aggfunc=list,  # bundle all runs for a specific (sample, task) into one cell
    ).dropna()

    # the `probability_of_improvement` function receives raw lists
    pi_per_task_boot = pairs.apply(
        lambda row: probability_of_improvement(row[x], row[y]), axis=1
    )

    # mean over p(x>y | task), then quantiles over bootstrap samples
    pi_boot = pi_per_task_boot.groupby("sample").mean()

    # p(x>y) estimate
    pi_mean = (
        df.pivot_table(index=[strata], columns=col, values=metric, aggfunc=list)
        .dropna()
        .apply(lambda row: probability_of_improvement(row[x], row[y]), axis=1)
        .mean()
    )

    return DataFrame(
        {
            "X": [x],
            "Y": [y],
            "y": [pi_mean],
            "ymin": [pi_boot.quantile(0.025)],
            "ymax": [pi_boot.quantile(0.975)],
        }
    )
