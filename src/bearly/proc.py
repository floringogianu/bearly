from pandas import DataFrame, Series

__all__ = ("min_max_normalisation",)


def min_max_normalisation(
    df: DataFrame,
    min_max: DataFrame,
    col: str,
    metric: str,
) -> Series:
    """Group by `col` and normalise values in `metric` using min and max values
    in `min_max`.

    Args:
        df (DataFrame): must contain a `col` with values present in min_max's
            index.
        min_max (DataFrame): is required to have a special structure comprised
            of index `{col}` and columns `max`, `min`. For example: `game`, `max`,
            `min`, where `min` could be the performance of the random policy.
        col (str): the column we groupby, usually something along the lines of
            "game", "rom", "task", "dataset".
        metric (str): the column containing measurements.
    """

    assert set(min_max.columns) == {"min", "max"}, (
        f"Improper columns: {min_max.columns}, should be [min, max]."
    )
    assert min_max.index.name == col, (
        f"Name of the min_max index is required to match `{col}`."
    )
    vals_in_df = set(df[col].unique())
    idxs_in_min_max = set(min_max.index.values)
    assert vals_in_df.issubset(idxs_in_min_max), (
        f"All `{col}` values in `df` are rquired to be preset in the index of `min_max`!"
        f" {vals_in_df - idxs_in_min_max} missing from `df`."
    )

    # map is vectorized, preserves row order, and returns a Series for any
    # group shape (unlike groupby-apply, which changes its output layout with
    # equal vs unequal group sizes).
    mn = df[col].map(min_max["min"])
    mx = df[col].map(min_max["max"])
    res = ((df[metric] - mn) / (mx - mn)).rename(metric)
    assert isinstance(res, Series), f"The result should be a pandas.Series: {res}"
    return res
