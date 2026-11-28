"""Tests for bearly.min_max_normalisation."""

import numpy as np
import pandas as pd
import pytest

from bearly import min_max_normalisation


@pytest.fixture
def flat_df():
    return pd.DataFrame(
        {
            "rom": ["g1", "g1", "g2", "g2"],
            "return": [0.0, 50.0, 10.0, 30.0],
        }
    )


def test_normalises_per_group(minmax_df, flat_df):
    out = min_max_normalisation(flat_df, minmax_df, "rom", "return")
    # g1: (0-0)/100, (50-0)/100 ; g2: (10-10)/40, (30-10)/40
    np.testing.assert_allclose(out.to_numpy(), [0.0, 0.5, 0.0, 0.5], atol=1e-12)
    assert out.index.equals(flat_df.index)
    assert out.name == "return"


def test_output_aligned_with_input_order(minmax_df):
    df = pd.DataFrame(
        {"rom": ["g2", "g1", "g1", "g2"], "return": [30.0, 100.0, 0.0, 10.0]}
    )
    out = min_max_normalisation(df, minmax_df, "rom", "return")
    np.testing.assert_allclose(out.to_numpy(), [0.5, 1.0, 0.0, 0.0], atol=1e-12)


def test_unequal_group_sizes_preserve_index_and_order(minmax_df):
    df = pd.DataFrame(
        {
            "rom": ["g1", "g2", "g2", "g1", "g1"],
            "return": [25.0, 10.0, 30.0, 50.0, 100.0],
        }
    )
    out = min_max_normalisation(df, minmax_df, "rom", "return")
    np.testing.assert_allclose(out.to_numpy(), [0.25, 0.0, 0.5, 0.5, 1.0])
    assert out.index.equals(df.index)


def test_single_row_group(minmax_df):
    df = pd.DataFrame({"rom": ["g1"], "return": [25.0]})
    out = min_max_normalisation(df, minmax_df, "rom", "return")
    np.testing.assert_allclose(out.to_numpy(), [0.25])


def test_scores_outside_range(minmax_df):
    df = pd.DataFrame({"rom": ["g1", "g1"], "return": [-10.0, 150.0]})
    out = min_max_normalisation(df, minmax_df, "rom", "return")
    np.testing.assert_allclose(out.to_numpy(), [-0.1, 1.5])


def test_nan_metric_propagates(minmax_df):
    df = pd.DataFrame({"rom": ["g1", "g1"], "return": [np.nan, 50.0]})
    out = min_max_normalisation(df, minmax_df, "rom", "return")
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.5)


def test_nan_min_max_bounds_propagate():
    mnmx = pd.DataFrame({"rom": ["g1"], "min": [np.nan], "max": [100.0]}).set_index(
        "rom"
    )
    df = pd.DataFrame({"rom": ["g1"], "return": [10.0]})
    out = min_max_normalisation(df, mnmx, "rom", "return")
    assert np.isnan(out.iloc[0])


class TestRejectsMalformedInput:
    def test_missing_task_in_bounds(self, minmax_df):
        df = pd.DataFrame({"rom": ["g1", "nope"], "return": [1.0, 2.0]})
        with pytest.raises(AssertionError):
            min_max_normalisation(df, minmax_df, "rom", "return")

    def test_wrong_bound_columns(self, flat_df):
        mnmx = pd.DataFrame(
            {"rom": ["g1", "g2"], "lo": [0, 0], "hi": [1, 1]}
        ).set_index("rom")
        with pytest.raises(AssertionError, match="Improper columns"):
            min_max_normalisation(flat_df, mnmx, "rom", "return")

    def test_bound_index_name_mismatch(self, flat_df, minmax_df):
        mnmx = minmax_df.rename_axis("task")
        with pytest.raises(AssertionError, match="match `rom`"):
            min_max_normalisation(flat_df, mnmx, "rom", "return")

    def test_degenerate_bounds_infinite_or_nan(self):
        mnmx = pd.DataFrame({"rom": ["g1"], "min": [5.0], "max": [5.0]}).set_index(
            "rom"
        )
        df = pd.DataFrame({"rom": ["g1", "g1"], "return": [5.0, 6.0]})
        with np.errstate(invalid="ignore", divide="ignore"):
            out = min_max_normalisation(df, mnmx, "rom", "return")
        assert np.isnan(out.iloc[0]) and np.isinf(out.iloc[1])

    def test_incompatible_dtypes(self, minmax_df):
        df = pd.DataFrame({"rom": ["g1", "g1"], "return": ["low", "high"]})
        with pytest.raises((TypeError, ValueError)):
            min_max_normalisation(df, minmax_df, "rom", "return")
