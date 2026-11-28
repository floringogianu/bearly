"""Shared fixtures: small deterministic benchmark-shaped frames."""

import numpy as np
import pandas as pd
import pytest

# -- tiny benchmark: 2 agents x 2 tasks x 2 steps x 3 runs -------------------
# structure encodes the degenerate cases we care about:
#   (A, t1, s0): 3 strong runs          (A, t1, s1): 2 runs, one strong
#   (A, t2, s0): 2 weak-ish runs        (A, t2, s1): 1 weak run
#   (B, ...): constant 4.0  (pairwise-identical runs -> tie behaviour)
ROWS = [
    # agent, task, step, run, score
    ("A", "t1", 0, 0, 3.0),
    ("A", "t1", 0, 1, 4.0),
    ("A", "t1", 0, 2, 5.0),
    ("A", "t2", 0, 0, 1.0),
    ("A", "t2", 0, 1, 3.0),
    ("A", "t1", 1, 0, 6.0),
    ("A", "t1", 1, 1, 7.0),
    ("A", "t2", 1, 0, 2.0),
    ("B", "t1", 0, 0, 4.0),
    ("B", "t2", 0, 0, 4.0),
    ("B", "t2", 0, 1, 4.0),
    ("B", "t1", 1, 0, 4.0),
    ("B", "t2", 1, 0, 4.0),
]


def make_df(rows=ROWS) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["agent", "task", "step", "run", "score"])


@pytest.fixture
def bench_df() -> pd.DataFrame:
    return make_df()


@pytest.fixture
def nan_df(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark frame with one missing score and one missing task."""
    df = bench_df.copy()
    df.loc[
        (df["agent"] == "A")
        & (df["task"] == "t1")
        & (df["step"] == 0)
        & (df["run"] == 2),
        "score",
    ] = np.nan
    df.loc[
        (df["agent"] == "A") & (df["task"] == "t2") & (df["step"] == 1),
        "task",
    ] = None  # this row only: (A, t2, s1, r0)
    return df


@pytest.fixture
def minmax_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rom": ["g1", "g2"],
            "min": [0.0, 10.0],
            "max": [100.0, 50.0],
        }
    ).set_index("rom")
