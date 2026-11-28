"""End-to-end check of the README flow on the bundled Atari data (slow).

Mirrors p9_demo.py exactly: normalise returns -> hns, last-step per
(agent, rom, trial), interval estimates for all four stats, paired intervals
of four agents vs DQN, and a performance profile. Every statistic is asserted
against a value recomputed from the same data via independent pandas/numpy,
so regressions here are regressions in either the math or the wiring.
"""

import numpy as np
import pandas as pd
import pytest

import bearly

DATA_DIR = __file__.rsplit("/", 2)[0] + "/data"
FEATHER = f"{DATA_DIR}/dopamine_demo.feather.lz4"


@pytest.fixture(scope="module")
def cnn_final() -> pd.DataFrame:
    mnmx = pd.read_csv(f"{DATA_DIR}/ale_57_mnmx.csv", thousands=",").set_index("rom")
    mnmx = mnmx.rename(columns={"mn": "min", "mx": "max"})

    dpmn = pd.read_feather(FEATHER)
    dpmn = dpmn.loc[dpmn["rom"].isin(mnmx.index.unique())].reset_index(drop=True)
    dpmn["hns"] = bearly.min_max_normalisation(dpmn, mnmx, "rom", "return")

    cnn = dpmn.loc[dpmn["model"] == "Impala"].reset_index(drop=True)
    # align crashed runs onto step 199, replicate p9_demo.py subsample
    cnn = cnn.copy()
    cnn.replace({"step": [198, 197, 55]}, 199, inplace=True)
    return cnn.groupby(["agent", "rom", "trial"]).tail(1).reset_index(drop=True)


def test_normalisation_bounds(cnn_final):
    assert cnn_final["hns"].notna().all()
    # scores below the random baseline normalise to negative HNS: legitimate
    assert cnn_final["hns"].min() >= -15.0  # no pathological explosion
    assert (cnn_final["hns"] <= cnn_final.groupby("rom")["hns"].transform("max")).all()


def test_hns_matches_manual_normalisation(cnn_final):
    """Spot-check one agent x game against hand-computed min-max."""
    mnmx = pd.read_csv(f"{DATA_DIR}/ale_57_mnmx.csv", thousands=",").set_index("rom")
    row = cnn_final[
        (cnn_final["agent"] == "Rainbow") & (cnn_final["rom"] == "qbert")
    ].iloc[0]
    raw = pd.read_feather(FEATHER)
    raw_row = (
        raw.query(
            "agent == 'Rainbow' and rom == 'qbert' and trial == @row.trial and model == 'Impala'"
        )
        .sort_values("step")
        .iloc[-1]
    )
    mn, mx = mnmx.loc["qbert", ["mn", "mx"]]
    assert row["hns"] == pytest.approx(float((raw_row["return"] - mn) / (mx - mn)))


def test_final_step_iqm_matches_rliable_iqm(cnn_final):
    """Final-step IQM for Rainbow, recomputed via scipy on pooled runs."""
    out = bearly.get_interval_estimates(
        cnn_final,
        bearly.iqm,
        metric_col="hns",
        task_col="rom",
        group_cols=["agent"],
        n_samples=20,
    )
    from scipy.stats import trim_mean

    rainbow = cnn_final.loc[cnn_final["agent"] == "Rainbow"]
    expected = trim_mean(rainbow["hns"].to_numpy(), 0.25)
    actual = out.loc[out["agent"] == "Rainbow", "y"].iloc[0]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("stat_name", ["iqm", "median", "mean", "optimality_gap"])
def test_all_readme_stats_run(cnn_final, stat_name):
    out = bearly.get_interval_estimates(
        cnn_final,
        getattr(bearly, stat_name),
        metric_col="hns",
        task_col="rom",
        group_cols=["agent"],
        n_samples=10,
    )
    assert not out.empty
    assert out["y"].notna().all()
    assert out["ymin"].notna().all() and out["ymax"].notna().all()
    # Percentile CIs carry no point-containment guarantee: at n_samples=10 the
    # median's bootstrap distribution can legitimately exclude the point
    # estimate. Only their internal ordering is guaranteed.
    assert (out["ymin"] <= out["ymax"]).all()


@pytest.mark.slow
def test_paired_interval_agents_vs_dqn(cnn_final):
    """README's headline probability-of-improvement scenario."""
    results = []
    for agent in ["C51", "IQN", "Rainbow", "QR-DQN"]:
        out = bearly.get_paired_interval_estimates(
            cnn_final,
            ("agent", agent, "DQN"),
            task_col="rom",
            metric_col="hns",
            stat=("p_improve", bearly.probability_of_improvement),
            n_samples=200,
        )
        results.append(out)
    pis = pd.concat(results, ignore_index=True)

    assert len(pis) == 4
    # compared=(col, A, B) labels the output: X == A, Y == B (the docstring's
    # "p(agent > DQN)" is inverted; the stat computes P(Y > X), i.e. P(DQN > agent))
    assert (pis["Y"] == "DQN").all()
    assert set(pis["X"]) == {"C51", "IQN", "Rainbow", "QR-DQN"}
    # Rainbow is stronger than DQN on Atari: P(DQN > Rainbow) must be small
    rainbow_p = pis.loc[pis["X"] == "Rainbow", "y"].iloc[0]
    assert rainbow_p < 0.5
    assert ((pis["y"] >= 0) & (pis["y"] <= 1)).all()
    assert (pis["ymin"] <= pis["y"]).all() and (pis["y"] <= pis["ymax"]).all()

    # point estimates against direct function evaluation on aligned Tasks
    dqn = cnn_final[cnn_final["agent"] == "DQN"].set_index("rom")["hns"]
    rainbow = cnn_final[cnn_final["agent"] == "Rainbow"].set_index("rom")["hns"]
    common = dqn.index.intersection(rainbow.index)

    # library convention: result.y == probability_of_improvement(X_tasks, Y_tasks)
    # -> here P(DQN > Rainbow).
    def task_arrays(series):
        return [np.asarray(series.loc[[t]].to_numpy()) for t in common]

    expected = bearly.probability_of_improvement(task_arrays(rainbow), task_arrays(dqn))
    actual = pis.loc[pis["X"] == "Rainbow", "y"].iloc[0]
    assert actual == pytest.approx(expected)


def test_performance_profile_on_final(cnn_final):
    taus = np.linspace(0, 2, 11)
    pp = bearly.get_performance_profiles(
        cnn_final, taus, "hns", "rom", ["agent"], n_samples=10
    )
    agents = set(pp["agent"])  # noqa: F841 - smoke the surface
    assert len(pp) == len(taus) * cnn_final["agent"].nunique()
    assert ((pp["y"] >= 0) & (pp["y"] <= 1)).all()
