import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotnine as p9

    from bearly import (
        get_interval_estimates,
        get_probability_of_improvement,
        iqm,
        min_max_normalisation,
        optimality_gap,
    )
    return (
        get_interval_estimates,
        get_probability_of_improvement,
        iqm,
        min_max_normalisation,
        mo,
        np,
        optimality_gap,
        p9,
        pd,
    )


@app.cell
def _(p9):
    _ = p9.theme_set(p9.theme_classic())
    return


@app.cell
def _(min_max_normalisation, pd):
    mnmx = pd.read_csv("./data/ale_57_mnmx.csv", thousands=",").set_index("rom")
    mnmx = mnmx.rename(columns={"mn": "min", "mx": "max"})

    dpmn = pd.read_feather("./data/dopamine.feather.lz4")
    dpmn = dpmn.loc[dpmn["rom"].isin(mnmx.index.unique())].reset_index(drop=True)

    dpmn["hns"] = min_max_normalisation(dpmn, mnmx, "rom", "return")
    dpmn = dpmn[["step", "agent", "model", "rom", "trial", "return", "hns"]]

    # subsample
    _steps = [0, 25, 50, 75, 100, 125, 150, 175]
    _group = ["agent", "model", "rom", "trial"]
    max_steps = dpmn.groupby(_group)["step"].max().reset_index(name="max_step")
    dpmn_max = dpmn.merge(max_steps, on=_group, how="left")
    mask = dpmn_max["step"].isin(_steps) | (dpmn_max["step"] == dpmn_max["max_step"])
    dpmn = dpmn_max.loc[mask].drop(columns="max_step").reset_index(drop=True)
    return (dpmn,)


@app.cell
def _(dpmn):
    # some runs crashed
    print(dpmn["step"].unique())
    dpmn.replace({"step": [198, 197, 55]}, 199, inplace=True)
    print(dpmn["step"].unique())
    return


@app.cell
def _(dpmn, iqm, p9):
    (
        p9.ggplot(dpmn, p9.aes(x="step", y="hns", color="agent"))
        + p9.stat_summary(geom="line", fun_y=iqm, size=1)
        + p9.facet_wrap("model", ncol=5, labeller="label_context")
        + p9.labs(
            y="episodict return (average over seeds)",
            x="steps (M)",
            color="agent",
            subtitle="sample efficiency of different training protocols",
        )
        + p9.theme(
            figure_size=(9, 3),
            # legend_position="top",
            # legend_direction="horizontal",
            axis_text_x=p9.element_text(rotation=35),
            strip_background=p9.themes.elements.element_blank(),
        )
    ).draw()
    return


@app.cell
def _(dpmn):
    cnn = dpmn.loc[dpmn["model"] == "Impala"].reset_index(drop=True)
    cnn.groupby(["agent"]).sample(2)
    return (cnn,)


@app.cell
def _(iqm, np, optimality_gap):
    stat_fns = {
        "iqm": iqm,
        "median": np.median,
        "mean": np.mean,
        "optimality gap": optimality_gap,
    }
    return (stat_fns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sample efficiency curves
    """)
    return


@app.cell
def _(cnn, get_interval_estimates, stat_fns):
    # stratified bootstrapping over each (agent, step) combination, # where "rom"
    # (game) is the strata and "hns" is the metric we care about (here, the human
    # normalised score).
    ci = get_interval_estimates(cnn, stat_fns, "hns", "rom", ["agent", "step"], 2_000)
    ci
    return (ci,)


@app.cell
def _(ci, p9):
    sample_efficiency = (
        p9.ggplot(
            ci.loc[ci["stat_fn"] != "optimality gap"],
            p9.aes(x="step", y="y", color="agent", fill="agent"),
        )
        + p9.geom_line(size=0.75)
        + p9.geom_point()
        + p9.geom_ribbon(p9.aes(ymax="ymax", ymin="ymin"), alpha=0.2, linetype="")
        + p9.facet_wrap("~stat_fn", scales="free_y")
        + p9.labs(
            y="human normalised score", x="frames (M)", color="agent:", fill="agent:",
            subtitle="sample efficiency curves (Impala estimator)"
        )
        + p9.theme(
            figure_size=(9, 3),
            legend_position="top",
            legend_direction="horizontal",
            legend_justification="left",
            # axis_text_x=p9.element_text(rotation=35),
            strip_background=p9.themes.elements.element_blank(),
        )
    )

    sample_efficiency.save("./img/sample_efficiency.png", dpi=320)
    sample_efficiency.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### final performance comparison
    """)
    return


@app.cell
def _(ci, p9):
    ci_final = ci.groupby(["agent", "stat_fn"]).tail(1)

    final_performance = (
        p9.ggplot(ci_final, p9.aes(x="agent", fill="agent", color="agent"))
        + p9.geom_crossbar(p9.aes(y="y", ymin="ymin", ymax="ymax"), alpha=0.1)
        + p9.coord_flip()
        + p9.facet_wrap("~stat_fn", ncol=4, scales="free_x")
        + p9.theme(
            figure_size=(9, 2.5),
            legend_position="none",
            strip_background=p9.themes.elements.element_blank(),
        )
        + p9.labs(y="human normalised score", subtitle="final score estimates and 95% confidence intervals")
    )

    final_performance.save("./img/final_performance.png", dpi=320)
    final_performance.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### probability of improvement
    """)
    return


@app.cell
def _(cnn):
    cnn_final = cnn.groupby(["agent", "rom", "trial"]).tail(1).reset_index()
    cnn_final.groupby(["agent"])["hns"].count()
    return (cnn_final,)


@app.cell
def _(cnn_final, get_probability_of_improvement, pd):
    # using DQN as a baseline
    # it takes a while, don't worry :)
    pis = []
    for agent in ["C51", "IQN", "Rainbow", "QR-DQN"]:
        print(f"computing p({agent} > DQN)")
        pis.append(
            get_probability_of_improvement(
                cnn_final, ("agent", agent, "DQN"), "rom", "hns", 2000
            )
        )
    pis = pd.concat(pis, ignore_index=True)
    return (pis,)


@app.cell
def _(pis):
    pis
    return


@app.cell
def _(p9, pis):
    probability_of_improvement = (
        p9.ggplot(pis, p9.aes(x="X", fill="X", color="X"))
        + p9.geom_crossbar(p9.aes(y="y", ymin="ymin", ymax="ymax"), alpha=0.1)
        + p9.coord_flip()
        # + p9.scale_color_discrete(breaks=reversed)
        + p9.theme(
            figure_size=(4, 2),
            legend_position="none",
            # legend_direction="horizontal",
            strip_background=p9.themes.elements.element_blank(),
        )
        + p9.labs(
            x="algorithm Y", y="p(Y > DQN)", subtitle="probability of improvement over DQN"
        )
    )

    probability_of_improvement.save("./img/probability_of_improvement.png", dpi=320)
    probability_of_improvement.draw()
    return


if __name__ == "__main__":
    app.run()
