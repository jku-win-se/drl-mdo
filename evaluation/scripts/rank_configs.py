#!/usr/bin/env python3
"""Rank settings by held-out HV per model within (model, seed) blocks, to find the best configuration.

ranks.csv also shows each setting's best training metric, its rank by that metric and its mean FEs
to convergence, so you can see whether the best on training also wins on held-out data.
"""

import numpy as np
import pandas as pd
from scipy import stats

import analysis_common as common


def hv_rows(runs, dataset):
    curves = {run: common.final_curves(run, dataset) for run in runs}
    bounds = common.Bounds()
    for by_model in curves.values():
        for model, blocks in by_model.items():
            for _, points in blocks:
                bounds.add(model, points)
    return [
        {"setting": run.setting, "seed": run.seed, "model": model,
         "hv": bounds.hv(model, blocks[-1][1]) if blocks else 0.0}
        for run, by_model in curves.items()
        for model, blocks in by_model.items()
    ]


def significance(wide, best):
    """Friedman over all settings, then Wilcoxon against the best with Holm correction."""
    friedman_p = float("nan")
    if wide.shape[1] >= 3 and len(wide) >= 2:
        friedman_p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns]).pvalue
    others = [c for c in wide.columns if c != best]
    raw = []
    for other in others:
        if np.allclose(wide[best].values, wide[other].values):
            raw.append(1.0)
        else:
            raw.append(stats.wilcoxon(wide[best].values, wide[other].values, zero_method="zsplit").pvalue)
    adjusted = common.holm(raw) if raw else []
    return friedman_p, dict(zip(others, adjusted))


def main():
    parser = common.parser_with_runs(__doc__)
    parser.add_argument("--dataset", default="test", help="Final-run dataset to rank on")
    options = parser.parse_args()

    runs = common.find_runs(options.runs)
    rows = hv_rows(runs, options.dataset)
    if not rows:
        raise SystemExit(f"No solutions for dataset '{options.dataset}'; did the runs use --final-eval?")

    wide, ranks = common.rank_within_blocks(pd.DataFrame(rows))
    mean_ranks = ranks.mean().sort_values()
    best = mean_ranks.index[0]
    friedman_p, adjusted = significance(wide, best)

    training = pd.DataFrame([
        {"setting": run.setting, "metric": common.best_training_metric(run),
         "fes": common.read_runtimes(run)["fes"]}
        for run in runs
    ]).groupby("setting").mean()

    summary = pd.DataFrame({
        "setting": mean_ranks.index,
        "mean_rank": mean_ranks.values,
        "mean_hv": wide.mean()[mean_ranks.index].values,
        "holm_p_vs_best": [adjusted.get(s, float("nan")) for s in mean_ranks.index],
        "best_training_metric": training["metric"][mean_ranks.index].values,
        "training_rank": training["metric"].rank(ascending=False)[mean_ranks.index].values,
        "mean_fes": training["fes"][mean_ranks.index].values,
    })
    summary["significantly_worse"] = summary["holm_p_vs_best"] < common.ALPHA

    common.write_table(summary, options.out, "ranks",
                       caption=f"Mean ranks on {options.dataset}, Friedman p = {friedman_p:.4g}")
    print(summary.to_string(index=False))
    print(f"Blocks: {len(wide)} | Friedman p = {friedman_p:.4g} | alpha = {common.ALPHA}")
    print(f"Best mean rank: {best}")


if __name__ == "__main__":
    main()
