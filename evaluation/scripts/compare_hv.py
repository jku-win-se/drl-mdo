#!/usr/bin/env python3
"""HV and raw primary objective of settings.

From the final run of each run's best policy, or with --source training from the rollout archive
at the end of training on one model.
"""

import numpy as np
import pandas as pd
from scipy import stats

import analysis_common as common


def collect(runs, dataset, source):
    curves = {
        run: common.training_curves(run) if source == "training" else common.final_curves(run, dataset)
        for run in runs
    }
    bounds = common.Bounds()
    for by_model in curves.values():
        for model, blocks in by_model.items():
            for _, points in blocks:
                bounds.add(model, points)
    rows = []
    for run, by_model in curves.items():
        for model, blocks in by_model.items():
            points = blocks[-1][1] if blocks else np.empty((0, 0))
            rows.append({
                "setting": run.setting, "seed": run.seed, "model": model,
                "hv": bounds.hv(model, points),
                "best_primary": common.best_primary(run.case, points) if points.size else float("nan"),
            })
    return pd.DataFrame(rows)


def summarize(table, model_label, reference):
    """One row per setting: per-seed means over the group's models, tested against the reference."""
    per_seed = table.groupby(["setting", "seed"])[["hv", "best_primary"]].mean().reset_index()
    means = per_seed.groupby("setting")["hv"].mean()
    base = reference if reference in means.index else means.idxmax()
    base_hv = per_seed.loc[per_seed["setting"] == base, "hv"].values
    rows = []
    for setting, part in per_seed.groupby("setting"):
        hv = part["hv"]
        row = {
            "model": model_label, "setting": setting, "mean_hv": hv.mean(),
            "std_hv": hv.std(ddof=1) if len(hv) > 1 else 0.0, "median_hv": hv.median(),
            "iqr_hv": hv.quantile(0.75) - hv.quantile(0.25),
            "mean_best_primary": part["best_primary"].mean(), "seeds": len(hv), "reference": base,
            "a12_vs_reference": float("nan"), "mwu_p_vs_reference": float("nan"),
        }
        if setting != base:
            row["a12_vs_reference"] = common.a12(hv.values, base_hv)
            row["mwu_p_vs_reference"] = stats.mannwhitneyu(hv.values, base_hv, alternative="two-sided").pvalue
        rows.append(row)
    return rows


def main():
    parser = common.parser_with_runs(__doc__)
    parser.add_argument("--dataset", default="test",
                        help="Final-run dataset (val, test), or any to match by model name across datasets")
    parser.add_argument("--source", choices=["final", "training"], default="final",
                        help="training: the rollout archive when training ended; output is named hv_training")
    parser.add_argument("--reference", help="Setting to test against; default: the highest mean HV per row group")
    options = parser.parse_args()

    runs = common.find_runs(options.runs)
    table = collect(runs, options.dataset, options.source)
    if table.empty:
        raise SystemExit(f"No solutions for dataset '{options.dataset}'; did the runs use --test-path (and --final-mode for greedy)?")

    rows = []
    # Means over models are only comparable on models every setting covers.
    settings = table["setting"].nunique()
    coverage = table.groupby("model")["setting"].nunique()
    shared = coverage[coverage == settings].index
    if len(shared) > 0:
        rows += summarize(table[table["model"].isin(shared)], "all", options.reference)
    if len(shared) < table["model"].nunique():
        print(f"Note: 'all' rows use the {len(shared)} of {table['model'].nunique()} models every setting covers")
    for model, part in table.groupby("model"):
        rows += summarize(part, model, options.reference)

    summary = pd.DataFrame(rows)
    summary["significant"] = summary["mwu_p_vs_reference"] < common.ALPHA
    name = "training" if options.source == "training" else options.dataset
    common.write_table(summary, options.out, f"hv_{name}", caption=f"HV on {name}")
    table.to_csv(f"{options.out}/hv_{name}_per_run.csv", index=False)
    print(summary[summary["model"] == "all"].to_string(index=False) if len(shared) else summary.to_string(index=False))


if __name__ == "__main__":
    main()
