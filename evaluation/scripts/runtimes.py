#!/usr/bin/env python3
"""Mean training and inference times across runs, per setting and model, for break-even analysis."""

import pandas as pd

import analysis_common as common


def main():
    parser = common.parser_with_runs(__doc__)
    options = parser.parse_args()

    rows = []
    for run in common.find_runs(options.runs):
        times = common.read_runtimes(run)
        base = {
            "setting": run.setting, "seed": run.seed, "training_seconds": times["training_seconds"],
            "validation_seconds": times["validation_seconds"], "mask_seconds": times["mask_seconds"],
            "training_fes": times["fes"],
        }
        finals = [(d, m, v) for d, ms in times["final_eval"].items() for m, v in ms.items()]
        if not finals:
            rows.append({**base, "dataset": "", "model": ""})
        for dataset, model, values in finals:
            rows.append({
                **base, "dataset": dataset, "model": model, "inference_seconds": values["seconds"],
                "inference_seconds_per_episode": values["seconds"] / max(values["episodes"], 1),
                "inference_episodes": values["episodes"], "inference_fes": values["fes"],
            })
    table = pd.DataFrame(rows)

    columns = [c for c in table.columns if c not in ("setting", "seed", "dataset", "model")]
    per_model = table.groupby(["setting", "dataset", "model"], dropna=False)
    summary = per_model[columns].mean().add_prefix("mean_").reset_index()
    summary.insert(3, "runs", per_model["seed"].nunique().values)
    # One row per setting over everything: seeds with generated data share no model names.
    per_run = table.groupby(["setting", "seed"])[columns].mean().reset_index()
    per_setting = per_run.groupby("setting")[columns].mean().add_prefix("mean_").reset_index()
    per_setting.insert(1, "dataset", "")
    per_setting.insert(2, "model", "all")
    per_setting.insert(3, "runs", per_run.groupby("setting")["seed"].nunique().values)
    summary = pd.concat([per_setting, summary[summary["model"] != ""]], ignore_index=True)
    common.write_table(summary, options.out, "runtimes", caption="Mean runtimes across runs")
    print(per_setting[["setting", "runs", "mean_training_seconds", "mean_validation_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
