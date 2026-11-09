#!/usr/bin/env python3
"""HV over fitness evaluations, one line per setting with its spread over seeds.

Sources:
  training    Archive HV of the server's running-solutions.txt against training FEs, one point
              per rollout. Only for training on a single model, since the server merges fronts.
  sampling    Archive HV of run_policy.sh --mode batches against the number of batches per model.
  validation  HV of each validation phase against training FEs, for runs that validate.
"""

import os

import numpy as np
import pandas as pd

import analysis_common as common


def validation_rows(runs, model_filter):
    phases = {run: common.validation_phases(run) for run in runs}
    bounds = common.Bounds()
    for by_model in phases.values():
        for model, by_phase in by_model.items():
            for points in by_phase.values():
                bounds.add(model, points)
    rows = []
    for run, by_model in phases.items():
        for model, by_phase in by_model.items():
            if model_filter in (None, model):
                for (update, fe), points in by_phase.items():
                    rows.append({"setting": run.setting, "seed": run.seed, "model": model,
                                 "update": update, "fe": fe, "value": bounds.hv(model, points)})
    table = pd.DataFrame(rows)
    if not table.empty:
        # Seeds reach a phase at slightly different FEs. Align on the phase, so seeds average.
        table["fe"] = table.groupby(["setting", "update"])["fe"].transform("mean")
    return table


def bounds_of(curves):
    bounds = common.Bounds()
    for by_model in curves.values():
        for model, blocks in by_model.items():
            for _, points in blocks:
                bounds.add(model, points)
    return bounds


def sampling_rows(runs, dataset, model_filter):
    """Archive HV after each batch. Models differ in FEs per batch, so batches are the shared axis."""
    curves = {run: common.final_curves(run, dataset, "sampling_curve") for run in runs}
    bounds = bounds_of(curves)
    return pd.DataFrame([
        {"setting": run.setting, "seed": run.seed, "model": model, "fe": batch + 1, "value": bounds.hv(model, points)}
        for run, by_model in curves.items()
        for model, blocks in by_model.items()
        if model_filter in (None, model)
        for batch, (_, points) in enumerate(blocks)
    ])


def training_rows(runs):
    """Step curves of each run's single model, read at one shared grid so seeds can be averaged."""
    curves = {run: common.training_curves(run) for run in runs}
    bounds = bounds_of(curves)
    rows = []
    for run, by_model in curves.items():
        for model, blocks in by_model.items():
            fes = np.array([fe for fe, _ in blocks])
            rows.append((f"{run.setting}|{model}", run.seed, fes,
                         np.array([bounds.hv(model, points) for _, points in blocks])))
    if not rows:
        return pd.DataFrame()
    table = common.curves_at_grid(rows).fillna(0.0)
    table[["setting", "model"]] = table["setting"].str.split("|", expand=True)
    return table


def main():
    parser = common.parser_with_runs(__doc__)
    parser.add_argument("--source", choices=["training", "sampling", "validation"], required=True)
    parser.add_argument("--dataset", default="any", help="With --source sampling: the label, or any")
    parser.add_argument("--model", help="One model only; default: mean over models")
    options = parser.parse_args()

    runs = common.find_runs(options.runs)
    if options.source == "validation":
        table = validation_rows(runs, options.model)
    elif options.source == "sampling":
        table = sampling_rows(runs, options.dataset, options.model)
    else:
        table = training_rows(runs)
    if table.empty:
        raise SystemExit(f"No solutions for source '{options.source}'")

    # Mean over models first, then the spread over seeds at each FE.
    per_seed = table.groupby(["setting", "seed", "fe"])["value"].mean().reset_index()
    curves = common.summarize_curves(per_seed)

    os.makedirs(options.out, exist_ok=True)
    stem = f"hv_over_fes_{options.source}" + (f"_{options.model}" if options.model else "")
    curves.to_csv(os.path.join(options.out, f"{stem}.csv"), index=False)
    x_label = "Batches per model" if options.source == "sampling" else "Training fitness evaluations"
    title = f"HV over FEs ({options.source}), {options.model or 'mean over models'}"
    common.plot_curves(curves, os.path.join(options.out, f"{stem}.pdf"), title, x_label, "Normalized hypervolume")
    print(f"Written to {options.out}/{stem}.pdf and .csv")


if __name__ == "__main__":
    main()
