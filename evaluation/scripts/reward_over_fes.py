#!/usr/bin/env python3
"""Learning curves: the training metric over training FEs, one line per setting, spread over seeds."""

import os

import numpy as np

import analysis_common as common


def main():
    parser = common.parser_with_runs(__doc__)
    options = parser.parse_args()

    rows = []
    for run in common.find_runs(options.runs):
        log = common.checkpoint_log(run)
        if log:
            rows.append((run.setting, run.seed, np.array([int(r["fe"]) for r in log]),
                         np.array([float(r["metric"]) for r in log])))
    if not rows:
        raise SystemExit("No ticks of the training metric; runs may be too short for it to start.")

    curves = common.summarize_curves(common.curves_at_grid(rows).dropna())
    os.makedirs(options.out, exist_ok=True)
    curves.to_csv(os.path.join(options.out, "reward_over_fes.csv"), index=False)
    common.plot_curves(curves, os.path.join(options.out, "reward_over_fes.pdf"), "Training return over FEs",
                       "Training fitness evaluations", "Normalized training return")
    print(f"Written to {options.out}/reward_over_fes.pdf and .csv")


if __name__ == "__main__":
    main()
