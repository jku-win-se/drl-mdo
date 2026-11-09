#!/usr/bin/env python3
"""For runs that validate: per seed, the validation phase with the best HV.

Reads the stored validation solutions. With --keep-checkpoints all during training, the checkpoint
nearest that FE can then be rerun on any dataset with run_policy.sh.
"""

import numpy as np
import pandas as pd

import analysis_common as common


def main():
    parser = common.parser_with_runs(__doc__)
    options = parser.parse_args()

    rows = []
    for run in common.find_runs(options.runs):
        phases = common.validation_phases(run)
        bounds = common.Bounds()
        for model, by_phase in phases.items():
            for points in by_phase.values():
                bounds.add(model, points)
        scores = {}
        for model, by_phase in phases.items():
            for (update, fe), points in by_phase.items():
                scores.setdefault((update, fe), []).append(bounds.hv(model, points))
        means = {phase: float(np.mean(values)) for phase, values in scores.items()}
        by_hv = max(means, key=means.get)
        rows.append({
            "setting": run.setting, "seed": run.seed,
            "best_hv_update": by_hv[0], "best_hv_fe": by_hv[1], "best_hv": means[by_hv],
        })
    table = pd.DataFrame(rows)
    common.write_table(table, options.out, "best_by_hv", caption="Best validation phase by HV")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
