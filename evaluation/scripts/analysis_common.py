"""Shared utilities of the analysis scripts: finding runs, reading solutions, HV and statistics.

The scripts only read run folders. All objectives are minimized (maximized ones are stored
negated), and infeasible and length-0 states never reach the solution files.
"""

import argparse
import ast
import csv
import json
import os
import re
import warnings
from collections import namedtuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy import stats

ALPHA = 0.05
TIE_TOLERANCE = 1e-6
REFERENCE_POINT = 1.1
PHASE = re.compile(r"iter(\d+)-t(\d+)$")

# setting: experiment folder name. path: <experiment>/<seed>-<ts>.
Run = namedtuple("Run", "path setting seed case")

# Fixed categorical order from the validated default palette. The line style is a second encoding,
# so identity survives grayscale printing and color-vision differences.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"


def find_runs(experiments):
    """Completed seeds (those with runtimes.json) of the given experiment folders."""
    runs = []
    for experiment in experiments:
        experiment = os.path.abspath(experiment)
        if not os.path.isdir(experiment):
            raise SystemExit(f"Not an experiment folder: {experiment}")
        seeds = {}
        for entry in sorted(os.listdir(experiment)):
            path = os.path.join(experiment, entry)
            if not re.match(r"\d+-", entry) or not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, "runtimes.json")):
                print(f"Warning: skipping unfinished run {path}")
                continue
            seed = int(entry.split("-")[0])
            if seed in seeds:
                raise SystemExit(
                    f"Seed {seed} has two finished runs, which would be counted twice:\n"
                    f"  {seeds[seed]}\n  {path}\nRemove the outdated one."
                )
            seeds[seed] = path
            runs.append(
                Run(path, os.path.basename(experiment), seed, os.path.basename(os.path.dirname(experiment)))
            )
    if not runs:
        raise SystemExit(f"No finished runs in {list(experiments)}")
    return runs


def read_solutions(path):
    """Solution file as a list of (FE, points). A file without FE lines is one block at FE 0."""
    blocks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                if not blocks:
                    blocks.append((0, []))
                blocks[-1][1].append(ast.literal_eval(line))
            else:
                blocks.append((int(line), []))
    return [(fe, np.array(points, dtype=np.float64)) for fe, points in blocks]


def final_datasets(run, dataset, folder):
    """Dataset folders of ``dataset`` below ``folder``; ``any`` takes all of them."""
    root = os.path.join(run.path, folder)
    if not os.path.isdir(root):
        return []
    if dataset == "any":
        return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    return [dataset] if os.path.isdir(os.path.join(root, dataset)) else []


def final_curves(run, dataset, folder="final_eval"):
    """{model: [(inference FE, archive after that batch), ...]} below final_eval or sampling_curve."""
    curves = {}
    for name in final_datasets(run, dataset, folder):
        root = os.path.join(run.path, folder, name)
        for model in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
            if model in curves:
                raise SystemExit(f"{model} is in several final datasets of {run.path}; name one with --dataset")
            curves[model] = read_solutions(os.path.join(root, model, "nn-solutions.txt"))
    return curves


def validation_phases(run, variant="stochastic"):
    """{model: {(update, fe): points}} of the stored validation solutions."""
    root = os.path.join(run.path, "eval_solutions", variant)
    if not os.path.isdir(root):
        raise SystemExit(f"{run.path} has no stored validation solutions (--eval-save-solutions)")
    phases = {}
    for model in sorted(os.listdir(root)):
        phases[model] = {}
        for phase in sorted(os.listdir(os.path.join(root, model))):
            match = PHASE.match(phase)
            if match:
                blocks = read_solutions(os.path.join(root, model, phase, "nn-solutions.txt"))
                points = blocks[0][1] if blocks else np.empty((0, 0))
                phases[model][(int(match.group(1)), int(match.group(2)))] = points
    return phases


def training_curves(run):
    """{model: [(training FE, archive), ...]} of the server's running-solutions.txt.

    Only for training on one model, because the server merges the fronts of all its models.
    """
    with open(os.path.join(run.path, "run_args.json")) as f:
        train_path = json.load(f)["run_info"]["datasets"]["train"]
    if not train_path.endswith(".xmi"):
        raise SystemExit(f"The training archive needs training on one model, but {run.path} used {train_path}.")
    logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server-executables", "output", "nn_logs")
    path = os.path.join(logs, run.case, run.setting, os.path.basename(run.path), "running-solutions.txt")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}")
    return {os.path.basename(train_path)[: -len(".xmi")]: read_solutions(path)}


def checkpoint_log(run):
    """Rows of checkpoints.csv: one per tick of the training metric."""
    with open(os.path.join(run.path, "checkpoints.csv"), newline="") as f:
        return list(csv.DictReader(f))


def best_training_metric(run):
    rows = checkpoint_log(run)
    best = [row for row in rows if row["is_best"] == "True"]
    return float(best[0]["metric"]) if best else float("nan")


class Bounds:
    """Per model: min and max of each objective over all solutions of the compared runs."""

    def __init__(self):
        self.low = {}
        self.high = {}

    def add(self, model, points):
        if points.size == 0:
            return
        low, high = points.min(axis=0), points.max(axis=0)
        self.low[model] = np.minimum(self.low[model], low) if model in self.low else low
        self.high[model] = np.maximum(self.high[model], high) if model in self.high else high

    def normalize(self, model, points):
        span = self.high[model] - self.low[model]
        span = np.where(span > 0, span, 1.0)
        return (points - self.low[model]) / span

    def hv(self, model, points):
        if points.size == 0:
            return 0.0
        return hypervolume(self.normalize(model, points))


def hypervolume(normalized_points):
    """HV of normalized objectives with reference point 1.1 in every objective."""
    indicator = HV(ref_point=np.full(normalized_points.shape[1], REFERENCE_POINT))
    return float(indicator(normalized_points))


def primary_objective(case, points):
    """The reported raw objective, in its natural sign."""
    if case == "cra":
        return -points[:, 0]  # Stored negated: this is the CRA-Index.
    if case in ("sp", "sp_uc"):
        return points[:, 0] + points[:, 1]
    return points[:, 0]


def best_primary(case, points):
    values = primary_objective(case, points)
    return float(values.max() if case == "cra" else values.min())


def a12(values_a, values_b, tolerance=TIE_TOLERANCE):
    """Vargha-Delaney A12: probability that a value of A beats a value of B."""
    a = np.asarray(values_a, dtype=np.float64)[:, None]
    b = np.asarray(values_b, dtype=np.float64)[None, :]
    difference = a - b
    wins = (difference > tolerance).sum()
    ties = (np.abs(difference) <= tolerance).sum()
    return float((wins + 0.5 * ties) / difference.size)


def holm(p_values):
    """Holm-adjusted p-values, in the given order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for position, index in enumerate(np.argsort(p_values)):
        running = max(running, (len(p_values) - position) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def rank_within_blocks(table):
    """HV per complete (model, seed) block, and each setting's rank in it; rank 1 is the highest."""
    wide = table.pivot_table(index=["model", "seed"], columns="setting", values="hv", aggfunc="mean")
    incomplete = int(wide.isna().any(axis=1).sum())
    if incomplete:
        print(f"Warning: dropping {incomplete} blocks without a value for every setting")
        wide = wide.dropna()
    ranks = wide.apply(lambda row: stats.rankdata(-row.values), axis=1, result_type="expand")
    ranks.columns = wide.columns
    return wide, ranks


def parser_with_runs(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment folders, one per setting")
    parser.add_argument("--out", required=True, help="Output folder")
    return parser


def write_table(table, directory, name, caption=None):
    """CSV for further use, LaTeX for the thesis."""
    os.makedirs(directory, exist_ok=True)
    table.to_csv(os.path.join(directory, f"{name}.csv"), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # to_latex's future defaults
        latex = table.to_latex(index=False, float_format="%.4f", caption=caption, escape=True)
    with open(os.path.join(directory, f"{name}.tex"), "w") as f:
        f.write(latex)


def read_runtimes(run):
    with open(os.path.join(run.path, "runtimes.json")) as f:
        return json.load(f)


def plot_curves(curves, path, title, x_label, y_label):
    """One line per setting with its spread: columns setting, fe, mean, std."""
    settings = list(dict.fromkeys(curves["setting"]))
    if len(settings) > len(COLORS):
        print(f"Warning: {len(settings)} settings exceed the {len(COLORS)} distinct colors; plot fewer.")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, setting in enumerate(settings):
        part = curves[curves["setting"] == setting]
        color = COLORS[i % len(COLORS)]
        ax.fill_between(part["fe"], part["mean"] - part["std"], part["mean"] + part["std"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(part["fe"], part["mean"], color=color, linewidth=1.6, linestyle=STYLES[i % len(STYLES)],
                solid_capstyle="round", label=setting, marker="o" if len(part) == 1 else None, markersize=6)
    ax.set_xlabel(x_label, color=INK_SECONDARY)
    ax.set_ylabel(y_label, color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    if len(settings) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curves_at_grid(rows, points=200):
    """Average step curves over seeds: rows of (setting, seed, fes, values) on a shared FE grid.

    A run that ended early holds its last value, so seeds that converged sooner still count.
    """
    highest = max(fes[-1] for _, _, fes, _ in rows)
    grid = np.unique(np.linspace(1, highest, points).astype(int))
    table = []
    for setting, seed, fes, values in rows:
        positions = np.searchsorted(fes, grid, side="right") - 1
        at_grid = np.where(positions >= 0, values[np.maximum(positions, 0)], np.nan)
        table += [{"setting": setting, "seed": seed, "fe": fe, "value": value}
                  for fe, value in zip(grid, at_grid)]
    return pd.DataFrame(table)


def summarize_curves(table, value="value"):
    """Mean over seeds per (setting, fe), with std and the number of seeds."""
    grouped = table.groupby(["setting", "fe"])[value]
    return pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(ddof=1).fillna(0.0),
                         "seeds": grouped.count()}).reset_index()
