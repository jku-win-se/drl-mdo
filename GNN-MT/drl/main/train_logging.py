"""What a training run reports: the iteration summary, W&B, TensorBoard and its log files."""

import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from drl.env.env_util import pop_mask_seconds
from drl.utils.episode_stats import metric_name

f_logger = logging.getLogger(__name__)

# Collected per minibatch, averaged and cleared per iteration.
UPDATE_METRICS = (
    "actor_losses",
    "critic_losses",
    "entropy",
    "entropy_rule",
    "entropy_node",
    "entropy_val",
    "entropy_rule_norm",
    "entropy_node_norm",
    "entropy_val_norm",
    "approx_kl",
    "approx_kl_k3",
    "approx_kl_rule",
    "approx_kl_node",
    "approx_kl_val",
    "clip_fraction",
    "ratio_min",
    "ratio_max",
    "adv_mean",
    "adv_std",
    "grad_norm",
    "grad_norm_critic",
)


def _is_nonfinite(value):
    return isinstance(value, float) and not math.isfinite(value)


def _tensorboard_writer(args):
    """Below <project>/runs, named after the W&B group and the checkpoint a run started from."""
    project_root = Path(__file__).parent.parent.parent
    if "group" not in args.wandb_config:
        return SummaryWriter(log_dir=str(project_root / "runs" / f"-{args.seed}"))
    log_name = args.wandb_config["group"]
    if args.ac_model:
        log_name = f"{log_name}_{os.path.splitext(os.path.basename(args.ac_model))[0]}"
    return SummaryWriter(log_dir=str(project_root / "runs" / log_name / f"-{args.seed}"))


class TrainLogger:
    """Collects a run's metrics in ``metrics``, and reports them per iteration."""

    def __init__(self, args, env, num_envs, episode_stats, eval_variants=()):
        self.args = args
        self.env = env
        self.num_envs = num_envs
        self.episode_stats = episode_stats
        self.eval_variants = eval_variants
        self.writer = None
        if args.verbose:
            self.writer = _tensorboard_writer(args)
            print(f"TensorBoard logs will be saved to: {self.writer.log_dir}")

        self.metrics = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "e_so_far": 0,  # episodes so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "avg_lens": [],  # episodic lengths in batch
            "avg_rews": [],  # pooled episodic returns per iteration
            "avg_macro_rews": [],  # per-env-mean (macro) returns per iteration
            "avg_actor_losses": [],  # all actor losses
            "value_bin_indices": None,
            "explained_variance": 0.0,
            "epochs_run": 0,
            "kl_early_stop": False,
            "lr": 0.0,
            "value_head_log_std": 0.0,
            "skipped_invalid_steps": 0,
            "skipped_invalid_nodes": 0,
            "invalid_action_rate": float("nan"),
            "explained_variance_by_instance": {},
            "batch_returns_mean": float("nan"),
            "batch_returns_std": float("nan"),
            "es_metric": float("nan"),
            "es_ticked": 0,
            "time_env_wait": 0.0,
            "time_inference": 0.0,
            "time_decode": 0.0,
            **{key: [] for key in UPDATE_METRICS},
        }
        # Logged with the next summary (or right away for the initial/final phase).
        self.pending_eval_log = {}
        self._last_row = {}
        self._checkpoint_rows = []

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def record_minibatch(self, **values):
        """Add per-minibatch values, averaged by the next summary."""
        for key, value in values.items():
            self.metrics[key].append(value)

    def add_eval_log(self, log):
        self.pending_eval_log.update(log)

    def flush_eval_log(self, iteration, timesteps):
        """Log a pending eval phase on its own, outside an iteration summary."""
        log = self._take_eval_log()
        if not log:
            return
        self._log_eval_timers(log)
        if self.args.use_wandb:
            wandb.log({**log, "iteration": iteration, "timesteps": timesteps})
        if self.writer is not None:
            for key, value in log.items():
                self.writer.add_scalar(f"ppo/{key}", value, iteration)

    def log_histograms(self, ac, iteration):
        if self.writer is None:
            return
        for name, param in ac.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param, iteration)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, iteration)

    def summarize_iteration(self):
        """Print and publish the most recent iteration, then clear its per-minibatch metrics."""
        averages = self._update_averages()
        eval_log = self._take_eval_log()
        self._print_summary(averages, eval_log)
        row = self._summary_row(averages, eval_log)
        self._last_row = row
        if self.args.use_wandb:
            wandb.log(row)
        if self.writer is not None:
            for key, val in row.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self.writer.add_scalar(f"ppo/{key}", val, self.metrics["i_so_far"])
        self._print_history(averages)
        self.metrics["batch_lens"] = []
        self.metrics["value_bin_indices"] = None
        for key in UPDATE_METRICS:
            self.metrics[key] = []

    def write_train_log(self, iteration, fes, *stats):
        """One line of ``train_log.jsonl`` from the last summary and the given stats."""
        row = {
            "iteration": iteration,
            "fe": fes,
            "rollout_seconds": self.metrics.get("rollout_seconds"),
            "update_seconds": self.metrics.get("update_seconds"),
            **self._last_row,
        }
        for extra in stats:
            row.update(extra)
        with open(os.path.join(self.args.model_path, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def add_checkpoint_row(self, update_step, fes, file_name, metric, mean_return, std_return, improved):
        """Rewrite ``checkpoints.csv`` with one more tick of the training metric.

        The returns come from the moving policy on training models, so they describe training
        rather than a separate evaluation of the stored weights.
        """
        self._checkpoint_rows.append(
            {
                "update": update_step,
                "fe": fes,
                "file": file_name or "",
                "metric": metric,
                "mean_return": mean_return,
                "std_return": std_return,
                "improved": improved,
            }
        )
        best_row = None
        for row in self._checkpoint_rows:
            if row["improved"]:
                best_row = row
        with open(os.path.join(self.args.model_path, "checkpoints.csv"), "w", newline="") as f:
            columns = ["update", "fe", "file", "metric", "mean_return", "std_return", "is_best"]
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self._checkpoint_rows:
                writer.writerow({**row, "is_best": row is best_row})

    def write_runtimes(self, training_seconds, validation_seconds, iterations, fes, final_eval):
        """Written last, so its presence marks the seed as complete."""
        runtimes = {
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "mask_seconds": pop_mask_seconds(),
            "iterations": iterations,
            "fes": fes,
            "final_eval": final_eval,
        }
        path = os.path.join(self.args.model_path, "runtimes.json")
        with open(path + ".tmp", "w") as f:
            json.dump(runtimes, f, indent=2)
        os.replace(path + ".tmp", path)

    def _take_eval_log(self):
        """Pending eval values without non-finite entries; clears the pending log."""
        log = {key: value for key, value in self.pending_eval_log.items() if not _is_nonfinite(value)}
        self.pending_eval_log = {}
        return log

    def _log_eval_timers(self, log):
        if "time_eval" not in log:
            return
        parts = [f"total={log['time_eval']:.3f}s"]
        for variant in self.eval_variants:
            if f"time_eval_{variant}_env_wait" in log:
                parts.append(
                    f"{variant}: env_wait={log[f'time_eval_{variant}_env_wait']:.3f}s "
                    f"inference={log[f'time_eval_{variant}_inference']:.3f}s "
                    f"decode={log[f'time_eval_{variant}_decode']:.3f}s"
                )
        f_logger.info("Eval timers: " + " | ".join(parts))

    def _update_averages(self):
        """Per-iteration means of what the update collected per minibatch."""
        averages = {}
        for key in UPDATE_METRICS:
            vals = self.metrics[key]
            averages[key] = (
                float(
                    np.mean(
                        [v.cpu().float().mean().item() if torch.is_tensor(v) else float(v) for v in vals]
                    )
                )
                if vals
                else 0.0
            )
        batch_lens = self.metrics["batch_lens"]
        averages["ep_len"] = float(np.mean(batch_lens)) if batch_lens else 0.0
        return averages

    def _grouped_by_instance(self):
        """Per-model stats when envs share models; per-env stats otherwise."""
        if len(self.env.instance_names) < self.num_envs:
            return "inst", self.env.instance_names, self.episode_stats.get_last_rollout_instance_stats()
        return "env", [str(i) for i in range(self.num_envs)], self.episode_stats.get_last_rollout_env_stats()

    def _print_summary(self, avg, eval_log):
        m = self.metrics
        delta_t = m["delta_t"]
        m["delta_t"] = time.time_ns()
        delta_t = str(round((m["delta_t"] - delta_t) / 1e9, 2))
        stats = self.episode_stats
        buffer_stats = stats.get_stats()
        recent_stats = stats.get_recent_stats(50)
        explained_var = float(m.get("explained_variance", 0.0) or 0.0)

        f_logger.info(f"-------------------- Iteration #{m['i_so_far']} --------------------")
        f_logger.info(f"Batch Episodes: {len(m['batch_lens'])}")
        f_logger.info(f"Average Episodic Length: {avg['ep_len']:.2f}")
        f_logger.info(f"Average Episodic Return: {stats.get_last_rollout_pooled_return():.3f}")
        f_logger.info(f"Macro Return (per-env mean): {stats.get_last_rollout_macro_return():.3f}")
        f_logger.info(f"Per-step return: {stats.get_last_rollout_per_step_return():.5f}")
        self._print_grouped_returns(*self._grouped_by_instance())
        ev_by_instance = m.get("explained_variance_by_instance") or {}
        finite_ev = [v for v in ev_by_instance.values() if isinstance(v, float) and math.isfinite(v)]
        if finite_ev:
            f_logger.info(
                "  per-instance explained var: "
                f"min={min(finite_ev):.3f} median={float(np.median(finite_ev)):.3f}"
            )
        f_logger.info(f"Rolling Buffer Stats (last {buffer_stats['num_episodes']} episodes):")
        f_logger.info(f"  Mean Return: {buffer_stats['mean_return']:.3f} ± {buffer_stats['std_return']:.3f}")
        f_logger.info(f"Recent 50 Episodes Mean Return: {recent_stats['mean_return']:.3f}")
        f_logger.info(f"Total Episodes Completed: {buffer_stats['total_episodes']}")
        f_logger.info(f"Average Loss: {avg['actor_losses']:.5f}")
        f_logger.info(
            f"PPO diagnostics: kl={avg['approx_kl']:.5f} kl_k3={avg['approx_kl_k3']:.5f} "
            f"clip_frac={avg['clip_fraction']:.3f} explained_var={explained_var:.3f} "
            f"epochs={int(m.get('epochs_run', 0) or 0)} kl_stop={bool(m.get('kl_early_stop', False))} "
            f"lr={float(m.get('lr', 0.0) or 0.0):.2e}"
        )
        f_logger.info(
            f"  factor KL: rule={avg['approx_kl_rule']:.5f} node={avg['approx_kl_node']:.5f} "
            f"val={avg['approx_kl_val']:.5f}"
        )
        f_logger.info(
            f"  entropy: joint={avg['entropy']:.4f} rule={avg['entropy_rule']:.4f} "
            f"node={avg['entropy_node']:.4f} val={avg['entropy_val']:.4f} "
            f"(norm rule={avg['entropy_rule_norm']:.4f} node={avg['entropy_node_norm']:.4f} "
            f"val={avg['entropy_val_norm']:.4f})"
        )
        f_logger.info(
            f"  ratio=[{avg['ratio_min']:.3f}, {avg['ratio_max']:.3f}] "
            f"adv_mean={avg['adv_mean']:.4f} adv_std={avg['adv_std']:.4f} "
            f"grad_norm={avg['grad_norm']:.4f} grad_norm_critic={avg['grad_norm_critic']:.4f}"
        )
        f_logger.info(f"Timesteps So Far: {m['t_so_far']}")
        f_logger.info(f"Episodes So Far: {m['e_so_far']}")
        f_logger.info(
            f"Skipped invalid steps: {m.get('skipped_invalid_steps', 0)} "
            f"(invalid_action_rate={m.get('invalid_action_rate', float('nan'))})"
        )
        f_logger.info(f"Skipped invalid nodes: {m.get('skipped_invalid_nodes', 0)}")
        f_logger.info(
            f"Early stopping: es_metric={m.get('es_metric', float('nan'))} "
            f"es_ticked={int(m.get('es_ticked', 0))} "
            f"fresh_instances={stats.fresh_instance_count()}"
        )
        f_logger.info(
            "Rollout timers: env_wait={:.3f}s inference={:.3f}s decode={:.3f}s".format(
                float(m.get("time_env_wait", 0.0)),
                float(m.get("time_inference", 0.0)),
                float(m.get("time_decode", 0.0)),
            )
        )
        self._log_eval_timers(eval_log)
        f_logger.info(f"Iteration took: {delta_t} secs")
        f_logger.info("------------------------------------------------------")

    @staticmethod
    def _print_grouped_returns(label, names, stats):
        for index, name in enumerate(names):
            mean = stats["means"][index]
            count = int(stats["counts"][index])
            if np.isnan(mean):
                f_logger.info(f"  {label} {name}: no completed episodes")
                continue
            length = stats["mean_lengths"][index]
            length_text = f"{length:.1f}" if not np.isnan(length) else "n/a"
            f_logger.info(
                f"  {label} {name}: {mean:.3f} ± {stats['stds'][index]:.3f} "
                f"(n={count} episodes, mean len={length_text})"
            )

    def _summary_row(self, avg, eval_log):
        """One iteration's metrics for W&B, TensorBoard and ``train_log.jsonl``."""
        m = self.metrics
        stats = self.episode_stats
        buffer_stats = stats.get_stats()
        recent_stats = stats.get_recent_stats(50)
        row = {
            "iteration": m["i_so_far"],
            "timesteps": m["t_so_far"],
            "avg_episode_length": float(avg["ep_len"]),
            "avg_episode_return": float(stats.get_last_rollout_pooled_return()),
            "macro_episode_return": float(stats.get_last_rollout_macro_return()),
            "per_step_return": float(stats.get_last_rollout_per_step_return()),
            "avg_actor_loss": float(avg["actor_losses"]),
            "avg_entropy": float(avg["entropy"]),
            "avg_entropy_rule": float(avg["entropy_rule"]),
            "avg_entropy_node": float(avg["entropy_node"]),
            "avg_entropy_val": float(avg["entropy_val"]),
            "avg_entropy_rule_norm": float(avg["entropy_rule_norm"]),
            "avg_entropy_node_norm": float(avg["entropy_node_norm"]),
            "avg_entropy_val_norm": float(avg["entropy_val_norm"]),
            "approx_kl": float(avg["approx_kl"]),
            "approx_kl_k3": float(avg["approx_kl_k3"]),
            "approx_kl_rule": float(avg["approx_kl_rule"]),
            "approx_kl_node": float(avg["approx_kl_node"]),
            "approx_kl_val": float(avg["approx_kl_val"]),
            "clip_fraction": float(avg["clip_fraction"]),
            "ratio_min": float(avg["ratio_min"]),
            "ratio_max": float(avg["ratio_max"]),
            "adv_mean": float(avg["adv_mean"]),
            "adv_std": float(avg["adv_std"]),
            "grad_norm": float(avg["grad_norm"]),
            "grad_norm_critic": float(avg["grad_norm_critic"]),
            "explained_variance": float(m.get("explained_variance", 0.0) or 0.0),
            "epochs_run": int(m.get("epochs_run", 0) or 0),
            "kl_early_stop": int(bool(m.get("kl_early_stop", False))),
            "lr": float(m.get("lr", 0.0) or 0.0),
            "skipped_invalid_nodes": int(m.get("skipped_invalid_nodes", 0) or 0),
            "avg_critic_loss": float(avg["critic_losses"]),
            "std_parameter": float(m["value_head_log_std"]),
            # Rolling buffer stats
            "buffer_mean_return": buffer_stats["mean_return"],
            "buffer_std_return": buffer_stats["std_return"],
            "buffer_min_return": buffer_stats["min_return"],
            "buffer_max_return": buffer_stats["max_return"],
            "buffer_mean_length": buffer_stats["mean_length"],
            "recent_50_mean_return": recent_stats["mean_return"],
            "total_episodes_completed": buffer_stats["total_episodes"],
            "batch_episodes_count": len(m["batch_lens"]),
            "time_env_wait": float(m.get("time_env_wait", 0.0)),
            "time_inference": float(m.get("time_inference", 0.0)),
            "time_decode": float(m.get("time_decode", 0.0)),
        }
        self._add_value_bins(row)
        self._add_grouped_returns(row, *self._grouped_by_instance())
        for instance_id, name in enumerate(self.env.instance_names):
            scale = stats.instance_return_scale[instance_id]
            if scale is not None:
                row[f"return_scale_inst_{metric_name(name)}"] = float(scale)
        for name, ev in (m.get("explained_variance_by_instance") or {}).items():
            if isinstance(ev, float) and math.isfinite(ev):
                row[f"ev_inst_{metric_name(name)}"] = ev
        for key in ("batch_returns_mean", "batch_returns_std", "invalid_action_rate", "es_metric"):
            value = m.get(key, float("nan"))
            if isinstance(value, float) and math.isfinite(value):
                row[key] = value
        row["es_ticked"] = int(m.get("es_ticked", 0))
        row.update(eval_log)
        row["fresh_instance_count"] = int(stats.fresh_instance_count())
        return {key: value for key, value in row.items() if not _is_nonfinite(value)}

    def _add_value_bins(self, row):
        bin_idxs = self.metrics.get("value_bin_indices")
        if bin_idxs is None:
            return
        if torch.is_tensor(bin_idxs):
            bin_np = bin_idxs.detach().cpu().float().numpy()
        else:
            bin_np = np.asarray(bin_idxs, dtype=np.float32)
        if bin_np.size > 0:
            row["value_bin_mean"] = float(np.mean(bin_np))
            if self.args.use_wandb:
                row["value_bin_indices"] = wandb.Histogram(bin_np)

    @staticmethod
    def _add_grouped_returns(row, label, names, stats):
        for index, name in enumerate(names):
            key = metric_name(name) if label == "inst" else str(name)
            mean = stats["means"][index]
            length = stats["mean_lengths"][index]
            if np.isfinite(mean):
                row[f"return_{label}_{key}"] = float(mean)
            if np.isfinite(length):
                row[f"length_{label}_{key}"] = float(length)
            row[f"episodes_{label}_{key}"] = int(stats["counts"][index])

    def _print_history(self, avg):
        m = self.metrics
        m["avg_lens"].append(float(avg["ep_len"]))
        m["avg_rews"].append(float(self.episode_stats.get_last_rollout_pooled_return()))
        m["avg_macro_rews"].append(float(self.episode_stats.get_last_rollout_macro_return()))
        m["avg_actor_losses"].append(float(avg["actor_losses"]))
        f_logger.info("Mean episodic rewards (pooled):")
        f_logger.info(m["avg_rews"])
        f_logger.info("Mean episodic rewards (macro, per-env):")
        f_logger.info(m["avg_macro_rews"])

        for variant, returns in self.episode_stats.eval_mean_returns.items():
            f_logger.info(f"Evaluation ({variant}): Mean episodic rewards:")
            f_logger.info([round(x, 3) for x in returns])
