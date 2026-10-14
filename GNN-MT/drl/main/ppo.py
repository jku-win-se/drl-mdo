"""
The file contains the PPO class to train with.
NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
        It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import copy
import warnings
from networkx import nodes
import procbridge as pb
import math
import torch.nn.functional as F
from pathlib import Path

import numpy as np

import json
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.data import Batch
import wandb
from torch.utils.tensorboard import SummaryWriter
import os
from drl.env.RemEnv import SubprocEnv
from drl.env.batched_env import BatchedRemoteEnv
from drl.env.time_feature import TIME_FEAT_DIM
from drl.main.inference_sampler import (
    INFERENCE_MODES,
    FixedEpisodeCount,
    InferenceSampler,
    OneEpisodePerEnv,
    UntilNoImprovement,
)
from utils.seed_util import seed_everything
from nqdm import nqdm
import logging
from collections import namedtuple
from drl.utils.episode_stats import EpisodeStats
from drl.utils.pallete import PalleteEncoder
from drl.utils.running_mean_std import RunningMeanStd
from drl.neural_model.early_stopping import EarlyStopping
from drl.neural_model.gnn_predictor import LOGIT_MASK

TFAction = namedtuple("TForceAction", ["action", "nodes", "nodes_dim_mask", "values"])

f_logger = logging.getLogger(__name__)

eps_log_clip = 1e-6
EVAL_VARIANTS = {
    "stochastic": ("stochastic",),
    "greedy": ("greedy",),
    "both": ("stochastic", "greedy"),
}
NUM_LOGP_FACTORS = 3  # rule, node, value
REWARD_NORM_EPS = 1e-8


def scale_reward(normalizer, discounted_return, raw_reward, reward_clip, update_stats):
    """Divide a reward by the instance return scale, in SB3 VecNormalize order.

    The running variance is updated with the discounted return first. The raw
    reward is then divided by sqrt(variance + eps) and clipped. It is not
    mean-centered.
    """
    if update_stats:
        normalizer.update(np.array([discounted_return], dtype=np.float64))
    variance = float(np.asarray(normalizer.var, dtype=np.float64).reshape(-1)[0])
    scaled = float(raw_reward) / math.sqrt(variance + REWARD_NORM_EPS)
    if scaled > reward_clip:
        return float(reward_clip)
    if scaled < -reward_clip:
        return float(-reward_clip)
    return scaled


def explained_variance(returns, values):
    """1 - Var(returns - values) / Var(returns). NaN when the variance is undefined."""
    if returns.numel() < 2:
        return float("nan")
    variance = returns.var()
    if variance.item() == 0:
        return float("nan")
    residual_variance = (returns - values).var()
    return (1.0 - residual_variance / (variance + 1e-8)).item()


def _metric_name(name):
    base = os.path.basename(str(name))
    cleaned = []
    for character in base:
        if character.isalnum() or character in ("-", "_"):
            cleaned.append(character)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "instance"


def _as_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.is_cuda else obj
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(x) for x in obj)
    elif isinstance(obj, list):
        return [move_to_cpu(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, gnn_actor_critic, args, pallete, run_id=None):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
            policy_class - the policy class to use for our actor/critic networks.
            env - the environment to train on.
            hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
            None
        """

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(args.rl_hyperparams)
        print("Using normalize_rewards:", self.normalize_rewards)
        self.args = args
        self.pallete = pallete

        with open(os.path.join(self.args.model_path, "pallete.json"), "w") as f:

            json.dump(self.pallete, f, cls=PalleteEncoder)
        # Extract environment information
        self.num_envs = args.rl_hyperparams["num_envs"]
        if self.es_tick_episodes is None:
            self.es_tick_episodes = self.num_envs
        self._apply_saved_time_feature_config()

        # Track cumulative episode rewards and lengths across rollout boundaries (SB3 style)
        # These accumulate rewards/steps for in-progress episodes spanning multiple rollouts
        self._current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        # Track discounted returns for each environment (like OpenAI Baselines)
        self.returns = np.zeros(self.num_envs)
        self._reward_norm_updates = 0

        conn = pb.Client("127.0.0.1", args.port)
        conn.request("resetInstance", str(run_id))

        port = getattr(args, "port", 8000)
        transport = getattr(args, "env_transport", "batched")
        n_groups = getattr(args, "env_groups", 1)
        env_kwargs = {
            "time_feature": self.time_feature,
            "time_scale": self.time_scale,
        }
        if transport == "subproc":
            self.env = SubprocEnv(
                self.num_envs, args.metamodel, pallete, port=port, **env_kwargs
            )
        else:
            self.env = BatchedRemoteEnv(
                self.num_envs, pallete, port=port, groups=n_groups, **env_kwargs
            )

        # Logged with the next summary (or right away for the initial/final phase).
        self.pending_eval_log = {}
        # Per variant: env index -> |return| of the first evaluation, frozen.
        self._eval_return_scales = {}
        # Eval variant whose normalized return drives early stopping; None = rollouts.
        self.es_eval_variant = (
            None if args.rl_es_metric == "rollout" else args.rl_es_metric[len("eval_"):]
        )
        if self.es_eval_variant is not None and not self.args.rl_evaluate:
            raise ValueError(f"--rl_es_metric {args.rl_es_metric} needs --rl_evaluate")
        if self.args.rl_evaluate:
            if transport == "subproc":
                raise ValueError("--rl_evaluate requires --env_transport batched")
            if args.rl_eval_interval < 1:
                raise ValueError("--rl_eval_interval must be at least 1")
            self.eval_interval = args.rl_eval_interval
            self.eval_variants = EVAL_VARIANTS[args.rl_eval_mode]
            if (
                self.es_eval_variant is not None
                and self.es_eval_variant not in self.eval_variants
            ):
                raise ValueError(
                    f"--rl_es_metric {args.rl_es_metric} needs --rl_eval_mode "
                    f"{self.es_eval_variant} or both (got {args.rl_eval_mode})"
                )
            print("Setting up evaluation environment (port 8001)...")
            conn_eval = pb.Client("127.0.0.1", 8001)
            conn_eval.request("resetInstance", os.path.join(str(run_id), "eval"))
            num_test_models = conn_eval.request("numInstances")
            self.eval_env = BatchedRemoteEnv(
                num_test_models, pallete, port=8001, groups=n_groups, **env_kwargs
            )

        self.obs = self.env.reset()
        self._resolve_rollout_length()
        self.time_scale = float(self.env.time_scale)
        self.time_feat_dim = TIME_FEAT_DIM if self.time_feature else 0
        print(
            f"Resolved n_steps={self.n_steps} "
            f"(batch={self.timesteps_per_batch}, max horizon={self.max_horizon})"
        )
        if self.time_feature:
            print(f"Time feature on, time_scale={self.time_scale}")
        else:
            print("Time feature off")

        self.episode_stats = EpisodeStats(
            buffer_size=50,
            num_envs=self.num_envs,
            instance_ids=self.env.instance_ids,
            instance_names=self.env.instance_names,
            es_warmup_episodes=self.es_warmup_episodes,
        )
        self._instance_reward_normalizers = [
            RunningMeanStd() for _ in range(len(self.env.instance_names))
        ]
        self._write_model_config()

        # Create SummaryWriter with custom log directory name
        if self.args.verbose:
            if hasattr(self.args, "wandb_config") and "group" in self.args.wandb_config:
                log_name = self.args.wandb_config["group"]
                if hasattr(args, "ac_model") and args.ac_model:
                    model_basename = os.path.splitext(os.path.basename(args.ac_model))[
                        0
                    ]
                    log_name = f"{log_name}_{model_basename}"
                # Make log directory relative to project root (where main script is located)
                project_root = Path(__file__).parent.parent.parent
                log_dir = project_root / "runs" / log_name / f"-{args.seed}"
                self.writer = SummaryWriter(log_dir=str(log_dir))
                print(f"TensorBoard logs will be saved to: {log_dir}")
            else:
                # For default case, also make it relative to project root
                project_root = Path(__file__).parent.parent.parent
                self.writer = SummaryWriter(
                    log_dir=str(project_root / "runs" / f"-{args.seed}")
                )
                print(f"TensorBoard logs will be saved to: {self.writer.log_dir}")

        # Initialize actor and critic networks
        self.ac = gnn_actor_critic(
            pallete,
            args.hidden_dim,
            len(pallete.units),
            **self._ac_kwargs(),
        ).to(args.device)
        n_params = sum(p.numel() for p in self.ac.parameters())
        print(f"Actor-critic parameters: {n_params:,}")

        if args.no_nn_updates:
            for param in self.ac.parameters():
                param.requires_grad = False

        # Initialize optimizers for actor and critic
        if args.rl_hyperparams["encoder"] == "shared":
            self.ac_optim = Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
            decay_steps = int(args.rl_tsteps / math.ceil(self.timesteps_per_batch))
            if decay_steps < 1:
                decay_steps = 1
            print(f"Learning rate will be decayed over {decay_steps} steps")
            self.lr_scheduler = LinearLR(
                self.ac_optim, start_factor=1.0, end_factor=0.1, total_iters=decay_steps
            )

        elif args.rl_hyperparams["encoder"] == "separate":
            self.a_optim = Adam(self.ac.policy_parameters(), lr=self.lr, eps=1e-5)
            self.c_optim = Adam(self.ac.value_parameters(), lr=self.lr, eps=1e-5)

        # Initialize early stopping for RL training
        if args.rl_hyperparams["encoder"] == "shared":
            optimizer = self.ac_optim
        else:
            optimizer = {"actor": self.a_optim, "critic": self.c_optim}

        self.early_stopping = EarlyStopping(
            optimizer=optimizer,
            model=self.ac,
            path=os.path.join(self.args.model_path, "best_model.pth"),
            patience=self.es_patience,
            min_delta=self.es_min_delta,
            min_timesteps=self.es_min_timesteps,
            min_delta_rel=self.es_min_delta_rel,
        )

        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "e_so_far": 0,  # episodes so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "avg_lens": [],  # episodic lengths in batch
            "avg_test_rews": [],  # episodic returns on test set
            "avg_rews": [],  # pooled episodic returns per iteration
            "avg_macro_rews": [],  # per-env-mean (macro) returns per iteration
            "avg_mean_rews": [],  # all batch mean rewards
            "actor_losses": [],  # losses of actor network in current iteration
            "critic_losses": [],
            "avg_critic_losses": [],
            "avg_actor_losses": [],  # all actor losses
            "entropy": [],
            "entropy_rule": [],
            "entropy_node": [],
            "entropy_val": [],
            "entropy_rule_norm": [],
            "entropy_node_norm": [],
            "entropy_val_norm": [],
            "value_bin_indices": None,
            "approx_kl": [],
            "approx_kl_k3": [],
            "approx_kl_rule": [],
            "approx_kl_node": [],
            "approx_kl_val": [],
            "clip_fraction": [],
            "ratio_min": [],
            "ratio_max": [],
            "adv_mean": [],
            "adv_std": [],
            "grad_norm": [],
            "grad_norm_critic": [],
            "explained_variance": 0.0,
            "epochs_run": 0,
            "kl_early_stop": False,
            "lr": 0.0,
            "test_model_bov": {},  # BOV of the model on test set
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
        }

        self.time_env_wait = 0.0
        self.time_inference = 0.0
        self.time_decode = 0.0

        # Initialize NaN debugging tools (set debug_nan=True to enable)
        self.debug_nan = getattr(args, "debug_nan", False)

    def inference_sampling(
        self, mode, seed, nn_class, num_solutions=None, converge_patience=None
    ):
        """Sample solutions with the checkpoint ``args.ac_model``.

        Modes (``INFERENCE_MODES``):
        - ``one-per-model``: one stochastic episode per env, i.e. per model.
        - ``fixed-count``: exactly ``num_solutions`` episodes; their Pareto set is written.
        - ``until-converged``: episodes until ``converge_patience`` episodes in a row
          (default ``num_envs``) leave the Pareto set unchanged.
        - ``max-likelihood``: one greedy episode per model, with its transformation sequence.
        """
        greedy = mode == "max-likelihood"
        schedule = self.inference_schedule(mode, num_solutions, converge_patience)
        self.load_inference_policy(nn_class, greedy=greedy)

        print(f"Running {mode} inference (seed={seed})")
        seed_everything(seed)
        result = InferenceSampler(self, schedule, greedy=greedy).run()
        print(result.describe())

        if isinstance(schedule, OneEpisodePerEnv):
            # Greedy episodes are deterministic, so the seed does not name the run.
            run_label = "greedy" if greedy else str(seed)
            self.write_per_model_results(
                result,
                self.env,
                lambda model_name: self.inference_output_dir(run_label, model_name),
                seed,
                {"checkpoint": self.args.ac_model},
            )
        else:
            self.write_solutions_to_dir(
                seed,
                result.pareto_front.tolist(),
                self.inference_output_dir(str(seed)),
                "nn-solutions.txt",
                len(result.episodes),
                result.elapsed,
                result.mean_return,
            )
        self.obs = self.env.reset()
        return result

    def inference_schedule(self, mode, num_solutions=None, converge_patience=None):
        if mode in ("one-per-model", "max-likelihood"):
            return OneEpisodePerEnv(self.num_envs)
        if mode == "fixed-count":
            if not num_solutions or num_solutions < 1:
                raise ValueError("fixed-count inference needs a positive num_solutions.")
            return FixedEpisodeCount(num_solutions)
        if mode == "until-converged":
            return UntilNoImprovement(converge_patience or self.num_envs)
        raise ValueError(f"Unknown inference mode {mode!r}; use one of {INFERENCE_MODES}.")

    def load_inference_policy(self, nn_class, greedy=False):
        """Build the network from the checkpoint's own config and load its weights."""
        ac_model = self.args.ac_model
        if not ac_model:
            raise ValueError("No checkpoint (--ac_model) given for inference.")
        self.ac = nn_class(
            self.pallete,
            self.args.hidden_dim,
            len(self.pallete.units),
            **self._ac_kwargs(model_dir=os.path.dirname(ac_model)),
        ).to(self.args.device)
        self.ac.load_state_dict(torch.load(ac_model, map_location=self.args.device))
        self.ac.eval()
        self.ac.set_greedy(greedy)

    def inference_output_dir(self, run_label, model_name=None):
        """``<runs dir>/sampled_solutions/<group>[/<model>]/<checkpoint>--<run_label>``."""
        checkpoint = Path(self.args.ac_model).resolve()
        parts = [
            str(checkpoint.parent.parent),
            "sampled_solutions",
            self.args.wandb_config.get("group", "inference"),
        ]
        if model_name is not None:
            parts.append(model_name)
        parts.append(f"{checkpoint.stem}--{run_label}")
        return os.path.join(*parts)

    @staticmethod
    def episodes_by_model(result, env):
        """``(model_name, episode)`` per episode in env order; repeated models get an env suffix."""
        used_names = set()
        for episode in sorted(result.episodes, key=lambda e: e.env_index):
            env_index = episode.env_index
            model_name = env.instance_names[env.instance_ids[env_index]]
            if model_name in used_names:
                model_name = f"{model_name}-env{env_index}"
            used_names.add(model_name)
            yield model_name, episode

    def write_per_model_results(self, result, env, model_dir, seed, extra, verbose=True):
        """Write the solutions and the transformation sequence of each model's episode.

        ``model_dir(model_name)`` gives the output directory, ``extra`` is added to
        each ``sequence.json``.
        """
        for model_name, episode in self.episodes_by_model(result, env):
            if verbose:
                stalled_note = " (stalled)" if episode.stalled else ""
                print(
                    f"Model: {model_name}, return: {episode.total_reward:.4f}, "
                    f"steps: {episode.length}{stalled_note}"
                )
            solutions_path = model_dir(model_name)
            self.write_solutions_to_dir(
                seed,
                episode.solutions,
                solutions_path,
                "nn-solutions.txt",
                1,
                result.elapsed,
                episode.total_reward,
                verbose=verbose,
            )
            sequence = {
                "model": model_name,
                **extra,
                "return": episode.total_reward,
                "length": episode.length,
                "stalled": episode.stalled,
                "steps": episode.steps,
            }
            with open(os.path.join(solutions_path, "sequence.json"), "w") as f:
                json.dump(sequence, f, indent=2)

    def write_solutions_to_dir(
        self,
        seed,
        solution_set,
        solutions_path,
        file_path,
        sampled_episodes,
        time_taken,
        mean_rew,
        verbose=True,
    ):
        os.makedirs(solutions_path, exist_ok=True)
        if verbose:
            print("Writing solutions to", solutions_path)
        full_file_path = os.path.join(solutions_path, file_path)
        with open(full_file_path, "w") as f:
            for sol in solution_set:
                f.write(str(sol) + "\n")
        with open(os.path.join(solutions_path, "info.txt"), "a") as f:
            f.write(
                f"Inference time (seed={seed}): {time_taken:.3f} seconds ({sampled_episodes} episodes)\n"
            )
            if mean_rew is not None:
                f.write(f"Mean reward (seed={seed}): {mean_rew:.3f}\n")

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
            total_timesteps - the total number of timesteps to train for

        Return:
            None
        """
        print(
            f"Learning... {self.n_steps} steps per environment, "
            f"{self.timesteps_per_batch} transitions per batch, "
            f"{self.num_envs} parallel environments "
            f"({self.n_updates_per_iteration} updates of {self.mini_batch_size} steps)",
            flush=True,
        )
        t_so_far = 0  # Timesteps simulated so far
        e_so_far = 0  # Episodes simulated so far
        i_so_far = 0  # Iterations ran so far

        if self.args.rl_evaluate:
            print("Evaluating initial policy before training...", flush=True)
            eval_log = self.run_evaluation("initial", 0, 0)
            if self.es_eval_variant is not None:
                eval_metric = eval_log[f"eval_return_norm_{self.es_eval_variant}"]
                _, improved, _ = self._consider_early_stopping(
                    eval_metric, 0, 0, from_evaluation=True
                )
                self._save_named_checkpoint(0, 0, eval_metric, improved)
            self._flush_eval_log(0, 0)

        #  t_roll, t_upd = [], []

        while t_so_far < total_timesteps:

            # Collect a batch of experience
            if self.args.rl_sample_partial_trajectories:
                print("Using truncated rollouts!", flush=True)

                (
                    batch_obs,
                    batch_acts,
                    batch_log_probs,
                    batch_returns,
                    batch_advantages,
                    batch_lens,
                    finished_eps,
                    batch_env_ids,
                ) = self.rollout()
            else:
                print("Using full trajectory rollouts!", flush=True)
                (
                    batch_obs,
                    batch_acts,
                    batch_log_probs,
                    batch_returns,
                    batch_advantages,
                    batch_lens,
                    finished_eps,
                    batch_env_ids,
                ) = self.full_episodes_rollout()

            # Count stored transitions (not segment_lens, which can overshoot after truncation)
            t_so_far += len(batch_obs)
            e_so_far += finished_eps
            # Increment the number of iterations
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["e_so_far"] = e_so_far
            self.logger["i_so_far"] = i_so_far

            device = self.args.device
            batch_obs = [obs.to(device) for obs in batch_obs]
            batch_act, batch_nodes, batch_vals_raw = self._stack_actions(batch_acts)
            if (
                batch_vals_raw is not None
                and getattr(self, "value_head", "gaussian") == "bins"
            ):
                self.logger["value_bin_indices"] = (
                    batch_vals_raw.detach().cpu().reshape(-1)
                )
            batch_log_probs = batch_log_probs.to(device)
            batch_returns = batch_returns.to(device)
            batch_advantages = batch_advantages.to(device)

            if batch_log_probs.ndim == 1:
                batch_log_probs = batch_log_probs.unsqueeze(-1)

            if len(batch_obs) == 0:
                self.logger["explained_variance"] = float("nan")
                self.logger["explained_variance_by_instance"] = {}
                self.logger["batch_returns_mean"] = float("nan")
                self.logger["batch_returns_std"] = float("nan")
                self.logger["epochs_run"] = 0
                self.logger["kl_early_stop"] = False
            else:
                with torch.no_grad():
                    predicted_values = batch_returns - batch_advantages
                    self.logger["explained_variance"] = explained_variance(
                        batch_returns, predicted_values
                    )
                    self.logger["explained_variance_by_instance"] = (
                        self._per_instance_explained_variance(
                            batch_returns, predicted_values, batch_env_ids
                        )
                    )
                    self.logger["batch_returns_mean"] = float(batch_returns.mean().item())
                    if batch_returns.numel() > 1:
                        self.logger["batch_returns_std"] = float(
                            batch_returns.std().item()
                        )
                    else:
                        self.logger["batch_returns_std"] = float("nan")

            indices = np.arange(len(batch_obs))
            continue_training = True
            kl_early_stop = False
            epochs_run = 0
            # Create balanced mini-batches to ensure equal environment representation
            n_epochs = self.n_updates_per_iteration if len(batch_obs) > 0 else 0
            for epoch in range(n_epochs):
                if not continue_training:
                    break
                epochs_run = epoch + 1
                np.random.shuffle(indices)
                mini_batch_indices = [
                    indices[i : i + self.mini_batch_size]
                    for i in range(0, len(batch_obs), self.mini_batch_size)
                ]

                for mb_indices in mini_batch_indices:
                    mb_indices = np.asarray(mb_indices)
                    batch_obs_mb = [batch_obs[i] for i in mb_indices]
                    batch_log_probs_mb = batch_log_probs[mb_indices]
                    batch_returns_mb = batch_returns[mb_indices]
                    batch_advantages_mb = batch_advantages[mb_indices]
                    mb_act = batch_act[mb_indices]
                    mb_nodes = (
                        batch_nodes[mb_indices] if batch_nodes is not None else None
                    )
                    mb_vals = (
                        batch_vals_raw[mb_indices]
                        if batch_vals_raw is not None
                        else None
                    )

                    if self.debug_nan and batch_advantages_mb.numel() <= 1:
                        f_logger.warning(
                            f"[NaN Debug] batch_advantages_mb has {batch_advantages_mb.numel()} elements - std() will have <= 0 degrees of freedom!"
                        )

                    A_k = (batch_advantages_mb - batch_advantages_mb.mean()) / (
                        batch_advantages_mb.std() + 1e-8
                    )

                    if self.debug_nan and torch.isnan(A_k).any():
                        f_logger.error(f"[NaN Debug] NaN in A_k after normalization!")
                        f_logger.error(
                            f"  batch_advantages_mb: min={batch_advantages_mb.min():.4f}, max={batch_advantages_mb.max():.4f}, mean={batch_advantages_mb.mean():.4f}, std={batch_advantages_mb.std():.4f}"
                        )

                    mb = Batch.from_data_list(batch_obs_mb)

                    V, stats = self.evaluate(mb, mb_act, mb_nodes, mb_vals)
                    curr_logp = stats["logp"]
                    old_logp = batch_log_probs_mb
                    if old_logp.shape[-1] != curr_logp.shape[-1]:
                        # Backward compat: old joint scalars vs new [K] factors
                        k = min(old_logp.shape[-1], curr_logp.shape[-1])
                        curr_joint = curr_logp[..., :k].sum(-1)
                        old_joint = old_logp[..., :k].sum(-1)
                    else:
                        curr_joint = curr_logp.sum(-1)
                        old_joint = old_logp.sum(-1)

                    if self.debug_nan:
                        if V.shape != batch_returns_mb.shape:
                            f_logger.warning(
                                f"[NaN Debug] Shape mismatch between V and batch_returns_mb!"
                            )
                            f_logger.warning(
                                f"  V shape: {V.shape}, batch_returns_mb shape: {batch_returns_mb.shape}"
                            )
                        if torch.isnan(V).any():
                            f_logger.error(
                                f"NaN in V after evaluate: {torch.isnan(V).sum()}/{V.numel()} values"
                            )
                        if torch.isnan(curr_joint).any():
                            f_logger.error(
                                f"NaN in curr_log_probs: {torch.isnan(curr_joint).sum()}/{curr_joint.numel()} values"
                            )

                    log_ratio = curr_joint - old_joint
                    with torch.no_grad():
                        approx_kl = (-log_ratio).mean()
                        approx_kl_k3 = ((log_ratio.exp() - 1) - log_ratio).mean()
                        k = min(old_logp.shape[-1], curr_logp.shape[-1])
                        factor_kl = (old_logp[..., :k] - curr_logp[..., :k]).mean(dim=0)
                        if factor_kl.numel() < NUM_LOGP_FACTORS:
                            factor_kl = F.pad(
                                factor_kl, (0, NUM_LOGP_FACTORS - factor_kl.numel())
                            )

                    if self.target_kl and approx_kl.item() > 1.5 * self.target_kl:
                        continue_training = False
                        kl_early_stop = True
                        break

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    ratios = torch.exp(log_ratio)

                    # Calculate surrogate losses.
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    ent_rule = (
                        stats["ent_rule_norm"]
                        if self.entropy_normalize
                        else stats["ent_rule"]
                    )
                    ent_node = (
                        stats["ent_node_norm"]
                        if self.entropy_normalize
                        else stats["ent_node"]
                    )
                    ent_val = (
                        stats["ent_val_norm"]
                        if self.entropy_normalize
                        and stats.get("ent_val_norm") is not None
                        else stats["ent_val"]
                    )
                    entropy_bonus = (
                        self.entropy_coef_rule * ent_rule
                        + self.entropy_coef_node * ent_node
                        + self.entropy_coef_val * ent_val
                    )
                    actor_loss = (-torch.min(surr1, surr2)).mean() - entropy_bonus.mean()
                    curr_dist_entropies = (
                        stats["ent_rule"] + stats["ent_node"] + stats["ent_val"]
                    )

                    critic_loss = 0.5 * nn.MSELoss()(V, batch_returns_mb)

                    if self.debug_nan and (
                        torch.isnan(critic_loss) or torch.isinf(critic_loss)
                    ):
                        f_logger.error(
                            f"[NaN Debug] critic_loss is NaN/Inf: {critic_loss}"
                        )

                    with torch.no_grad():
                        clip_frac = (
                            ((ratios - 1.0).abs() > self.clip).float().mean()
                        )
                        self.logger["actor_losses"].append(actor_loss.detach())
                        self.logger["critic_losses"].append(critic_loss.detach())
                        self.logger["entropy"].append(curr_dist_entropies.detach())
                        self.logger["entropy_rule"].append(stats["ent_rule"].detach())
                        self.logger["entropy_node"].append(stats["ent_node"].detach())
                        self.logger["entropy_val"].append(stats["ent_val"].detach())
                        self.logger["entropy_rule_norm"].append(
                            stats["ent_rule_norm"].detach()
                        )
                        self.logger["entropy_node_norm"].append(
                            stats["ent_node_norm"].detach()
                        )
                        self.logger["entropy_val_norm"].append(
                            stats["ent_val_norm"].detach()
                            if stats.get("ent_val_norm") is not None
                            else torch.zeros_like(stats["ent_val"].detach())
                        )
                        self.logger["approx_kl"].append(approx_kl.detach())
                        self.logger["approx_kl_k3"].append(approx_kl_k3.detach())
                        self.logger["approx_kl_rule"].append(factor_kl[0].detach())
                        self.logger["approx_kl_node"].append(factor_kl[1].detach())
                        self.logger["approx_kl_val"].append(factor_kl[2].detach())
                        self.logger["clip_fraction"].append(clip_frac.detach())
                        self.logger["ratio_min"].append(ratios.min().detach())
                        self.logger["ratio_max"].append(ratios.max().detach())
                        self.logger["adv_mean"].append(
                            batch_advantages_mb.mean().detach()
                        )
                        self.logger["adv_std"].append(
                            batch_advantages_mb.std().detach()
                        )

                    if self.args.no_nn_updates:
                        continue

                    if self.debug_nan:
                        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                            f_logger.error(
                                f"NaN/Inf detected in actor_loss: {actor_loss.item()}"
                            )
                            f_logger.error(
                                f"  curr_log_probs stats: min={curr_joint.min():.4f}, max={curr_joint.max():.4f}, has_nan={torch.isnan(curr_joint).any()}"
                            )
                            f_logger.error(
                                f"  batch_log_probs_mb stats: min={old_joint.min():.4f}, max={old_joint.max():.4f}"
                            )
                            f_logger.error(
                                f"  ratios stats: min={ratios.min():.4f}, max={ratios.max():.4f}"
                            )
                            f_logger.error(
                                f"  A_k stats: min={A_k.min():.4f}, max={A_k.max():.4f}"
                            )
                        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                            f_logger.error(
                                f"NaN/Inf detected in critic_loss: {critic_loss.item()}"
                            )
                            f_logger.error(
                                f"  V stats: min={V.min():.4f}, max={V.max():.4f}, has_nan={torch.isnan(V).any()}"
                            )
                            f_logger.error(
                                f"  batch_returns_mb stats: min={batch_returns_mb.min():.4f}, max={batch_returns_mb.max():.4f}"
                            )

                    # Calculate gradients and perform backward propagation for actor network
                    gn_policy = torch.tensor(0.0)
                    gn_critic = torch.tensor(0.0)
                    if self.encoder == "shared":
                        cb_loss = actor_loss + critic_loss

                        self.ac_optim.zero_grad()
                        cb_loss.backward()

                        gn_policy = torch.nn.utils.clip_grad_norm_(
                            self.ac.parameters(), max_norm=0.5
                        )
                        self.ac_optim.step()

                    elif self.encoder == "separate":
                        self.a_optim.zero_grad()
                        self.c_optim.zero_grad()
                        actor_loss.backward()

                        if self.debug_nan:
                            self.gradient_monitor.log_gradients()

                        gn_policy = torch.nn.utils.clip_grad_norm_(
                            self.ac.policy_parameters(), max_norm=0.5
                        )
                        self.a_optim.step()

                        critic_loss.backward()

                        if self.debug_nan:
                            self.gradient_monitor.log_gradients()

                        gn_critic = torch.nn.utils.clip_grad_norm_(
                            self.ac.value_parameters(), max_norm=0.5
                        )
                        self.c_optim.step()

                        if self.debug_nan:
                            self.nan_detector.step()
                            post_step_issues = (
                                self.nan_detector.check_model_parameters()
                            )
                            if post_step_issues:
                                f_logger.error(
                                    f"NaN/Inf in parameters after optimizer step: {len(post_step_issues)} issues"
                                )

                    self.logger["grad_norm"].append(
                        gn_policy.detach()
                        if torch.is_tensor(gn_policy)
                        else torch.tensor(float(gn_policy))
                    )
                    self.logger["grad_norm_critic"].append(
                        gn_critic.detach()
                        if torch.is_tensor(gn_critic)
                        else torch.tensor(float(gn_critic))
                    )

            self.logger["epochs_run"] = epochs_run
            self.logger["kl_early_stop"] = kl_early_stop
            if self.encoder == "shared":
                self.logger["lr"] = float(self.ac_optim.param_groups[0]["lr"])
            else:
                self.logger["lr"] = float(self.a_optim.param_groups[0]["lr"])

           # if self.encoder == "shared":
           #     self.lr_scheduler.step()

            eval_log = None
            if self.args.rl_evaluate and i_so_far % self.eval_interval == 0:
                eval_log = self.run_evaluation(
                    f"iter{i_so_far:05d}-t{t_so_far}", i_so_far, t_so_far
                )

            if self.es_eval_variant is None:
                # Default: early stopping follows the rollout returns.
                es_metric = self.episode_stats.get_windowed_macro_return(self.es_window)
                should_stop, improved, es_ticked = self._consider_early_stopping(
                    es_metric, i_so_far, t_so_far
                )
            elif eval_log is not None:
                es_metric = eval_log[f"eval_return_norm_{self.es_eval_variant}"]
                should_stop, improved, es_ticked = self._consider_early_stopping(
                    es_metric, i_so_far, t_so_far, from_evaluation=True
                )
            else:
                es_metric = float("nan")
                should_stop, improved, es_ticked = False, False, False

            self.logger["es_metric"] = es_metric
            self.logger["es_ticked"] = int(es_ticked)
            self._save_named_checkpoint(i_so_far, t_so_far, es_metric, improved)

            if should_stop:
                self._flush_eval_log(i_so_far, t_so_far)
                print(
                    f"Early stopping triggered after {self.early_stopping.patience} updates without improvement"
                )
                print(f"Best reward achieved: {self.early_stopping.best_reward:.3f}")
                break

            self._log_summary()
            if self.args.verbose:
                for name, param in self.ac.named_parameters():
                    self.writer.add_histogram(f"weights/{name}", param, i_so_far)
                    if param.grad is not None:
                        self.writer.add_histogram(f"grads/{name}", param.grad, i_so_far)

        # Reached after the timestep budget is used up and after an early-stopping break.
        if self.args.rl_evaluate:
            print(
                f"Training finished after {i_so_far} iterations ({t_so_far} timesteps); "
                "running final evaluation...",
                flush=True,
            )
            self.run_evaluation("final", i_so_far, t_so_far)
            self._flush_eval_log(i_so_far, t_so_far)
            self.eval_env.close()

        # Cleanup after run
        self.env.post_training()
        self.env.close()

    def run_evaluation(self, label, update_step, timesteps):
        """Play one episode per evaluation model for each variant in ``self.eval_variants``.

        The eval env groups are double-buffered as in the rollout. Results are stored in
        ``self.pending_eval_log`` for logging only; early stopping does not use them.
        """
        log = {}
        t0 = time.perf_counter()
        self.ac.eval()
        try:
            for variant in self.eval_variants:
                greedy = variant == "greedy"
                self.eval_env.reset()
                self.ac.set_greedy(greedy)
                result = InferenceSampler(
                    self,
                    OneEpisodePerEnv(self.eval_env.num_envs),
                    greedy=greedy,
                    env=self.eval_env,
                    desc=f"Eval {variant}",
                ).run()
                print(f"[eval {variant} @ {label}] {result.describe()}", flush=True)

                self.episode_stats.add_eval_mean_return(result.mean_return, variant)
                log[f"eval_return_norm_{variant}"] = self._normalized_eval_return(
                    variant, result
                )
                log[f"eval_return_{variant}"] = float(result.mean_return)
                for model_name, episode in self.episodes_by_model(result, self.eval_env):
                    log[f"eval_return_{variant}_{_metric_name(model_name)}"] = float(
                        episode.total_reward
                    )
                if greedy:
                    log["eval_stalled_greedy"] = sum(e.stalled for e in result.episodes)
                log[f"time_eval_{variant}_env_wait"] = result.time_env_wait
                log[f"time_eval_{variant}_inference"] = result.time_inference
                log[f"time_eval_{variant}_decode"] = result.time_decode

                if self.args.rl_eval_save_solutions:
                    variant_dir = os.path.join(
                        self.args.model_path, "eval_solutions", variant
                    )
                    self.write_per_model_results(
                        result,
                        self.eval_env,
                        lambda model_name: os.path.join(variant_dir, model_name, label),
                        self.args.seed,
                        {
                            "variant": variant,
                            "iteration": update_step,
                            "timesteps": timesteps,
                        },
                        verbose=False,
                    )
        finally:
            self.ac.set_greedy(False)
            self.ac.train()
        log["time_eval"] = time.perf_counter() - t0
        self.pending_eval_log.update(log)
        return log

    def _normalized_eval_return(self, variant, result):
        """Mean over eval models of return / |return in the first evaluation of ``variant``|."""
        returns = {e.env_index: float(e.total_reward) for e in result.episodes}
        if not returns:
            return float("nan")
        scales = self._eval_return_scales.setdefault(
            variant, {i: max(abs(r), 1e-3) for i, r in returns.items()}
        )
        if set(returns) != set(scales):
            raise ValueError(
                f"Evaluation ({variant}) finished envs {sorted(returns)}, "
                f"but the frozen scale covers {sorted(scales)}"
            )
        return float(np.mean([returns[i] / scales[i] for i in returns]))

    def _take_eval_log(self):
        """Pending eval values without non-finite entries; clears the pending log."""
        log = {
            key: value
            for key, value in self.pending_eval_log.items()
            if not (isinstance(value, float) and not math.isfinite(value))
        }
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

    def _flush_eval_log(self, iteration, timesteps):
        """Log a pending eval phase on its own, outside an iteration summary."""
        log = self._take_eval_log()
        if not log:
            return
        self._log_eval_timers(log)
        if self.args.use_wandb:
            wandb.log({**log, "iteration": iteration, "timesteps": timesteps})
        if self.args.verbose and hasattr(self, "writer"):
            for key, value in log.items():
                self.writer.add_scalar(f"ppo/{key}", value, iteration)

    def _reset_episode_trackers(self):
        self.returns[:] = 0
        self._current_episode_rewards[:] = 0
        self._current_episode_lengths[:] = 0

    def _normalize_reward(self, env_i, raw_rew):
        if not self.normalize_rewards:
            return float(raw_rew)
        self.returns[env_i] = self.returns[env_i] * self.gamma + float(raw_rew)
        update_stats = (
            self.rew_norm_freeze_after < 0
            or self._reward_norm_updates < self.rew_norm_freeze_after
        )
        instance_id = int(self.env.instance_ids[env_i])
        scaled = scale_reward(
            self._instance_reward_normalizers[instance_id],
            self.returns[env_i],
            raw_rew,
            self.reward_clip,
            update_stats,
        )
        if update_stats:
            self._reward_norm_updates += 1
        return scaled

    def stage_teacher_forced_observations(self, obs, action, node_mask):
        """Build teacher-forcing copies while env workers execute the step."""
        return [
            self.add_act_forcing_data_to_obs(obs[i], action[i], node_mask[i])
            for i in range(len(obs))
        ]

    def infer_group_actions(self, indices):
        obs_g = [self.obs[i] for i in indices]
        t0 = time.perf_counter()
        (action, log_prob, node_mask), value = self.get_action_and_value(obs_g)
        self.time_inference += time.perf_counter() - t0
        value = np.atleast_1d(np.asarray(value, dtype=np.float32).reshape(-1))
        staged = self.stage_teacher_forced_observations(obs_g, action, node_mask)
        return {
            "indices": indices,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "staged": staged,
        }

    def send_group_actions(self, group_index, pending_step):
        if hasattr(self.env, "step_async_group"):
            self.env.step_async_group(group_index, pending_step["action"])
        else:
            self.env.step_async(pending_step["action"])

    def wait_group_results(self, group_index, env=None):
        env = self.env if env is None else env
        t0 = time.perf_counter()
        if hasattr(env, "step_wait_group"):
            result = env.step_wait_group(group_index)
        else:
            result = env.step_wait()
        wait_dt = time.perf_counter() - t0
        decode_dt = float(getattr(env, "last_decode_time", 0.0) or 0.0)
        self.time_decode += decode_dt
        self.time_env_wait += max(0.0, wait_dt - decode_dt)
        return result

    def store_rollout_timers(self):
        self.logger["time_env_wait"] = self.time_env_wait
        self.logger["time_inference"] = self.time_inference
        self.logger["time_decode"] = self.time_decode

    def collect_double_buffered_steps(self, group_should_continue, rollout):
        self.time_env_wait = 0.0
        self.time_inference = 0.0
        self.time_decode = 0.0
        groups = getattr(self.env, "groups", None) or [list(range(self.num_envs))]
        num_groups = len(groups)
        pending = [None] * num_groups
        for group_index in range(num_groups):
            if not group_should_continue(groups[group_index]):
                continue
            pending[group_index] = self.infer_group_actions(groups[group_index])
            self.send_group_actions(group_index, pending[group_index])

        while any(step is not None for step in pending):
            for group_index in range(num_groups):
                if pending[group_index] is None:
                    continue
                pending_step = pending[group_index]
                next_obs, rew, done = self.wait_group_results(group_index)
                pending[group_index] = None
                self.record_group_transitions(rollout, pending_step, next_obs, rew, done)
                if group_should_continue(groups[group_index]):
                    pending[group_index] = self.infer_group_actions(groups[group_index])
                    self.send_group_actions(group_index, pending[group_index])

    def record_transition(
        self,
        i,
        staged_obs_i,
        action_i,
        log_prob_i,
        value_i,
        rew_i,
        done_i,
        next_obs_i,
        rollout,
    ):
        if next_obs_i is None:
            rollout["skipped_invalid_steps"] += 1
            return False

        if self._action_has_invalid_nodes(action_i):
            rollout["skipped_invalid_nodes"] = (
                rollout.get("skipped_invalid_nodes", 0) + 1
            )
            self.obs[i] = next_obs_i
            rollout["last_done"][i] = float(done_i)
            self._current_episode_rewards[i] += rew_i
            self._current_episode_lengths[i] += 1
            self.episode_stats.add_step_reward(rew_i)
            if done_i:
                episode_return = float(self._current_episode_rewards[i])
                episode_length = int(self._current_episode_lengths[i])
                self.episode_stats.add_episode(
                    episode_return, episode_length, env_id=i
                )
                rollout["episodes_this_rollout"][i] += 1
                if rollout["full_episodes"]:
                    rollout["obs_env"][i] = []
                    rollout["acts_env"][i] = []
                    rollout["log_probs_env"][i] = []
                    rollout["rews_env"][i] = []
                    rollout["dones_env"][i] = []
                    rollout["vals_env"][i] = []
                    if self.normalize_rewards:
                        self.returns[i] = 0.0
                    self._current_episode_rewards[i] = 0.0
                    self._current_episode_lengths[i] = 0
                    rollout["t_env"][i] = 0
                else:
                    self._current_episode_rewards[i] = 0.0
                    self._current_episode_lengths[i] = 0
                    if self.normalize_rewards:
                        self.returns[i] = 0.0
            return False

        normalized_rew = self._normalize_reward(i, rew_i)
        rollout["obs_env"][i].append(staged_obs_i)
        rollout["acts_env"][i].append(action_i)
        rollout["log_probs_env"][i].append(log_prob_i)
        rollout["rews_env"][i].append(normalized_rew)
        rollout["dones_env"][i].append(float(done_i))
        rollout["vals_env"][i].append(float(value_i))
        rollout["last_done"][i] = float(done_i)

        self._current_episode_rewards[i] += rew_i
        self._current_episode_lengths[i] += 1
        self.episode_stats.add_step_reward(rew_i)
        self.obs[i] = next_obs_i

        if rollout["full_episodes"]:
            rollout["t_env"][i] += 1
            rollout["progress_bar"].update(1)
            if done_i:
                current_ep_len = int(self._current_episode_lengths[i])
                episode_return = float(self._current_episode_rewards[i])
                rollout["ep_lens"].append(current_ep_len)
                self.episode_stats.add_episode(
                    episode_return, current_ep_len, env_id=i
                )
                rollout["episodes_this_rollout"][i] += 1
                rollout["completed_env_ids"].extend([i] * len(rollout["obs_env"][i]))
                rollout["completed_obs"].extend(rollout["obs_env"][i])
                rollout["completed_acts"].extend(rollout["acts_env"][i])
                rollout["completed_log_probs"].extend(rollout["log_probs_env"][i])
                rollout["completed_rews"].extend(rollout["rews_env"][i])
                rollout["completed_dones"].extend(rollout["dones_env"][i])
                rollout["completed_vals"].extend(rollout["vals_env"][i])
                rollout["obs_env"][i] = []
                rollout["acts_env"][i] = []
                rollout["log_probs_env"][i] = []
                rollout["rews_env"][i] = []
                rollout["dones_env"][i] = []
                rollout["vals_env"][i] = []
                if self.normalize_rewards:
                    self.returns[i] = 0.0
                self._current_episode_rewards[i] = 0.0
                self._current_episode_lengths[i] = 0
                rollout["t_env"][i] = 0
        else:
            rollout["steps_in_current_segment"][i] += 1
            rollout["t"] += 1
            rollout["progress_bar"].update(1)
            if done_i:
                episode_return = float(self._current_episode_rewards[i])
                episode_length = int(self._current_episode_lengths[i])
                rollout["completed_ep_returns"].append(episode_return)
                rollout["completed_ep_lens"].append(episode_length)
                self.episode_stats.add_episode(
                    episode_return, episode_length, env_id=i
                )
                rollout["episodes_this_rollout"][i] += 1
                rollout["segment_lens"].append(int(rollout["steps_in_current_segment"][i]))
                self._current_episode_rewards[i] = 0.0
                self._current_episode_lengths[i] = 0
                rollout["steps_in_current_segment"][i] = 0
                if self.normalize_rewards:
                    self.returns[i] = 0.0
        return True

    def record_group_transitions(self, rollout, pending_step, next_obs, rew, done):
        for local_i, env_i in enumerate(pending_step["indices"]):
            self.record_transition(
                env_i,
                pending_step["staged"][local_i],
                pending_step["action"][local_i],
                pending_step["log_prob"][local_i],
                pending_step["value"][local_i],
                rew[local_i],
                done[local_i],
                next_obs[local_i],
                rollout,
            )

    def _stack_actions(self, batch_acts):
        device = self.args.device
        batch_act = torch.stack(
            [
                a[0] if torch.is_tensor(a[0]) else torch.tensor(a[0])
                for a in batch_acts
            ]
        ).to(device=device, dtype=torch.long)
        batch_nodes = None
        batch_vals_raw = None
        if batch_acts[0][1] is not None:
            nodes = [a[1] for a in batch_acts]
            batch_nodes = self.pad_and_stack(nodes).squeeze(1).to(device)
        if batch_acts[0][2] is not None:
            batch_vals_raw = torch.stack(
                [a[2][1] for a in batch_acts], dim=0
            ).squeeze(1).to(device)
        return batch_act, batch_nodes, batch_vals_raw

    def full_episodes_rollout(self):
        """
        Collects experience using vectorized environments, but only records data from
        completed episodes (done[i] = True) for training the PPO network.

        Returns:
            batch_obs:        [T] list of observations (from completed episodes only)
            batch_acts:       [T] list of actions (from completed episodes only)
            batch_log_probs:  [T] tensor of log probs (from completed episodes only)
            batch_returns:    [T] tensor of returns (from completed episodes only)
            batch_advantages: [T] tensor of advantages (from completed episodes only)
            batch_lens:       [N] list of episode lengths (completed episodes only)
            finished_episodes: int number of completed episodes
            batch_env_ids: env index of each stored transition
        """
        self.episode_stats.start_rollout()

        obs_env = [[] for _ in range(self.num_envs)]
        acts_env = [[] for _ in range(self.num_envs)]
        log_probs_env = [[] for _ in range(self.num_envs)]
        rews_env = [[] for _ in range(self.num_envs)]
        dones_env = [[] for _ in range(self.num_envs)]
        vals_env = [[] for _ in range(self.num_envs)]

        completed_obs = []
        completed_acts = []
        completed_log_probs = []
        completed_rews = []
        completed_dones = []
        completed_vals = []
        completed_env_ids = []

        ep_lens = []
        progress_bar = nqdm(total=self.timesteps_per_batch, position=0, leave=True)
        rollout = {
            "full_episodes": True,
            "obs_env": obs_env,
            "acts_env": acts_env,
            "log_probs_env": log_probs_env,
            "rews_env": rews_env,
            "dones_env": dones_env,
            "vals_env": vals_env,
            "completed_obs": completed_obs,
            "completed_acts": completed_acts,
            "completed_log_probs": completed_log_probs,
            "completed_rews": completed_rews,
            "completed_dones": completed_dones,
            "completed_vals": completed_vals,
            "completed_env_ids": completed_env_ids,
            "ep_lens": ep_lens,
            "skipped_invalid_steps": 0,
            "skipped_invalid_nodes": 0,
            "episodes_this_rollout": np.zeros(self.num_envs, dtype=int),
            "t_env": np.zeros(self.num_envs, dtype=int),
            "last_done": np.zeros(self.num_envs, dtype=np.float32),
            "progress_bar": progress_bar,
        }

        def full_episode_group_should_continue(_indices):
            batch_full = len(rollout["completed_obs"]) >= self.timesteps_per_batch
            every_env_finished = all(
                count >= 1 for count in rollout["episodes_this_rollout"]
            )
            return not (batch_full and every_env_finished)

        self.collect_double_buffered_steps(full_episode_group_should_continue, rollout)

        progress_bar.close()
        finished_episodes = len(ep_lens)
        self.env.post_rollout()
        self.logger["skipped_invalid_steps"] = rollout["skipped_invalid_steps"]
        self.logger["skipped_invalid_nodes"] = rollout.get("skipped_invalid_nodes", 0)
        self._set_invalid_action_rate(
            rollout["skipped_invalid_steps"], len(completed_obs)
        )
        self.store_rollout_timers()
        self.episode_stats.end_rollout()

        if len(completed_obs) > 0:
            completed_rews_per_ep = []
            completed_dones_per_ep = []
            completed_vals_per_ep = []

            start_idx = 0
            for ep_len in ep_lens:
                end_idx = start_idx + ep_len
                completed_rews_per_ep.append(completed_rews[start_idx:end_idx])
                completed_dones_per_ep.append(completed_dones[start_idx:end_idx])
                completed_vals_per_ep.append(completed_vals[start_idx:end_idx])
                start_idx = end_idx

            next_values_per_ep = [0.0 for _ in range(len(ep_lens))]
            batch_returns, batch_advantages = self.compute_gae_per_env(
                completed_rews_per_ep,
                completed_dones_per_ep,
                completed_vals_per_ep,
                next_values_per_ep,
            )
            batch_log_probs = torch.tensor(
                completed_log_probs, dtype=torch.float32, device=self.args.device
            )
            if batch_log_probs.ndim == 1:
                batch_log_probs = batch_log_probs.unsqueeze(-1)
        else:
            completed_obs = []
            completed_acts = []
            batch_log_probs = torch.zeros(
                0, NUM_LOGP_FACTORS, dtype=torch.float32, device=self.args.device
            )
            batch_returns = torch.tensor(
                [], dtype=torch.float32, device=self.args.device
            )
            batch_advantages = torch.tensor(
                [], dtype=torch.float32, device=self.args.device
            )

        self.logger["batch_lens"] = ep_lens

        if self.pallete.max_val_len > 0:
            self.logger["value_head_log_std"] = (
                self.ac.actor.value_decoder.log_std.item()
                if hasattr(self.ac.actor.value_decoder, "log_std")
                else 0.0
            )

        return (
            completed_obs,
            completed_acts,
            batch_log_probs,
            batch_returns,
            batch_advantages,
            ep_lens,
            finished_episodes,
            completed_env_ids,
        )

    def rollout(self):
        """
        Collects a fixed-length batch of experience using vectorized environments.
        Handles environments that step asynchronously by tracking per-env buffers.
        Computes per-env GAE returns and advantages.

        Tracks cumulative episode returns across rollout boundaries (SB3 style):
        - Episode rewards/lengths accumulate in self._current_episode_rewards/lengths
        - Only completed episodes (done=True) are reported for logging
        - All transitions are used for training regardless of episode boundaries

        Returns:
            batch_obs:        [T] list of observations (flattened)
            batch_acts:       [T] list of actions (flattened)
            batch_log_probs:  [T] tensor of log probs
            batch_returns:    [T] tensor of returns (target values for critic)
            batch_advantages: [T] tensor of advantages (for policy)
            segment_lens:     [N] list of per-env segment lengths in this rollout
            finished_episodes: int number of completed episodes
        """
        self.episode_stats.start_rollout()
        obs_env = [[] for _ in range(self.num_envs)]
        acts_env = [[] for _ in range(self.num_envs)]
        log_probs_env = [[] for _ in range(self.num_envs)]
        rews_env = [[] for _ in range(self.num_envs)]
        dones_env = [[] for _ in range(self.num_envs)]
        vals_env = [[] for _ in range(self.num_envs)]

        completed_ep_returns = []
        completed_ep_lens = []
        segment_lens = []
        progress_bar = nqdm(total=self.timesteps_per_batch, position=0, leave=True)
        rollout = {
            "full_episodes": False,
            "obs_env": obs_env,
            "acts_env": acts_env,
            "log_probs_env": log_probs_env,
            "rews_env": rews_env,
            "dones_env": dones_env,
            "vals_env": vals_env,
            "completed_ep_returns": completed_ep_returns,
            "completed_ep_lens": completed_ep_lens,
            "segment_lens": segment_lens,
            "skipped_invalid_steps": 0,
            "skipped_invalid_nodes": 0,
            "episodes_this_rollout": np.zeros(self.num_envs, dtype=int),
            "t": 0,
            "steps_in_current_segment": np.zeros(self.num_envs, dtype=int),
            "last_done": np.zeros(self.num_envs, dtype=np.float32),
            "progress_bar": progress_bar,
        }

        def partial_group_should_continue(indices):
            return any(
                len(rollout["obs_env"][env_index]) < self.n_steps
                for env_index in indices
            )

        self.collect_double_buffered_steps(partial_group_should_continue, rollout)

        progress_bar.close()

        for i in range(self.num_envs):
            if rollout["steps_in_current_segment"][i] > 0:
                segment_lens.append(int(rollout["steps_in_current_segment"][i]))

        finished_episodes = len(completed_ep_returns)
        self.env.post_rollout()
        self.logger["skipped_invalid_steps"] = rollout["skipped_invalid_steps"]
        self.logger["skipped_invalid_nodes"] = rollout.get("skipped_invalid_nodes", 0)
        self.store_rollout_timers()
        self.episode_stats.end_rollout()

        last_done = rollout["last_done"]
        to_bootstrap = [
            (i, self.obs[i]) for i in range(self.num_envs) if not last_done[i]
        ]
        if to_bootstrap:
            indices, obs_batch = zip(*to_bootstrap)
            values = self.get_value(Batch.from_data_list(obs_batch))
            values_np = values.detach().cpu().numpy()
            value_map = {i: float(v) for i, v in zip(indices, values_np)}
        else:
            value_map = {}

        next_values_per_env = [
            value_map.get(i, 0.0) for i in range(self.num_envs)
        ]

        batch_returns, batch_advantages = self.compute_gae_per_env(
            rews_env, dones_env, vals_env, next_values_per_env
        )

        batch_obs = [obs for sublist in obs_env for obs in sublist]
        batch_env_ids = [
            env_index
            for env_index, observations in enumerate(obs_env)
            for _observation in observations
        ]
        batch_acts = [act for sublist in acts_env for act in sublist]
        batch_log_probs = torch.tensor(
            [lp for sublist in log_probs_env for lp in sublist],
            dtype=torch.float32,
            device=self.args.device,
        )
        if batch_log_probs.numel() == 0:
            batch_log_probs = torch.zeros(
                0, NUM_LOGP_FACTORS, dtype=torch.float32, device=self.args.device
            )
        elif batch_log_probs.ndim == 1:
            batch_log_probs = batch_log_probs.unsqueeze(-1)

        self._set_invalid_action_rate(
            rollout["skipped_invalid_steps"], len(batch_obs)
        )
        self.logger["batch_lens"] = completed_ep_lens

        if self.pallete.max_val_len > 0:
            self.logger["value_head_log_std"] = (
                self.ac.actor.value_decoder.log_std.item()
                if hasattr(self.ac.actor.value_decoder, "log_std")
                else 0.0
            )

        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_returns,
            batch_advantages,
            segment_lens,
            finished_episodes,
            batch_env_ids,
        )

    def compute_gae_per_env(
        self, rewards_per_env, dones_per_env, values_per_env, next_values_per_env
    ):
        """
        GAE over per-env lists of Python/numpy scalars. Empty envs are skipped.
        """
        all_returns = []
        all_advantages = []
        gamma = self.gamma
        lam = self.gae_lambda

        for rews, dones, vals, next_val in zip(
            rewards_per_env, dones_per_env, values_per_env, next_values_per_env
        ):
            T = len(rews)
            if T == 0:
                continue
            rews = np.asarray(rews, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.float32)
            vals = np.asarray(vals, dtype=np.float32)
            next_val = float(next_val)
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                next_value = next_val if t == T - 1 else vals[t + 1]
                next_done = dones[t]
                delta = rews[t] + gamma * next_value * (1.0 - next_done) - vals[t]
                gae = delta + gamma * lam * (1.0 - next_done) * gae
                advantages[t] = gae
            all_returns.append(advantages + vals)
            all_advantages.append(advantages)

        if not all_returns:
            empty = torch.zeros(0, dtype=torch.float32, device=self.args.device)
            return empty, empty

        returns_flat = torch.as_tensor(
            np.concatenate(all_returns),
            dtype=torch.float32,
            device=self.args.device,
        )
        advantages_flat = torch.as_tensor(
            np.concatenate(all_advantages),
            dtype=torch.float32,
            device=self.args.device,
        )
        return returns_flat, advantages_flat

    def _ac_kwargs(self, model_dir=None):
        encoder_residual = self.encoder_residual
        node_global_context = self.node_global_context
        slot_embedding = self.slot_embedding
        rnn_type = self.rnn_type
        value_head = self.value_head
        value_bins = self.value_bins
        value_ordinal = self.value_ordinal
        node_decoder = self.node_decoder
        attn_heads = self.attn_heads
        attn_tanh_clip = self.attn_tanh_clip
        graph_encoder = self.graph_encoder
        encoder_heads = self.encoder_heads
        encoder_layers = self.encoder_layers
        encoder_ff = self.encoder_ff
        encoder_reverse_edges = self.encoder_reverse_edges
        time_feat_dim = 0
        config_candidates = []
        if model_dir:
            config_candidates.append(os.path.join(model_dir, "model_config.json"))
        if getattr(self.args, "model_path", None):
            config_candidates.append(
                os.path.join(self.args.model_path, "model_config.json")
            )
        for path in config_candidates:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    cfg = json.load(f)
                encoder_residual = _as_bool(
                    cfg.get("encoder_residual", encoder_residual), encoder_residual
                )
                node_global_context = _as_bool(
                    cfg.get("node_global_context", node_global_context),
                    node_global_context,
                )
                slot_embedding = _as_bool(
                    cfg.get("slot_embedding", slot_embedding), slot_embedding
                )
                rnn_type = cfg.get("rnn_type", rnn_type)
                # Old checkpoints omit these keys; they were trained with the Gaussian head.
                value_head = cfg.get("value_head", "gaussian")
                value_bins = int(cfg.get("value_bins", value_bins))
                value_ordinal = _as_bool(
                    cfg.get("value_ordinal", value_ordinal), value_ordinal
                )
                node_decoder = cfg.get("node_decoder", "rnn")
                attn_heads = int(cfg.get("attn_heads", 8))
                attn_tanh_clip = float(cfg.get("attn_tanh_clip", 10.0))
                graph_encoder = cfg.get("graph_encoder", "sage")
                encoder_heads = int(cfg.get("encoder_heads", 4))
                encoder_layers = int(cfg.get("encoder_layers", 3))
                encoder_ff = _as_bool(cfg.get("encoder_ff", False), False)
                encoder_reverse_edges = _as_bool(
                    cfg.get("encoder_reverse_edges", False), False
                )
                time_feat_dim = int(cfg.get("time_feat_dim", 0))
                break
        return dict(
            rnn_layer=rnn_type,
            encoder_residual=encoder_residual,
            node_global_context=node_global_context,
            slot_embedding=slot_embedding,
            value_head=value_head,
            num_bins=int(value_bins),
            ordinal=_as_bool(value_ordinal, True),
            node_decoder=node_decoder,
            attn_heads=int(attn_heads),
            attn_tanh_clip=float(attn_tanh_clip),
            graph_encoder=graph_encoder,
            encoder_heads=int(encoder_heads),
            encoder_layers=int(encoder_layers),
            encoder_ff=_as_bool(encoder_ff, False),
            encoder_reverse_edges=_as_bool(encoder_reverse_edges, False),
            time_feat_dim=int(time_feat_dim),
        )

    def _action_has_invalid_nodes(self, action_i):
        nodes = action_i[1] if action_i is not None and len(action_i) > 1 else None
        if nodes is None:
            return False
        if not torch.is_tensor(nodes):
            nodes = torch.as_tensor(nodes)
        return bool((nodes < 0).any())

    def _factor_stats(
        self,
        action_dist,
        action,
        action_mask=None,
        nodes_dist=None,
        nodes=None,
        seq_len_mask=None,
        values_dist_tt=None,
        values_dist=None,
        values_raw=None,
        logp_only=False,
    ):
        """Per-factor log-probs and, unless ``logp_only``, entropies.

        The rollout asks for log-probs only and returns before the entropy,
        valid-count and normalization work, including the action-mask copy.
        """
        B = action.shape[0]
        device = action.device
        zeros = torch.zeros(B, device=device, dtype=torch.float32)

        skip_rule = (
            len(action_dist.param_shape) == 2 and action_dist.param_shape[1] == 1
        )
        if skip_rule:
            logp_rule = zeros
        else:
            logp_rule = action_dist.log_prob(action)

        logp_node = zeros.clone()
        ent_node = zeros.clone()
        ent_node_norm = zeros.clone()
        n_valid_node = zeros.clone()
        if nodes_dist is not None and nodes is not None:
            if seq_len_mask is None:
                seq_len_mask = torch.ones(
                    nodes.shape[0], nodes.shape[-1], dtype=torch.bool, device=device
                )
            else:
                seq_len_mask = seq_len_mask.to(device=device)
                if seq_len_mask.dtype != torch.bool:
                    seq_len_mask = seq_len_mask.bool()
            L = min(nodes.shape[-1], seq_len_mask.shape[-1], nodes_dist.logits.shape[1])
            nodes = nodes[..., :L]
            seq_len_mask = seq_len_mask[..., :L]
            valid_mask = (nodes >= 0) & seq_len_mask
            nodes_safe = torch.clamp(nodes, min=0)
            logits = nodes_dist.logits[:, :L]
            node_dist_l = Categorical(logits=logits)
            node_log_probs = node_dist_l.log_prob(nodes_safe)
            logp_node = (node_log_probs * valid_mask).sum(dim=1)
            if not logp_only:
                slot_ent = node_dist_l.entropy()
                seq_f = seq_len_mask.float()
                ent_node = (slot_ent * seq_f).sum(dim=1)
                n_valid_t = (logits > (LOGIT_MASK / 2)).sum(dim=-1).float()
                slot_ent_norm = slot_ent / torch.log(n_valid_t.clamp(min=2.0))
                n_slots = seq_f.sum(dim=1).clamp(min=1.0)
                ent_node_norm = (slot_ent_norm * seq_f).sum(dim=1) / n_slots
                n_valid_node = (n_valid_t * seq_f).sum(dim=1) / n_slots

        logp_val = zeros.clone()
        if values_dist_tt is not None and values_raw is not None:
            if isinstance(values_dist_tt, Categorical):
                idx = values_raw.long()
                logits = values_dist_tt.logits
                if idx.dim() == 1:
                    idx = idx.unsqueeze(-1)
                L = min(idx.shape[-1], logits.shape[1])
                idx = idx[..., :L]
                val_dist = Categorical(logits=logits[:, :L])
                logp_val = val_dist.log_prob(idx).sum(dim=1)
            else:
                values_raw = values_raw.clamp(
                    min=-1 + eps_log_clip, max=1 - eps_log_clip
                )
                logp_val = values_dist_tt.log_prob(values_raw).sum(dim=1)

        logp = torch.stack([logp_rule, logp_node, logp_val], dim=-1)
        if logp_only:
            return {"logp": logp}

        if skip_rule:
            ent_rule = zeros
            n_valid_rule = torch.ones(B, device=device, dtype=torch.float32)
        else:
            ent_rule = action_dist.entropy()
            if action_mask is not None:
                action_mask = action_mask.to(device=device)
                n_valid_rule = action_mask.reshape(B, -1).sum(dim=-1).float()
            else:
                n_valid_rule = torch.full(
                    (B,),
                    float(action_dist.logits.shape[-1]),
                    device=device,
                    dtype=torch.float32,
                )
        ent_rule_norm = ent_rule / torch.log(n_valid_rule.clamp(min=2.0))

        ent_val = zeros.clone()
        ent_val_norm = None
        if values_dist_tt is not None and values_raw is not None:
            if isinstance(values_dist_tt, Categorical):
                ent_src = values_dist if values_dist is not None else values_dist_tt
                ent_val = ent_src.entropy()
                if ent_val.dim() > 1:
                    ent_val = ent_val[:, :L].sum(dim=1)
                K = logits.shape[-1]
                ent_val_norm = ent_val / (
                    float(max(L, 1)) * math.log(max(float(K), 2.0))
                )
            elif values_dist is not None:
                ent_val = values_dist.entropy().sum(dim=1)

        return {
            "logp": logp,
            "logp_rule": logp_rule,
            "logp_node": logp_node,
            "logp_val": logp_val,
            "ent_rule": ent_rule,
            "ent_node": ent_node,
            "ent_val": ent_val,
            "ent_rule_norm": ent_rule_norm,
            "ent_node_norm": ent_node_norm,
            "ent_val_norm": ent_val_norm,
            "n_valid_rule": n_valid_rule,
            "n_valid_node": n_valid_node,
        }

    @torch.no_grad()
    def get_action_and_value(self, obs, action_only=False):
        """Sample actions for a list of graphs.

        ``action_only`` skips the critic, log-probabilities and the node-mask
        copy. The training rollout still returns those, without entropy stats.
        """
        batch = Batch.from_data_list(obs)
        model_inputs = (
            batch.x.to(self.args.device),
            batch.edge_index.to(self.args.device),
            torch.squeeze(batch.edge_attr, dim=1).to(self.args.device),
            batch.batch.to(self.args.device),
            batch.nodes.to(self.args.device),
            batch.action_mask.to(self.args.device),
            batch.action_node_mask.to(self.args.device),
        )
        time_feat = self._batch_time_feat(batch)
        if action_only:
            (
                action,
                _action_dist,
                nodes,
                nodes_dist,
                _seq_len_mask,
                node_mask,
                values,
                values_dist_tt,
                _values_dist,
            ) = self.ac.act(*model_inputs, time_feat=time_feat)
            total_log_prob = None
            V = None
        else:
            (
                action,
                action_dist,
                nodes,
                nodes_dist,
                seq_len_mask,
                node_mask,
                values,
                values_dist_tt,
                values_dist,
            ), V = self.ac(*model_inputs, time_feat=time_feat)
            stats = self._factor_stats(
                action_dist,
                action,
                nodes_dist=nodes_dist,
                nodes=nodes,
                seq_len_mask=seq_len_mask,
                values_dist_tt=values_dist_tt,
                values_dist=values_dist,
                values_raw=values[1] if values_dist_tt is not None else None,
                logp_only=True,
            )
            total_log_prob = stats["logp"].detach()

        actions_cpu = action.detach().cpu()
        nodes_cpu = nodes.detach().cpu() if nodes_dist is not None else None
        values_real_cpu = values[0].detach().cpu() if values_dist_tt is not None else None
        values_raw_cpu = values[1].detach().cpu() if values_dist_tt is not None else None
        batch_size = actions_cpu.shape[0]
        batch_action = []

        for i in range(batch_size):
            sample = (
                actions_cpu[i],
                nodes_cpu[i].unsqueeze(0) if nodes_cpu is not None else None,
                (
                    (values_real_cpu[i].unsqueeze(0), values_raw_cpu[i].unsqueeze(0))
                    if values_raw_cpu is not None
                    else None
                ),
            )
            batch_action.append(sample)

        if action_only:
            return (batch_action, None, None), None

        if node_mask is not None:
            batch_node_masks = list(node_mask.detach().cpu().unbind(dim=0))
        else:
            batch_node_masks = [None] * batch_size
        lp_list = total_log_prob.detach().cpu().tolist()
        return (
            batch_action,
            lp_list,
            batch_node_masks,
        ), V.detach().cpu().numpy()

    def get_value(self, obs):
        """
        Queries the critic network for a value V(s) for each observation in obs.

        Parameters:
            obs - the observations at the current timestep

        Return:
            V - the predicted values of obs
        """
        # Query the critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        with torch.no_grad():
            V = self.ac.evaluate(
                obs.x.to(self.args.device),
                obs.edge_index.to(self.args.device),
                torch.squeeze(obs.edge_attr, dim=1).to(self.args.device),
                obs.batch.to(self.args.device),
                time_feat=self._batch_time_feat(obs),
            ).squeeze()
        V = V.unsqueeze(0) if V.dim() == 0 else V
        return V

    def get_action(self, obs):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        (
            action,
            action_dist,
            nodes,
            nodes_dist,
            node_seq_mask,
            node_dim_masks,
            values,
            values_dist_tt,
            _,
        ) = self.ac.act(
            obs.x.to(self.args.device),
            obs.edge_index.to(self.args.device),
            torch.squeeze(obs.edge_attr, dim=1).to(self.args.device),
            obs.batch.to(self.args.device),
            obs.nodes.to(self.args.device),
            obs.action_mask.to(self.args.device),
            obs.action_node_mask.to(self.args.device),
            time_feat=self._batch_time_feat(obs),
        )

        stats = self._factor_stats(
            action_dist,
            action,
            action_mask=obs.action_mask,
            nodes_dist=nodes_dist,
            nodes=nodes,
            seq_len_mask=node_seq_mask,
            values_dist_tt=values_dist_tt,
            values_dist=None,
            values_raw=values[1] if values_dist_tt is not None else None,
        )
        total_log_prob = stats["logp"].detach()

        actions = action.detach()  # Shape: (batch_size,)
        batch_size = actions.shape[0]
        batch_action = []

        values_real = values[0].detach() if values_dist_tt is not None else None
        values_raw = values[1].detach() if values_dist_tt is not None else None

        for i in range(batch_size):
            sample = (
                actions[i],
                nodes[i].unsqueeze(0) if nodes_dist is not None else None,
                (
                    (values_real[i].unsqueeze(0), values_raw[i].unsqueeze(0))
                    if values_dist_tt is not None
                    else None
                ),
            )
            batch_action.append(sample)
        return (
            batch_action,
            total_log_prob.tolist(),
        )

    def evaluate(self, batch, batch_act, batch_nodes=None, batch_vals_raw=None):
        """
        Estimate V(s) and log pi(a|s) for a mini-batch already on device.
        """
        force_nodes = batch.force_nodes if "force_nodes" in batch else None
        force_mask = batch.force_nodes_mask if "force_nodes_mask" in batch else None
        tf_action = TFAction(batch.force_act, force_nodes, force_mask, None)

        (
            _,
            action_dist,
            _,
            nodes_dist,
            seq_len_mask,
            _,
            _,
            values_dist_tt,
            values_dist,
        ), V = self.ac(
            batch.x,
            batch.edge_index,
            torch.squeeze(batch.edge_attr, dim=1),
            batch.batch,
            batch.nodes,
            batch.action_mask,
            batch.action_node_mask,
            tf_action,
            time_feat=self._batch_time_feat(batch),
        )

        if self.debug_nan and batch_nodes is not None and seq_len_mask is not None:
            L = min(batch_nodes.shape[-1], seq_len_mask.shape[-1])
            invalid = (batch_nodes[..., :L] < 0) & seq_len_mask[..., :L].bool()
            if invalid.any():
                f_logger.error(
                    f"[NaN Debug] evaluate received {int(invalid.sum())} invalid (-1) node indices"
                )

        stats = self._factor_stats(
            action_dist,
            batch_act,
            action_mask=batch.action_mask,
            nodes_dist=nodes_dist,
            nodes=batch_nodes,
            seq_len_mask=seq_len_mask,
            values_dist_tt=values_dist_tt,
            values_dist=values_dist,
            values_raw=batch_vals_raw,
        )
        return V.squeeze(), stats

    def needs_padding(self, tensor_list):
        """
        Check if a list of 1D tensors has varying lengths.

        Args:
            tensor_list (list of torch.Tensor): List of 1D tensors (sequence of actions/values)

        Returns:
            bool: True if padding is needed (different lengths), False otherwise.
        """
        lengths = [t.shape[1] for t in tensor_list]
        return len(set(lengths)) > 1

    def pad_batch_sequences(self, batch_acts_nodes, padding_value=0):
        """
        Pads a list of tensors (each with shape [1, seq_len]) along dim=1
        to the maximum sequence length in the batch.

        Args:
            batch_acts_nodes (list of torch.Tensor): list of tensors with shape [1, seq_len]
            padding_value (int, optional): value to pad with. Defaults to -1.

        Returns:
            padded_acts (torch.Tensor): shape [batch_size, 1, max_seq_len]
            mask (torch.Tensor): shape [batch_size, 1, max_seq_len] (1 for valid positions, 0 for padding)
        """
        # Find max sequence length
        max_len = max(tensor.shape[1] for tensor in batch_acts_nodes)

        padded_acts = []
        mask = []

        for acts in batch_acts_nodes:
            seq_len = acts.shape[1]
            pad_len = max_len - seq_len

            # Pad (left=0, right=pad_len) along dim=1
            padded = torch.nn.functional.pad(acts, (0, pad_len), value=padding_value)
            padded_acts.append(padded)

            # Create mask: 1 for valid positions, 0 for padding
            seq_mask = torch.nn.functional.pad(
                torch.ones_like(acts), (0, pad_len), value=0
            )
            mask.append(seq_mask)

        # Stack into batch tensors
        padded_acts = torch.cat(padded_acts, dim=0)  # [batch_size, 1, max_len]
        mask = torch.cat(mask, dim=0)  # [batch_size, 1, max_len]

        return padded_acts, mask

    def _apply_saved_time_feature_config(self):
        """An existing model config decides whether the time feature is on.

        A missing ``time_feat_dim`` is an old checkpoint and disables the feature
        so the loaded weights still match the network.
        """
        candidates = []
        ac_model = getattr(self.args, "ac_model", "") or ""
        if ac_model:
            candidates.append(os.path.join(os.path.dirname(ac_model), "model_config.json"))
        model_path = getattr(self.args, "model_path", None)
        if model_path:
            candidates.append(os.path.join(model_path, "model_config.json"))
        for path in candidates:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            if "time_feat_dim" not in config or int(config.get("time_feat_dim", 0)) <= 0:
                self.time_feature = False
                self.time_scale = None
                return
            self.time_feature = True
            saved_scale = config.get("time_scale", None)
            if saved_scale not in (None, 0, 0.0):
                self.time_scale = float(saved_scale)
            return

    def _resolve_rollout_length(self):
        """Set n_steps from 'auto', an explicit value, or timesteps_per_batch."""
        max_horizon = max(self.env.max_steps)
        self.max_horizon = max_horizon
        if self.n_steps == "auto":
            if self.timesteps_per_batch is not None:
                self.n_steps = math.ceil(self.timesteps_per_batch / self.num_envs)
            else:
                self.n_steps = max_horizon
        else:
            self.n_steps = int(self.n_steps)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.n_steps < max_horizon:
            warnings.warn(
                f"n_steps={self.n_steps} is below the longest horizon {max_horizon}; "
                "some episodes will not finish inside one rollout.",
                stacklevel=2,
            )
        self.timesteps_per_batch = self.num_envs * self.n_steps

    def _write_model_config(self):
        model_config = {
            "encoder_residual": self.encoder_residual,
            "node_global_context": self.node_global_context,
            "slot_embedding": self.slot_embedding,
            "rnn_type": self.rnn_type,
            "hidden_dim": self.args.hidden_dim,
            "encoder": getattr(self, "encoder", self.args.rl_hyperparams.get("encoder")),
            "entropy_coef": self.entropy_coef,
            "entropy_coef_rule": self.entropy_coef_rule,
            "entropy_coef_node": self.entropy_coef_node,
            "entropy_coef_val": self.entropy_coef_val,
            "entropy_normalize": self.entropy_normalize,
            "target_kl": self.target_kl,
            "value_head": self.value_head,
            "value_bins": int(self.value_bins),
            "value_ordinal": self.value_ordinal,
            "node_decoder": self.node_decoder,
            "attn_heads": int(self.attn_heads),
            "attn_tanh_clip": float(self.attn_tanh_clip),
            "graph_encoder": self.graph_encoder,
            "encoder_heads": int(self.encoder_heads),
            "encoder_layers": int(self.encoder_layers),
            "encoder_ff": self.encoder_ff,
            "encoder_reverse_edges": self.encoder_reverse_edges,
            "time_feat_dim": self.time_feat_dim,
            "time_scale": float(self.time_scale) if self.time_feature else 0.0,
        }
        with open(
            os.path.join(self.args.model_path, "model_config.json"),
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(model_config, config_file, indent=2)

    def _batch_time_feat(self, batch):
        time_feat = getattr(batch, "time_feat", None)
        if time_feat is None:
            return None
        return time_feat.to(self.args.device)

    def _set_invalid_action_rate(self, skipped, stored):
        if stored <= 0:
            self.logger["invalid_action_rate"] = float("nan")
            return
        self.logger["invalid_action_rate"] = float(skipped) / float(stored)

    def _consider_early_stopping(self, metric, update_step, timesteps, from_evaluation=False):
        """Tick early stopping only on a fresh, finite metric.

        By default that is a windowed training macro return after enough new
        episodes. With ``--rl_es_metric eval_*`` it is the normalized return of
        each evaluation phase, so patience counts evaluation phases.
        """
        if from_evaluation:
            ready = math.isfinite(metric)
        else:
            ready = (
                self.episode_stats.episodes_since_tick >= self.es_tick_episodes
                and math.isfinite(metric)
            )
        if not ready:
            return False, False, False
        if not from_evaluation:
            self.episode_stats.consume_tick()
        should_stop, improved = self.early_stopping.step(
            metric, update_step, timesteps
        )
        return should_stop, improved, True

    def _save_named_checkpoint(self, update_step, timesteps, metric, improved):
        if not improved or not math.isfinite(metric):
            return
        torch.save(
            self.ac.state_dict(),
            os.path.join(
                self.args.model_path,
                f"ac_{update_step}-{timesteps}-{metric:.3f}.pth",
            ),
        )

    def _per_instance_explained_variance(self, returns, predicted_values, env_ids):
        if not env_ids:
            return {}
        instance_of_env = np.asarray(self.env.instance_ids, dtype=np.int64)
        sample_instance = instance_of_env[np.asarray(env_ids, dtype=np.int64)]
        by_name = {}
        for instance_id, name in enumerate(self.env.instance_names):
            mask = torch.as_tensor(
                sample_instance == instance_id, device=returns.device
            )
            by_name[name] = explained_variance(returns[mask], predicted_values[mask])
        return by_name

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
            hyperparameters - the extra arguments included when creating the PPO model, should only include
                                hyperparameters defined below with custom values.

        Return:
            None
        """
        # Initialize default values for hyperparameters
        # None means "not set": n_steps="auto" then uses the longest horizon.
        self.timesteps_per_batch = None
        self.n_steps = "auto"
        self.mini_batch_size = 100  # Number of timesteps per mini batch
        self.n_updates_per_iteration = 4  # Updates of actor/critic per iteration
        self.lr = 3e-4  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.clip = 0.2  # Recommended 0.2
        self.rnn_type = "gru"
        self.gae_lambda = 0.95  # Lambda for GAE, recommended
        # Once rewards are normalized to about unit scale, 0.01 is a larger share of the loss.
        self.entropy_coef = 0.01  # Coefficient for entropy loss
        self.entropy_coef_rule = None
        self.entropy_coef_node = None
        self.entropy_coef_val = None
        self.entropy_normalize = True
        self.target_kl = 0.02
        self.encoder_residual = True
        self.node_global_context = True
        self.slot_embedding = True
        self.value_head = "bins"
        self.value_bins = 11
        self.value_ordinal = True
        self.node_decoder = "rnn"
        self.attn_heads = 8
        self.attn_tanh_clip = 10.0
        self.graph_encoder = "sage"
        self.encoder_heads = 4
        self.encoder_layers = 3
        self.encoder_ff = False
        self.encoder_reverse_edges = False

        # Reward normalization settings
        self.normalize_rewards = True
        self.reward_clip = 10.0
        self.rew_norm_freeze_after = -1
        self.time_feature = True
        self.time_scale = None
        self.es_patience = 10  # Early stopping patience  (iterations)
        self.es_min_delta = 0.01  # Minimum change in reward to qualify as
        self.es_min_delta_rel = 0.01
        self.es_window = 5
        self.es_warmup_episodes = 5
        self.es_tick_episodes = None
        self.es_min_timesteps = (
            100_000  # Minimum timesteps before early stopping can kick in
        )

        # Miscellaneous parameters
        self.render = False  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            try:
                exec("self." + param + " = " + str(val))
            except NameError:
                exec("self." + param + ' = "' + str(val) + '"')

        if self.entropy_coef_rule is None:
            self.entropy_coef_rule = self.entropy_coef
        if self.entropy_coef_node is None:
            self.entropy_coef_node = self.entropy_coef
        if self.entropy_coef_val is None:
            self.entropy_coef_val = self.entropy_coef
        self.entropy_normalize = _as_bool(self.entropy_normalize, True)
        self.encoder_residual = _as_bool(self.encoder_residual, True)
        self.node_global_context = _as_bool(self.node_global_context, True)
        self.slot_embedding = _as_bool(self.slot_embedding, True)
        self.value_head = str(self.value_head)
        if self.value_head not in ("bins", "gaussian"):
            raise ValueError(
                f"Unsupported value_head '{self.value_head}'. Use 'bins' or 'gaussian'."
            )
        self.value_bins = int(self.value_bins)
        self.value_ordinal = _as_bool(self.value_ordinal, True)
        self.node_decoder = str(self.node_decoder)
        if self.node_decoder not in ("rnn", "attention"):
            raise ValueError(
                f"Unsupported node_decoder '{self.node_decoder}'. Use 'rnn' or 'attention'."
            )
        self.attn_heads = int(self.attn_heads)
        self.attn_tanh_clip = float(self.attn_tanh_clip)
        self.graph_encoder = str(self.graph_encoder)
        if self.graph_encoder not in ("sage", "gatv2", "transformer"):
            raise ValueError(
                f"Unsupported graph_encoder '{self.graph_encoder}'. Use 'sage', 'gatv2', or 'transformer'."
            )
        self.encoder_heads = int(self.encoder_heads)
        self.encoder_layers = int(self.encoder_layers)
        self.encoder_ff = _as_bool(self.encoder_ff, False)
        self.encoder_reverse_edges = _as_bool(self.encoder_reverse_edges, False)
        if self.target_kl in (None, "None"):
            self.target_kl = 0.0
        self.target_kl = float(self.target_kl)
        self.normalize_rewards = _as_bool(self.normalize_rewards, True)
        self.time_feature = _as_bool(self.time_feature, True)
        self.reward_clip = float(self.reward_clip)
        self.rew_norm_freeze_after = int(self.rew_norm_freeze_after)
        self.es_window = int(self.es_window)
        self.es_warmup_episodes = int(self.es_warmup_episodes)
        self.es_min_delta_rel = float(self.es_min_delta_rel)
        if isinstance(self.n_steps, str) and self.n_steps.strip().lower() == "auto":
            self.n_steps = "auto"
        else:
            self.n_steps = int(self.n_steps)
        if self.timesteps_per_batch is not None:
            self.timesteps_per_batch = int(self.timesteps_per_batch)
        if self.time_scale in (None, "None"):
            self.time_scale = None
        else:
            self.time_scale = float(self.time_scale)
        if self.es_tick_episodes in (None, "None"):
            self.es_tick_episodes = None
        else:
            self.es_tick_episodes = int(self.es_tick_episodes)

    def _log_grouped_returns(self, label, names, stats):
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

    def _add_grouped_return_logs(self, log_val, label, names, stats):
        for index, name in enumerate(names):
            key = _metric_name(name) if label == "inst" else str(name)
            mean = stats["means"][index]
            length = stats["mean_lengths"][index]
            if np.isfinite(mean):
                log_val[f"return_{label}_{key}"] = float(mean)
            if np.isfinite(length):
                log_val[f"length_{label}_{key}"] = float(length)
            log_val[f"episodes_{label}_{key}"] = int(stats["counts"][index])

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.
        Now includes rolling buffer statistics.
        """
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        e_so_far = self.logger["e_so_far"]
        i_so_far = self.logger["i_so_far"]

        avg_ep_rews = self.episode_stats.get_last_rollout_pooled_return()
        macro_ep_rews = self.episode_stats.get_last_rollout_macro_return()
        per_step_rews = self.episode_stats.get_last_rollout_per_step_return()
        env_stats = self.episode_stats.get_last_rollout_env_stats()
        instance_stats = self.episode_stats.get_last_rollout_instance_stats()
        log_per_instance = len(self.env.instance_names) < self.num_envs

        avg_ep_lens = (
            float(np.mean(self.logger["batch_lens"]))
            if self.logger["batch_lens"]
            else 0.0
        )
        # Get rolling buffer statistics
        buffer_stats = self.episode_stats.get_stats()
        recent_stats = self.episode_stats.get_recent_stats(50)

        def _mean_logged(key, default=0.0):
            vals = self.logger.get(key, [])
            if not vals:
                return default
            return float(
                np.mean(
                    [
                        v.cpu().float().mean().item() if torch.is_tensor(v) else float(v)
                        for v in vals
                    ]
                )
            )

        avg_actor_loss = _mean_logged("actor_losses")
        avg_critic_loss = _mean_logged("critic_losses")
        avg_entropy = _mean_logged("entropy")
        avg_entropy_rule = _mean_logged("entropy_rule")
        avg_entropy_node = _mean_logged("entropy_node")
        avg_entropy_val = _mean_logged("entropy_val")
        avg_entropy_rule_norm = _mean_logged("entropy_rule_norm")
        avg_entropy_node_norm = _mean_logged("entropy_node_norm")
        avg_entropy_val_norm = _mean_logged("entropy_val_norm")
        avg_kl = _mean_logged("approx_kl")
        avg_kl_k3 = _mean_logged("approx_kl_k3")
        avg_kl_rule = _mean_logged("approx_kl_rule")
        avg_kl_node = _mean_logged("approx_kl_node")
        avg_kl_val = _mean_logged("approx_kl_val")
        avg_clip_frac = _mean_logged("clip_fraction")
        avg_ratio_min = _mean_logged("ratio_min")
        avg_ratio_max = _mean_logged("ratio_max")
        avg_adv_mean = _mean_logged("adv_mean")
        avg_adv_std = _mean_logged("adv_std")
        avg_grad_norm = _mean_logged("grad_norm")
        avg_grad_norm_critic = _mean_logged("grad_norm_critic")
        explained_variance = float(self.logger.get("explained_variance", 0.0) or 0.0)
        epochs_run = int(self.logger.get("epochs_run", 0) or 0)
        kl_early_stop = bool(self.logger.get("kl_early_stop", False))
        current_lr = float(self.logger.get("lr", 0.0) or 0.0)

        value_head_log_std = self.logger["value_head_log_std"]

        # Print logging statements
        f_logger.info(
            f"-------------------- Iteration #{i_so_far} --------------------"
        )
        f_logger.info(f"Batch Episodes: {len(self.logger['batch_lens'])}")
        f_logger.info(f"Average Episodic Length: {avg_ep_lens:.2f}")
        f_logger.info(f"Average Episodic Return: {avg_ep_rews:.3f}")
        f_logger.info(f"Macro Return (per-env mean): {macro_ep_rews:.3f}")
        f_logger.info(f"Per-step return: {per_step_rews:.5f}")
        if log_per_instance:
            self._log_grouped_returns(
                "inst", self.env.instance_names, instance_stats
            )
        else:
            self._log_grouped_returns(
                "env", [str(i) for i in range(self.num_envs)], env_stats
            )
        ev_by_instance = self.logger.get("explained_variance_by_instance") or {}
        finite_ev = [
            value
            for value in ev_by_instance.values()
            if isinstance(value, float) and math.isfinite(value)
        ]
        if finite_ev:
            f_logger.info(
                "  per-instance explained var: "
                f"min={min(finite_ev):.3f} median={float(np.median(finite_ev)):.3f}"
            )
        f_logger.info(
            f"Rolling Buffer Stats (last {buffer_stats['num_episodes']} episodes):"
        )
        f_logger.info(
            f"  Mean Return: {buffer_stats['mean_return']:.3f} ± {buffer_stats['std_return']:.3f}"
        )
        f_logger.info(
            f"Recent 50 Episodes Mean Return: {recent_stats['mean_return']:.3f}"
        )
        f_logger.info(f"Total Episodes Completed: {buffer_stats['total_episodes']}")
        f_logger.info(f"Average Loss: {avg_actor_loss:.5f}")
        f_logger.info(
            f"PPO diagnostics: kl={avg_kl:.5f} kl_k3={avg_kl_k3:.5f} "
            f"clip_frac={avg_clip_frac:.3f} explained_var={explained_variance:.3f} "
            f"epochs={epochs_run} kl_stop={kl_early_stop} lr={current_lr:.2e}"
        )
        f_logger.info(
            f"  factor KL: rule={avg_kl_rule:.5f} node={avg_kl_node:.5f} val={avg_kl_val:.5f}"
        )
        f_logger.info(
            f"  entropy: joint={avg_entropy:.4f} rule={avg_entropy_rule:.4f} "
            f"node={avg_entropy_node:.4f} val={avg_entropy_val:.4f} "
            f"(norm rule={avg_entropy_rule_norm:.4f} node={avg_entropy_node_norm:.4f} "
            f"val={avg_entropy_val_norm:.4f})"
        )
        f_logger.info(
            f"  ratio=[{avg_ratio_min:.3f}, {avg_ratio_max:.3f}] "
            f"adv_mean={avg_adv_mean:.4f} adv_std={avg_adv_std:.4f} "
            f"grad_norm={avg_grad_norm:.4f} grad_norm_critic={avg_grad_norm_critic:.4f}"
        )
        f_logger.info(f"Timesteps So Far: {t_so_far}")
        f_logger.info(f"Episodes So Far: {e_so_far}")
        invalid_rate = self.logger.get("invalid_action_rate", float("nan"))
        f_logger.info(
            f"Skipped invalid steps: {self.logger.get('skipped_invalid_steps', 0)} "
            f"(invalid_action_rate={invalid_rate})"
        )
        f_logger.info(
            f"Skipped invalid nodes: {self.logger.get('skipped_invalid_nodes', 0)}"
        )
        es_metric = self.logger.get("es_metric", float("nan"))
        f_logger.info(
            f"Early stopping: es_metric={es_metric} "
            f"es_ticked={int(self.logger.get('es_ticked', 0))} "
            f"fresh_instances={self.episode_stats.fresh_instance_count()}"
        )
        f_logger.info(
            "Rollout timers: env_wait={:.3f}s inference={:.3f}s decode={:.3f}s".format(
                float(self.logger.get("time_env_wait", 0.0)),
                float(self.logger.get("time_inference", 0.0)),
                float(self.logger.get("time_decode", 0.0)),
            )
        )
        eval_log = self._take_eval_log()
        self._log_eval_timers(eval_log)
        f_logger.info(f"Iteration took: {delta_t} secs")
        f_logger.info(f"------------------------------------------------------")

        # Enhanced wandb logging with buffer statistics
        log_val = {
            "iteration": i_so_far,
            "timesteps": t_so_far,
            "avg_episode_length": float(avg_ep_lens),
            "avg_episode_return": float(avg_ep_rews),
            "macro_episode_return": float(macro_ep_rews),
            "per_step_return": float(per_step_rews),
            "avg_actor_loss": float(avg_actor_loss),
            "avg_entropy": float(avg_entropy),
            "avg_entropy_rule": float(avg_entropy_rule),
            "avg_entropy_node": float(avg_entropy_node),
            "avg_entropy_val": float(avg_entropy_val),
            "avg_entropy_rule_norm": float(avg_entropy_rule_norm),
            "avg_entropy_node_norm": float(avg_entropy_node_norm),
            "avg_entropy_val_norm": float(avg_entropy_val_norm),
            "approx_kl": float(avg_kl),
            "approx_kl_k3": float(avg_kl_k3),
            "approx_kl_rule": float(avg_kl_rule),
            "approx_kl_node": float(avg_kl_node),
            "approx_kl_val": float(avg_kl_val),
            "clip_fraction": float(avg_clip_frac),
            "ratio_min": float(avg_ratio_min),
            "ratio_max": float(avg_ratio_max),
            "adv_mean": float(avg_adv_mean),
            "adv_std": float(avg_adv_std),
            "grad_norm": float(avg_grad_norm),
            "grad_norm_critic": float(avg_grad_norm_critic),
            "explained_variance": float(explained_variance),
            "epochs_run": epochs_run,
            "kl_early_stop": int(kl_early_stop),
            "lr": current_lr,
            "skipped_invalid_nodes": int(
                self.logger.get("skipped_invalid_nodes", 0) or 0
            ),
            "avg_critic_loss": float(avg_critic_loss),
            "std_parameter": float(value_head_log_std),
            # Rolling buffer stats
            "buffer_mean_return": buffer_stats["mean_return"],
            "buffer_std_return": buffer_stats["std_return"],
            "buffer_min_return": buffer_stats["min_return"],
            "buffer_max_return": buffer_stats["max_return"],
            "buffer_mean_length": buffer_stats["mean_length"],
            "recent_50_mean_return": recent_stats["mean_return"],
            "total_episodes_completed": buffer_stats["total_episodes"],
            "batch_episodes_count": len(self.logger["batch_lens"]),
            "time_env_wait": float(self.logger.get("time_env_wait", 0.0)),
            "time_inference": float(self.logger.get("time_inference", 0.0)),
            "time_decode": float(self.logger.get("time_decode", 0.0)),
        }
        bin_idxs = self.logger.get("value_bin_indices")
        if bin_idxs is not None:
            if torch.is_tensor(bin_idxs):
                bin_np = bin_idxs.detach().cpu().float().numpy()
            else:
                bin_np = np.asarray(bin_idxs, dtype=np.float32)
            if bin_np.size > 0:
                log_val["value_bin_mean"] = float(np.mean(bin_np))
                if self.args.use_wandb:
                    log_val["value_bin_indices"] = wandb.Histogram(bin_np)
        if log_per_instance:
            self._add_grouped_return_logs(log_val, "inst", self.env.instance_names, instance_stats)
        else:
            self._add_grouped_return_logs(
                log_val, "env", [str(i) for i in range(self.num_envs)], env_stats
            )
        for instance_id, name in enumerate(self.env.instance_names):
            scale = self.episode_stats.instance_return_scale[instance_id]
            if scale is None:
                continue
            log_val[f"return_scale_inst_{_metric_name(name)}"] = float(scale)
        ev_by_instance = self.logger.get("explained_variance_by_instance") or {}
        for name, ev in ev_by_instance.items():
            if isinstance(ev, float) and math.isfinite(ev):
                log_val[f"ev_inst_{_metric_name(name)}"] = ev
        for key in (
            "batch_returns_mean",
            "batch_returns_std",
            "invalid_action_rate",
            "es_metric",
        ):
            value = self.logger.get(key, float("nan"))
            if isinstance(value, float) and math.isfinite(value):
                log_val[key] = value
        log_val["es_ticked"] = int(self.logger.get("es_ticked", 0))
        log_val.update(eval_log)
        log_val["fresh_instance_count"] = int(self.episode_stats.fresh_instance_count())
        log_val = {
            key: value
            for key, value in log_val.items()
            if not (isinstance(value, float) and not math.isfinite(value))
        }

        if self.args.use_wandb:
            wandb.log(log_val)

        if self.args.verbose and hasattr(self, "writer"):
            for key, val in log_val.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self.writer.add_scalar(f"ppo/{key}", val, i_so_far)

        self.logger["avg_lens"].append(float(avg_ep_lens))
        self.logger["avg_rews"].append(float(avg_ep_rews))
        self.logger["avg_macro_rews"].append(float(macro_ep_rews))
        self.logger["avg_actor_losses"].append(float(avg_actor_loss))
        f_logger.info("Mean episodic rewards (pooled):")
        f_logger.info(self.logger["avg_rews"])
        f_logger.info("Mean episodic rewards (macro, per-env):")
        f_logger.info(self.logger["avg_macro_rews"])

        for variant, returns in self.episode_stats.eval_mean_returns.items():
            f_logger.info(f"Evaluation ({variant}): Mean episodic rewards:")
            f_logger.info([round(x, 3) for x in returns])

        self.logger["batch_lens"] = []

        self.logger["actor_losses"] = []
        self.logger["entropy"] = []
        self.logger["critic_losses"] = []
        self.logger["value_bin_indices"] = None
        for key in (
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
        ):
            self.logger[key] = []

    def pad_and_stack(self, tensors, pad_value=0):
        # Find the maximum sequence length

        # Pad each tensor to max_len
        padded_tensors = []
        for tensor in tensors:
            seq_len = tensor.shape[1]
            pad_amount = self.pallete.max_len - seq_len
            # Pad on the right side (dim=1) with `pad_value`
            padded_tensor = F.pad(tensor, (0, pad_amount), value=pad_value)
            padded_tensors.append(padded_tensor)

        # Stack them
        return torch.stack(padded_tensors, dim=0)

    def add_act_forcing_data_to_obs(self, obs, act, node_masks):
        # Shallow-copy so self.obs[i] stays free of force_* keys.
        # PyG collate takes keys from the first Data in the list; mutating
        # live observations would KeyError or drop attributes on the next Batch.
        force_act_data = {"force_act": act[0].unsqueeze(0)}

        if act[1] is not None:
            num_nodes = int(obs["nodes"].item())
            seq_len = act[1].shape[1]

            oneh_nodes = torch.zeros(
                num_nodes,
                self.pallete.max_len,
                dtype=torch.float32,
                device=act[1].device,
            )
            node_idx = act[1][0]
            valid = node_idx >= 0
            if valid.any():
                slots = torch.arange(seq_len, device=act[1].device)
                oneh_nodes[node_idx[valid], slots[valid]] = 1.0
            force_act_data["force_nodes"] = oneh_nodes

            nodes_mask = node_masks.T[:num_nodes]
            if nodes_mask.shape[1] < self.pallete.max_len:
                pad_width = self.pallete.max_len - nodes_mask.shape[1]
                nodes_mask = F.pad(nodes_mask, (0, pad_width), value=0.0)
            force_act_data["force_nodes_mask"] = nodes_mask

        obs_copy = copy.copy(obs)
        obs_copy.update(force_act_data)
        return obs_copy
