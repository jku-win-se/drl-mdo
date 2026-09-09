"""
The file contains the PPO class to train with.
NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
        It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import statistics
from networkx import nodes
import procbridge as pb
import math
import torch.nn.functional as F
from pathlib import Path

import numpy as np

import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.data import Batch
import wandb
from torch.utils.tensorboard import SummaryWriter
import os
from drl.env.RemEnv import SubprocEnv
from utils.seed_util import seed_everything
from nqdm import nqdm
from tqdm import tqdm
import logging
from collections import namedtuple
from drl.utils.episode_stats import EpisodeStats
from drl.utils.pallete import PalleteEncoder
from drl.utils.running_mean_std import RunningMeanStd
from drl.utils.balanced_sampler import BalancedMiniBatchSampler
from drl.neural_model.early_stopping import EarlyStopping
from paretoset import paretoset

TFAction = namedtuple("TForceAction", ["action", "nodes", "nodes_dim_mask", "values"])

f_logger = logging.getLogger(__name__)

eps_log_clip = 1e-6
eval_interval = 4  # Evaluate every n iterations during training


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

        # Track cumulative episode rewards and lengths across rollout boundaries (SB3 style)
        # These accumulate rewards/steps for in-progress episodes spanning multiple rollouts
        self._current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        # Initialize episode statistics buffer
        self.episode_stats = EpisodeStats(buffer_size=50)

        # Initialize reward normalization (OpenAI Baselines style)
        # Tracks discounted returns for normalization
        self.reward_normalizer = RunningMeanStd()

        # Track discounted returns for each environment (like OpenAI Baselines)
        self.returns = np.zeros(self.num_envs)

        # Initialize balanced mini-batch sampler
        self.balanced_sampler = BalancedMiniBatchSampler(self.num_envs)

        conn = pb.Client("127.0.0.1", args.port)
        conn.request("resetInstance", str(run_id))

        self.env = SubprocEnv(self.num_envs, args.metamodel, pallete)

        if self.args.rl_evaluate:
            print("Setting up evaluation environment (port 8001)...")
            conn_eval = pb.Client("127.0.0.1", 8001)
            conn_eval.request("resetInstance", os.path.join(str(run_id), "eval"))
            num_test_models = conn_eval.request("numInstances")
            self.eval_env = SubprocEnv(
                num_test_models, args.metamodel, pallete, False, 8001
            )

        self.obs = self.env.reset()

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
            rnn_layer=self.rnn_type,
        ).to(args.device)

        if args.no_nn_updates:
            for param in self.ac.parameters():
                param.requires_grad = False

        # Initialize optimizers for actor and critic
        if args.rl_hyperparams["encoder"] == "shared":
            self.ac_optim = Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
            decay_steps = int(args.rl_tsteps / math.ceil(self.timesteps_per_batch))
            print(f"Learning rate will be decayed over {decay_steps} steps")
            self.lr_scheduler = LinearLR(
                self.ac_optim, start_factor=1.0, end_factor=0.1, total_iters=decay_steps
            )

        elif args.rl_hyperparams["encoder"] == "separate":
            self.a_optim = Adam(self.ac.actor.parameters(), lr=self.lr, eps=1e-5)
            self.c_optim = Adam(self.ac.critic.parameters(), lr=self.lr, eps=1e-5)

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
        )

        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "e_so_far": 0,  # episodes so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "avg_lens": [],  # episodic lengths in batch
            "avg_test_rews": [],  # episodic returns on test set
            "avg_rews": [],  # all batch rewards
            "avg_mean_rews": [],  # all batch mean rewards
            "actor_losses": [],  # losses of actor network in current iteration
            "critic_losses": [],
            "avg_critic_losses": [],
            "avg_actor_losses": [],  # all actor losses
            "entropy": [],
            "test_model_bov": {},  # BOV of the model on test set
            "value_head_log_std": 0.0,
        }

        # Initialize NaN debugging tools (set debug_nan=True to enable)
        self.debug_nan = getattr(args, "debug_nan", False)

    def run_inference(self, num_samples, infer_set_single=False):
        if infer_set_single:
            solutions_name_rew = self.infer_set_single()
            return solutions_name_rew
        else:
            solution_set, sampled_episodes, mean_rew = self.infer(num_samples)
            return solution_set, num_samples, mean_rew

    def infer_set_single(self):

        num_eval_envs = len(self.obs)
        finished = np.zeros(num_eval_envs, dtype=bool)

        rews_env = [[] for _ in range(num_eval_envs)]
        solutions_name_rew = []
        progress_bar = tqdm()

        while not np.all(finished):
            (action, _, _), _ = self.get_action_and_value(self.obs)
            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.env.step(cpu_action)

            for i in range(num_eval_envs):
                if next_obs[i] is None:
                    continue

                rews_env[i].append(rew[i])

                if not finished[i]:
                    rews_env[i].append(rew[i])

                    if done[i]:
                        progress_bar.update(1)

                        env_i_solutions = self.env.get_env_solutions_and_clear(i)
                        env_i_name = self.env.get_env_model_name(i)
                        solutions_name_rew.append(
                            (env_i_solutions, env_i_name, sum(rews_env[i]))
                        )

                        rews_env[i] = []  # Clear for next episode
                        finished[i] = True

                self.obs[i] = next_obs[i]

        progress_bar.close()

        return solutions_name_rew

    def infer(self, num_samples):
        rews_env = [[] for _ in range(self.num_envs)]

        completed_rews = []

        completed_episodes = 0

        ep_lens = []
        t_env = np.zeros(self.num_envs, dtype=int)

        progress_bar = tqdm()

        # Track global Pareto front and no-improvement counter
        global_pareto_set = None  # numpy array of non-dominated solutions
        episodes_since_improvement = 0
        converged = False

        while True:
            if num_samples >= 0 and completed_episodes >= num_samples:
                break

            if num_samples < 0 and converged:
                break

            with torch.no_grad():
                (action, _, _), _ = self.get_action_and_value(self.obs)

            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.env.step(cpu_action)

            for i in range(self.num_envs):
                if next_obs[i] is None:
                    continue

                rews_env[i].append(rew[i])
                t_env[i] += 1
                self.obs[i] = next_obs[i]

                if done[i]:
                    ep_lens.append(t_env[i])
                    completed_rews.append(rews_env[i].copy())
                    rews_env[i] = []  # Clear for next episode

                    progress_bar.update(1)
                    completed_episodes += 1
                    t_env[i] = 0

                    env_i_solutions = self.env.get_env_solutions_and_clear(i)

                    global_pareto_set, improved = self.update_pareto_front(
                        global_pareto_set, env_i_solutions
                    )
                    episodes_since_improvement = (
                        0 if improved else episodes_since_improvement + 1
                    )
                    print(
                        f"Pareto size: {len(global_pareto_set) if global_pareto_set is not None else 0} | Episodes since improvement: {episodes_since_improvement}"
                    )

                if episodes_since_improvement >= self.num_envs:
                    converged = True

        progress_bar.close()

        mean_rew = statistics.mean([sum(r) for r in completed_rews])
        max_rew = max([sum(r) for r in completed_rews])
        min_rew = min([sum(r) for r in completed_rews])
        stdev_rew = (
            statistics.stdev([sum(r) for r in completed_rews])
            if len(completed_rews) > 1
            else 0
        )
        print(
            f"Mean reward: {mean_rew} ({stdev_rew}), Max reward: {max_rew}, Min reward: {min_rew}"
        )
        print("Pareto set: ", global_pareto_set)
        return (
            global_pareto_set.tolist() if global_pareto_set is not None else [],
            completed_episodes,
            mean_rew,
        )

    def infer_until_converge(self):
        rews_env = [[] for _ in range(self.num_envs)]

        completed_rews = []

        completed_episodes = 0

        ep_lens = []
        t_env = np.zeros(self.num_envs, dtype=int)

        progress_bar = tqdm()

        # Track global Pareto front and no-improvement counter
        global_pareto_set = None  # numpy array of non-dominated solutions
        episodes_since_improvement = 0
        converged = False

        while not converged:
            with torch.no_grad():
                (action, _, _), _ = self.get_action_and_value(self.obs)

            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.env.step(cpu_action)

            for i in range(self.num_envs):
                if next_obs[i] is None:
                    continue

                rews_env[i].append(rew[i])
                t_env[i] += 1
                self.obs[i] = next_obs[i]

                if done[i]:
                    ep_lens.append(t_env[i])
                    completed_rews.append(rews_env[i].copy())
                    rews_env[i] = []  # Clear for next episode

                    progress_bar.update(1)
                    completed_episodes += 1
                    t_env[i] = 0

                    env_i_solutions = self.env.get_env_solutions_and_clear(i)

                    global_pareto_set, improved = self.update_pareto_front(
                        global_pareto_set, env_i_solutions
                    )
                    episodes_since_improvement = (
                        0 if improved else episodes_since_improvement + 1
                    )
                    print(
                        f"Pareto size: {len(global_pareto_set) if global_pareto_set is not None else 0} | Episodes since improvement: {episodes_since_improvement}"
                    )

                if episodes_since_improvement >= self.num_envs:
                    converged = True

        progress_bar.close()

        mean_rew = statistics.mean([sum(r) for r in completed_rews])
        max_rew = max([sum(r) for r in completed_rews])
        min_rew = min([sum(r) for r in completed_rews])
        stdev_rew = (
            statistics.stdev([sum(r) for r in completed_rews])
            if len(completed_rews) > 1
            else 0
        )
        print(
            f"Mean reward: {mean_rew} ({stdev_rew}), Max reward: {max_rew}, Min reward: {min_rew}"
        )
        print("Pareto set: ", global_pareto_set)
        return (
            global_pareto_set.tolist() if global_pareto_set is not None else []
        ), completed_episodes

    def inference_sampling(
        self, num_samples, seed, infer_set_single=False, nn_class=None
    ):

        if self.args.ac_model and self.args.ac_model != "":
            ac_model = self.args.ac_model

        if not self.args.ac_model or self.args.ac_model == "":
            raise Exception("No actor model provided for sampling trajectories.")

        start_time = time.perf_counter()

        self.ac = nn_class(
            self.pallete,
            self.args.hidden_dim,
            len(self.pallete.units),
            rnn_layer=self.rnn_type,
        ).to(self.args.device)
        self.ac.load_state_dict(torch.load(ac_model))
        self.ac.eval()

        inf_times = []
        if infer_set_single:
            print(f"Running inference (using seed {seed})")
            seed_everything(seed)

            solutions_name_rew = self.run_inference(num_samples, infer_set_single=True)
            for solutions, name, rew in solutions_name_rew:
                print(f"Model: {name}, Reward: {rew}, Solutions: {solutions}")
                solutions_path = os.path.join(
                    Path(self.args.ac_model).resolve().parent.parent,
                    "sampled_solutions",
                    self.args.wandb_config["group"],
                    name,
                    f"{os.path.splitext(os.path.basename(self.args.ac_model))[0]}--{str(seed)}",
                )
                self.write_solutions_to_dir(
                    seed, solutions, solutions_path, "nn-solutions.txt", 1, 0, rew
                )

        else:
            print(f"Running inference (seed={seed})")
            seed_everything(seed)

            start_time = time.perf_counter()

            solution_set, sampled_episodes, mean_rew = self.run_inference(num_samples)

            end_time = time.perf_counter()
            inf_times.append(end_time - start_time)
            self.obs = self.env.reset()

            if solution_set:
                if num_samples > 0:
                    solutions_path = os.path.join(
                        Path(self.args.ac_model).resolve().parent.parent,
                        "sampled_solutions",
                        self.args.wandb_config["group"],
                        f"{os.path.splitext(os.path.basename(self.args.ac_model))[0]}--{str(seed)}",
                    )
                    self.write_solutions_to_dir(
                        seed,
                        solution_set,
                        solutions_path,
                        "nn-solutions.txt",
                        sampled_episodes,
                        end_time - start_time,
                        mean_rew,
                    )
                else:
                    solutions_path = os.path.join(
                        Path(self.args.ac_model).resolve().parent.parent,
                        "sampled_solutions",
                        self.args.wandb_config["group"],
                        f"{os.path.splitext(os.path.basename(self.args.ac_model))[0]}--{str(seed)}",
                    )
                    self.write_solutions_to_dir(
                        seed,
                        solution_set,
                        solutions_path,
                        "nn-solutions.txt",
                        sampled_episodes,
                        end_time - start_time,
                        mean_rew,
                    )

        model_name = os.path.splitext(os.path.basename(self.args.ac_model))[0]
        model_dir = os.path.join(os.path.dirname(self.args.ac_model), model_name)
        # Write inference times to file and the mean and total inference time
        if inf_times and len(inf_times) > 1:
            with open(
                os.path.join(model_dir, f"inference_times_seeds_N-{num_samples}.txt"),
                "w",
            ) as f:
                for t in inf_times:
                    f.write(f"{t:.3f}\n")
                f.write(
                    f"Mean inference time: {statistics.mean(inf_times):.3f} seconds (+- {statistics.stdev(inf_times):.3f})\n"
                )
                f.write(f"Total inference time: {sum(inf_times):.3f} seconds\n")

    def write_solutions_to_dir(
        self,
        seed,
        solution_set,
        solutions_path,
        file_path,
        sampled_episodes,
        time_taken,
        mean_rew,
    ):
        os.makedirs(solutions_path, exist_ok=True)
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

    def write_solutions_to_file(
        self, solution_set, file_name, sampled_episodes, time_taken
    ):
        ## Get filename of args.ac_model without extension
        model_name = os.path.splitext(os.path.basename(self.args.ac_model))[0]
        model_dir = os.path.join(os.path.dirname(self.args.ac_model), model_name)
        os.makedirs(os.path.join(model_dir, os.path.dirname(file_name)), exist_ok=True)
        # print(f"Writing solutions to {model_dir}")
        with open(os.path.join(model_dir, file_name), "w") as f:
            f.write(
                f"Inference time: {time_taken:.3f} seconds ({sampled_episodes} episodes)\n"
            )

            for sol in solution_set:
                f.write(str(sol) + "\n")

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
            total_timesteps - the total number of timesteps to train for

        Return:
            None
        """
        print(f"Learning... Running", end="")
        print(
            f"{self.timesteps_per_batch} timesteps per batch with {self.num_envs} parallel environments ({self.n_updates_per_iteration} updates a {self.mini_batch_size} steps)",
            end="",
        )
        t_so_far = 0  # Timesteps simulated so far
        e_so_far = 0  # Episodes simulated so far
        i_so_far = 0  # Iterations ran so far

        if self.args.rl_evaluate:
            print("Evaluating initial policy before training...", flush=True)
            self.ac.eval()
            with torch.no_grad():
                batch_rews = self.eval_policy()
                if self.args.use_wandb:
                    wandb.log(
                        {
                            "avg_evaluation_return": float(
                                np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
                            )
                        },
                        step=0,
                    )
                self.episode_stats.add_eval_mean_return(
                    np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
                )
            self.ac.train()

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
                ) = self.rollout()
            else:
                self.obs = self.env.reset()
                print("Using full trajectory rollouts!", flush=True)
                (
                    batch_obs,
                    batch_acts,
                    batch_log_probs,
                    batch_returns,
                    batch_advantages,
                    batch_lens,
                    finished_eps,
                ) = self.full_episodes_rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            e_so_far += finished_eps
            # Increment the number of iterations
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["e_so_far"] = e_so_far
            self.logger["i_so_far"] = i_so_far

            indices = np.arange(len(batch_obs))
            # Create balanced mini-batches to ensure equal environment representation
            for _ in range(self.n_updates_per_iteration):
                if self.balanced_sampler.has_tracking_info():
                    mini_batch_indices = self.balanced_sampler.create_minibatch_indices(
                        len(batch_obs), self.mini_batch_size
                    )
                else:
                    np.random.shuffle(indices)
                    mini_batch_indices = [
                        indices[i : i + self.mini_batch_size]
                        for i in range(0, len(batch_obs), self.mini_batch_size)
                    ]

                for mb_indices in mini_batch_indices:
                    batch_obs_mb = [batch_obs[i] for i in mb_indices]
                    batch_acts_mb = [batch_acts[i] for i in mb_indices]
                    batch_log_probs_mb = batch_log_probs[mb_indices]
                    batch_returns_mb = batch_returns[mb_indices]
                    batch_advantages_mb = batch_advantages[mb_indices]

                    if batch_advantages_mb.numel() <= 1:
                        f_logger.warning(
                            f"[NaN Debug] batch_advantages_mb has {batch_advantages_mb.numel()} elements - std() will have <= 0 degrees of freedom!"
                        )
                        f_logger.warning(
                            f"  Mini-batch size: {len(mb_indices)}, batch_advantages shape: {batch_advantages_mb.shape}"
                        )
                        f_logger.warning(
                            f"  This can lead to NaN in advantage normalization!"
                        )

                    A_k = (batch_advantages_mb - batch_advantages_mb.mean()) / (
                        batch_advantages_mb.std() + 1e-8
                    )

                    if torch.isnan(A_k).any():
                        f_logger.error(f"[NaN Debug] NaN in A_k after normalization!")
                        f_logger.error(
                            f"  batch_advantages_mb: min={batch_advantages_mb.min():.4f}, max={batch_advantages_mb.max():.4f}, mean={batch_advantages_mb.mean():.4f}, std={batch_advantages_mb.std():.4f}"
                        )
                        f_logger.error(
                            f"  batch_advantages_mb numel: {batch_advantages_mb.numel()}"
                        )

                    mb = Batch.from_data_list(batch_obs_mb)

                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, curr_dist_entropies = self.evaluate(
                        mb, batch_acts_mb
                    )

                    # DEBUG: Check for MSE loss shape mismatch (potential NaN/incorrect results source)
                    if V.shape != batch_returns_mb.shape:
                        f_logger.warning(
                            f"[NaN Debug] Shape mismatch between V and batch_returns_mb!"
                        )
                        f_logger.warning(
                            f"  V shape: {V.shape}, batch_returns_mb shape: {batch_returns_mb.shape}"
                        )
                        f_logger.warning(
                            f"  This will cause broadcasting in MSE loss and likely incorrect results!"
                        )

                    if self.debug_nan:
                        if torch.isnan(V).any():
                            f_logger.error(
                                f"NaN in V after evaluate: {torch.isnan(V).sum()}/{V.numel()} values"
                            )
                        if torch.isnan(curr_log_probs).any():
                            f_logger.error(
                                f"NaN in curr_log_probs: {torch.isnan(curr_log_probs).sum()}/{curr_log_probs.numel()} values"
                            )
                            # Log which log probs are NaN
                            nan_indices = torch.where(torch.isnan(curr_log_probs))[0]
                            f_logger.error(
                                f"  NaN indices: {nan_indices.tolist()[:10]}..."
                            )
                        if torch.isnan(curr_dist_entropies).any():
                            f_logger.error(
                                f"NaN in entropy: {torch.isnan(curr_dist_entropies).sum()}/{curr_dist_entropies.numel()} values"
                            )

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    ratios = torch.exp(curr_log_probs - batch_log_probs_mb)

                    # Calculate surrogate losses.
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (
                        -torch.min(surr1, surr2)
                    ).mean() - self.entropy_coef * curr_dist_entropies.mean()

                    # DEBUG: Log detailed info before MSE loss if shape mismatch detected
                    if V.shape != batch_returns_mb.shape:
                        f_logger.error(f"[NaN Debug] MSE Loss shape mismatch detected!")
                        f_logger.error(
                            f"  V: shape={V.shape}, dtype={V.dtype}, device={V.device}"
                        )
                        f_logger.error(
                            f"  batch_returns_mb: shape={batch_returns_mb.shape}, dtype={batch_returns_mb.dtype}"
                        )
                        f_logger.error(f"  V values: {V}")
                        f_logger.error(f"  batch_returns_mb values: {batch_returns_mb}")
                        f_logger.error(f"  Mini-batch size: {len(mb_indices)}")

                    critic_loss = 0.5 * nn.MSELoss()(V, batch_returns_mb)

                    if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                        f_logger.error(
                            f"[NaN Debug] critic_loss is NaN/Inf: {critic_loss}"
                        )
                        f_logger.error(
                            f"  V shape: {V.shape}, batch_returns_mb shape: {batch_returns_mb.shape}"
                        )
                        f_logger.error(
                            f"  V has NaN: {torch.isnan(V).any()}, batch_returns_mb has NaN: {torch.isnan(batch_returns_mb).any()}"
                        )

                    self.logger["actor_losses"].append(actor_loss.detach())
                    self.logger["critic_losses"].append(critic_loss.detach())
                    self.logger["entropy"].append(curr_dist_entropies.detach())

                    if self.args.no_nn_updates:
                        continue

                    if self.debug_nan:
                        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                            f_logger.error(
                                f"NaN/Inf detected in actor_loss: {actor_loss.item()}"
                            )
                            f_logger.error(
                                f"  curr_log_probs stats: min={curr_log_probs.min():.4f}, max={curr_log_probs.max():.4f}, has_nan={torch.isnan(curr_log_probs).any()}"
                            )
                            f_logger.error(
                                f"  batch_log_probs_mb stats: min={batch_log_probs_mb.min():.4f}, max={batch_log_probs_mb.max():.4f}"
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
                    if self.encoder == "shared":
                        cb_loss = actor_loss + critic_loss

                        self.ac_optim.zero_grad()
                        cb_loss.backward()

                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.ac.parameters(), max_norm=0.5
                        )
                        self.ac_optim.step()

                    elif self.encoder == "separate":
                        self.a_optim.zero_grad()
                        actor_loss.backward(retain_graph=True)

                        if self.debug_nan:
                            self.gradient_monitor.log_gradients()

                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.ac.actor.parameters(), max_norm=0.5
                        )
                        self.a_optim.step()

                        self.c_optim.zero_grad()
                        critic_loss.backward()

                        if self.debug_nan:
                            self.gradient_monitor.log_gradients()

                        torch.nn.utils.clip_grad_norm_(
                            self.ac.critic.parameters(), max_norm=0.5
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

           # if self.encoder == "shared":
           #     self.lr_scheduler.step()

            if self.args.rl_evaluate:
                if i_so_far % eval_interval == 0:
                    self.ac.eval()
                    with torch.no_grad():
                        batch_rews = self.eval_policy()
                        self.episode_stats.add_eval_mean_return(
                            np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
                        )
                    self.ac.train()

            mean_batch_rew = (
                self.episode_stats.get_last_rollout_mean_return()
                if not self.args.rl_sample_partial_trajectories
                else self.episode_stats.get_recent_stats(30)["mean_return"]
            )
            should_stop = self.early_stopping.step(mean_batch_rew, i_so_far, t_so_far)

            # If new best reward, save with the current naming scheme
            if self.early_stopping.best_reward == mean_batch_rew:
                torch.save(
                    self.ac.state_dict(),
                    os.path.join(
                        self.args.model_path,
                        f"ac_{i_so_far}-{t_so_far}-{mean_batch_rew:.3f}.pth",
                    ),
                )

            if should_stop:
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

        if self.args.rl_evaluate:
            self.ac.eval()
            with torch.no_grad():
                batch_rews = self.eval_policy()
                # Print final evaluation results
                mean_batch_rew = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
                print(f"Final mean return: {mean_batch_rew:.3f}")
                self.episode_stats.add_eval_mean_return(mean_batch_rew)
            self.ac.train()

        # Cleanup after run
        self.env.post_training()
        self.env.close()

    def eval_policy(self):

        self.eval_obs = self.eval_env.reset()
        num_eval_envs = len(self.eval_obs)
        finished = np.zeros(num_eval_envs, dtype=bool)

        rews_env = [[] for _ in range(num_eval_envs)]
        completed_rews = [None] * num_eval_envs

        while not np.all(finished):
            (action, _, _), _ = self.get_action_and_value(self.eval_obs)
            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.eval_env.step(cpu_action)

            for i in range(num_eval_envs):
                if next_obs[i] is None:
                    continue

                if not finished[i]:
                    rews_env[i].append(rew[i])

                    if done[i]:
                        completed_rews[i] = rews_env[i].copy()
                        finished[i] = True

                self.eval_obs[i] = next_obs[i]

        self.eval_env.post_evaluation()

        return [traj for traj in completed_rews if traj is not None]

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
        """
        # Per-env storage for current episode
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

        ep_lens = []
        log_ep_rews = []

        t = 0
        t_env = np.zeros(self.num_envs, dtype=int)
        progress_bar = nqdm(total=self.timesteps_per_batch, position=0, leave=True)

        while len(completed_obs) < self.timesteps_per_batch or any(
            len(obs_env[i]) > 0 for i in range(self.num_envs)
        ):

            (action, log_prob, node_mask), value = self.get_action_and_value(self.obs)

            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.env.step(cpu_action)

            for i in range(self.num_envs):
                if next_obs[i] is None:
                    continue

                if -1 in action[i][1]:
                    print(
                        f"Warning: Action for env {i} contains -1 in nodes: {action[i][1]}"
                    )
                # Update discounted returns and normalize reward
                if self.normalize_rewards:
                    self.returns[i] = self.returns[i] * self.gamma + rew[i]
                    normalized_rew = self.reward_normalizer.normalize(
                        np.array([rew[i]]), update_stats=False
                    )[0]
                    # Update statistics with the discounted return
                    self.reward_normalizer.update(np.array([self.returns[i]]))
                else:
                    normalized_rew = rew[i]

                # Store current step in per-env buffers
                obs_env[i].append(
                    self.add_act_forcing_data_to_obs(
                        self.obs[i], action[i], node_mask[i]
                    )
                )
                acts_env[i].append(action[i])
                log_probs_env[i].append(log_prob[i])
                rews_env[i].append(
                    torch.tensor(
                        normalized_rew, dtype=torch.float32, device=self.args.device
                    )
                )
                dones_env[i].append(
                    torch.tensor(done[i], dtype=torch.float32, device=self.args.device)
                )
                vals_env[i].append(
                    torch.tensor(value[i], dtype=torch.float32, device=self.args.device)
                )

                t_env[i] += 1
                t += 1
                progress_bar.update(1)

                self.obs[i] = next_obs[i]

                # If episode is done, move data to completed storage
                if done[i]:
                    current_ep_len = t_env[i]
                    current_ep_rews = [
                        r.item() if isinstance(r, torch.Tensor) else r
                        for r in rews_env[i]
                    ]
                    episode_return = sum(current_ep_rews)

                    ep_lens.append(current_ep_len)
                    log_ep_rews.append(current_ep_rews)

                    self.episode_stats.add_episode(episode_return, current_ep_len)

                    completed_obs.extend(obs_env[i])
                    completed_acts.extend(acts_env[i])
                    completed_log_probs.extend(log_probs_env[i])
                    completed_rews.extend(rews_env[i])
                    completed_dones.extend(dones_env[i])
                    completed_vals.extend(vals_env[i])

                    # Clear per-env buffers for this environment
                    obs_env[i] = []
                    acts_env[i] = []
                    log_probs_env[i] = []
                    rews_env[i] = []
                    dones_env[i] = []
                    vals_env[i] = []

                    if self.normalize_rewards:
                        self.returns[i] = 0.0

                    t_env[i] = 0

            # Break if we have enough completed episode data
            if len(completed_obs) >= self.timesteps_per_batch:
                break

        progress_bar.close()
        finished_episodes = len(ep_lens)
        self.env.post_rollout()

        if len(completed_obs) > 0:

            # Organize completed data by episodes for GAE computation
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

            # No need for next values since all episodes are complete (bootstrap value = 0)
            next_values_per_ep = [
                torch.tensor(0.0, device=self.args.device) for _ in range(len(ep_lens))
            ]

            batch_returns, batch_advantages = self.compute_gae_per_env(
                completed_rews_per_ep,
                completed_dones_per_ep,
                completed_vals_per_ep,
                next_values_per_ep,
            )

            # Truncate to desired batch size if we have more data than needed
            if len(completed_obs) > self.timesteps_per_batch:
                completed_obs = completed_obs[: self.timesteps_per_batch]
                completed_acts = completed_acts[: self.timesteps_per_batch]
                completed_log_probs = completed_log_probs[: self.timesteps_per_batch]
                batch_returns = batch_returns[: self.timesteps_per_batch]
                batch_advantages = batch_advantages[: self.timesteps_per_batch]

            batch_log_probs = torch.tensor(
                completed_log_probs, dtype=torch.float32, device=self.args.device
            )
        else:
            # Handle case where no episodes completed
            completed_obs = []
            completed_acts = []
            batch_log_probs = torch.tensor(
                [], dtype=torch.float32, device=self.args.device
            )
            batch_returns = torch.tensor(
                [], dtype=torch.float32, device=self.args.device
            )
            batch_advantages = torch.tensor(
                [], dtype=torch.float32, device=self.args.device
            )

        self.logger["batch_lens"] = ep_lens

        self.episode_stats.add_rollout_mean_return(
            np.mean([np.sum(ep_rews) for ep_rews in log_ep_rews])
        )

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
            batch_lens:       [N] list of episode lengths per env (for logging)
        """
        obs_env = [[] for _ in range(self.num_envs)]
        acts_env = [[] for _ in range(self.num_envs)]
        log_probs_env = [[] for _ in range(self.num_envs)]
        rews_env = [[] for _ in range(self.num_envs)]
        dones_env = [[] for _ in range(self.num_envs)]
        vals_env = [[] for _ in range(self.num_envs)]

        done = np.zeros(self.num_envs, dtype=np.float32)
        # Track completed episodes for logging)
        completed_ep_returns = []
        completed_ep_lens = []
        # Track segment lengths within this rollout for training data organization
        segment_lens = []
        # Track where the last episode ended in rews_env buffer (for slicing current episode rewards)
        last_env_done_idx = np.zeros(self.num_envs, dtype=int)

        t = 0
        # Steps collected in current rollout per environment (resets on done within this rollout)
        steps_in_current_segment = np.zeros(self.num_envs, dtype=int)
        progress_bar = nqdm(total=self.timesteps_per_batch, position=0, leave=True)

        while t < self.timesteps_per_batch:

            (action, log_prob, node_dim_masks), value = self.get_action_and_value(
                self.obs
            )

            cpu_action = [move_to_cpu(a) for a in action]
            next_obs, rew, done = self.env.step(cpu_action)

            for i in range(self.num_envs):
                if next_obs[i] is None:
                    continue

                # Update discounted returns and normalize reward (OpenAI Baselines style)
                if self.normalize_rewards:
                    # Update discounted return for this environment
                    self.returns[i] = self.returns[i] * self.gamma + rew[i]
                    # Normalize the reward using current statistics
                    normalized_rew = self.reward_normalizer.normalize(
                        np.array([rew[i]]), update_stats=False
                    )[0]
                    # Update statistics with the discounted return
                    self.reward_normalizer.update(np.array([self.returns[i]]))
                else:
                    normalized_rew = rew[i]

                obs_env[i].append(
                    self.add_act_forcing_data_to_obs(
                        self.obs[i], action[i], node_dim_masks[i]
                    )
                )
                acts_env[i].append(action[i])
                log_probs_env[i].append(log_prob[i])

                rews_env[i].append(
                    torch.tensor(
                        normalized_rew, dtype=torch.float32, device=self.args.device
                    )
                )
                dones_env[i].append(
                    torch.tensor(done[i], dtype=torch.float32, device=self.args.device)
                )
                vals_env[i].append(
                    torch.tensor(value[i], dtype=torch.float32, device=self.args.device)
                )

                steps_in_current_segment[i] += 1

                # Accumulate episode rewards/lengths across rollout boundaries (SB3 style)
                self._current_episode_rewards[i] += rew[i]  # Use raw reward for logging
                self._current_episode_lengths[i] += 1

                next_t = t + 1
                if next_t <= self.timesteps_per_batch:
                    progress_bar.update(1)
                t = next_t

                self.obs[i] = next_obs[i]

                if done[i]:
                    # Episode completed - record full episode stats (accumulated across rollouts)
                    episode_return = self._current_episode_rewards[i]
                    episode_length = self._current_episode_lengths[i]

                    # Log completed episode (SB3 style - only completed episodes are reported)
                    completed_ep_returns.append(episode_return)
                    completed_ep_lens.append(episode_length)

                    # Add to rolling buffer for statistics
                    self.episode_stats.add_episode(episode_return, episode_length)

                    # Track segment length for training data organization
                    segment_lens.append(steps_in_current_segment[i])

                    # Reset cumulative trackers for next episode (SB3 style)
                    self._current_episode_rewards[i] = 0.0
                    self._current_episode_lengths[i] = 0

                    last_env_done_idx[i] = len(rews_env[i])
                    steps_in_current_segment[i] = 0

                    # Reset discounted return for reward normalization
                    if self.normalize_rewards:
                        self.returns[i] = 0.0

        progress_bar.close()

        # Add incomplete segment lengths for training data organization
        for i in range(self.num_envs):
            if steps_in_current_segment[i] > 0:
                segment_lens.append(steps_in_current_segment[i])

        finished_episodes = len(completed_ep_returns)
        self.env.post_rollout()

        # Bootstrap value for each environment
        with torch.no_grad():
            to_bootstrap = [
                (i, self.obs[i]) for i in range(self.num_envs) if not done[i]
            ]

            # Step 2: Run batched value prediction
            if to_bootstrap:
                indices, obs_batch = zip(*to_bootstrap)
                values = self.get_value(Batch.from_data_list(obs_batch))
            else:
                indices, values = [], []

            # Step 3: Populate next_values with either value or 0
            next_values_per_env = []
            value_map = {i: v for i, v in zip(indices, values)}

            for i in range(self.num_envs):
                next_values_per_env.append(
                    value_map.get(i, torch.tensor(0.0, device=self.args.device))
                )

        # Compute returns and advantages via GAE
        batch_returns, batch_advantages = self.compute_gae_per_env(
            rews_env, dones_env, vals_env, next_values_per_env
        )

        batch_obs = [obs for sublist in obs_env for obs in sublist]
        batch_acts = [act for sublist in acts_env for act in sublist]
        batch_log_probs = torch.tensor(
            [lp for sublist in log_probs_env for lp in sublist],
            dtype=torch.float32,
            device=self.args.device,
        )

        # Truncate to exactly timesteps_per_batch to ensure mini-batches divide evenly
        # This handles cases where we collect slightly more samples due to env stepping
        actual_batch_size = len(batch_obs)
        if actual_batch_size > self.timesteps_per_batch:
            batch_obs = batch_obs[: self.timesteps_per_batch]
            batch_acts = batch_acts[: self.timesteps_per_batch]
            batch_log_probs = batch_log_probs[: self.timesteps_per_batch]
            batch_returns = batch_returns[: self.timesteps_per_batch]
            batch_advantages = batch_advantages[: self.timesteps_per_batch]

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
        )

    def compute_gae_per_env(
        self, rewards_per_env, dones_per_env, values_per_env, next_values_per_env
    ):
        """
        Vectorized GAE: handles envs that contribute uneven number of steps.

        Parameters:
            rewards_per_env: list of lists [ [r1, r2, ...], [r1, ...], ... ]
            dones_per_env: list of lists [ [d1, d2, ...], ... ]
            values_per_env: list of lists [ [v1, v2, ...], ... ]
            next_values_per_env: list of scalars (bootstrap value for each env)

        Returns:
            returns_flat: Tensor [T]
            advantages_flat: Tensor [T]
        """
        all_returns = []
        all_advantages = []

        for rews, dones, vals, next_val in zip(
            rewards_per_env, dones_per_env, values_per_env, next_values_per_env
        ):
            T = len(rews)
            returns = torch.zeros(T, device=self.args.device)
            advantages = torch.zeros(T, device=self.args.device)
            gae = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = next_val
                else:
                    next_value = vals[t + 1]

                next_done = dones[t]
                delta = rews[t] + self.gamma * next_value * (1 - next_done) - vals[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + vals[t]

            all_returns.append(returns)
            all_advantages.append(advantages)

        # Flatten across envs
        returns_flat = torch.cat(all_returns, dim=0)
        advantages_flat = torch.cat(all_advantages, dim=0)

        return returns_flat, advantages_flat

    def get_action_and_value(self, obs):
        batch = Batch.from_data_list(obs)

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
        ), V = self.ac(
            batch.x.to(self.args.device),
            batch.edge_index.to(self.args.device),
            torch.squeeze(batch.edge_attr, dim=1).to(self.args.device),
            batch.batch.to(self.args.device),
            batch.nodes.to(self.args.device),
            batch.action_mask.to(self.args.device),
            batch.action_node_mask.to(self.args.device),
        )

        total_log_prob = 0

        if not (len(action_dist.param_shape) == 2 and action_dist.param_shape[1] == 1):
            action_log_prob = action_dist.log_prob(action)
            total_log_prob += action_log_prob.detach()

        if nodes_dist is not None:
            nodes = nodes.detach()
            # Mask invalid indices (-1), replace with safe index 0 for log_prob, and zero-out their contribution
            valid_mask = (nodes >= 0).bool()
            nodes_safe = torch.clamp(nodes, min=0)
            node_log_probs = nodes_dist.log_prob(nodes_safe)
            node_log_probs = node_log_probs.detach() * seq_len_mask * valid_mask
            total_log_prob += node_log_probs.sum(dim=1)

        if values_dist_tt is not None:
            values_real = values[0].detach()  # Shape: (batch_size, value_dim)
            values_raw = values[1].detach()  # Values used to compute log_probs

            values_raw = values_raw.clamp(min=-1 + eps_log_clip, max=1 - eps_log_clip)
            values_log_probs = values_dist_tt.log_prob(values_raw)
            total_log_prob += values_log_probs.detach().sum(dim=1)

        actions = action.detach()  # Shape: (batch_size,)
        batch_size = actions.shape[0]
        batch_action = []

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

        # [1], [1,2], [1,1]
        batch_node_masks = list(node_mask.unbind(dim=0))
        return (batch_action, total_log_prob.tolist(), batch_node_masks), V.detach()

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
        )

        total_log_prob = 0

        if not (len(action_dist.param_shape) == 2 and action_dist.param_shape[1] == 1):
            action_log_prob = action_dist.log_prob(action)
            total_log_prob += action_log_prob.detach()

        if nodes_dist is not None:
            nodes = nodes.detach()
            valid_mask = (nodes >= 0).bool()
            nodes_safe = torch.clamp(nodes, min=0)
            node_log_probs = nodes_dist.log_prob(nodes_safe)
            node_log_probs = node_log_probs.detach() * node_seq_mask * valid_mask
            total_log_prob += node_log_probs.sum(dim=1)

        if values_dist_tt is not None:
            values_real = values[0].detach()  # Shape: (batch_size, value_dim)
            values_raw = values[1].detach()  # Values used to compute log_probs

            values_raw = values_raw.clamp(min=-1 + eps_log_clip, max=1 - eps_log_clip)
            values_log_probs = values_dist_tt.log_prob(values_raw)
            total_log_prob += values_log_probs.detach().sum(dim=1)

        actions = action.detach()  # Shape: (batch_size,)
        batch_size = actions.shape[0]
        batch_action = []

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

    def evaluate(self, batch, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs

        tf_action = TFAction(
            batch.force_act.to(self.args.device),
            batch.force_nodes.to(self.args.device),
            batch.force_nodes_mask.to(self.args.device),
            None,
        )

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
            batch.x.to(self.args.device),
            batch.edge_index.to(self.args.device),
            torch.squeeze(batch.edge_attr, dim=1).to(self.args.device),
            batch.batch.to(self.args.device),
            batch.nodes.to(self.args.device),
            batch.action_mask.to(self.args.device),
            batch.action_node_mask.to(self.args.device),
            tf_action,
        )

        if batch_acts[0][1] is None and batch_acts[0][2] is None:
            batch_acts_act = [tup[0] for tup in batch_acts]
        elif batch_acts[0][1] is None:
            batch_acts_act, batch_acts_vals = zip(*[(t[0], t[2]) for t in batch_acts])
        elif batch_acts[0][2] is None:
            batch_acts_act, batch_acts_nodes = zip(*[(t[0], t[1]) for t in batch_acts])
        else:
            batch_acts_act, batch_acts_nodes, batch_acts_vals = zip(*batch_acts)

        dist_entropy = []
        log_probs = []

        if not (len(action_dist.param_shape) == 2 and action_dist.param_shape[1] == 1):
            batch_acts_act = torch.tensor(
                batch_acts_act, dtype=torch.int, device=self.args.device
            )
            action_log_prob = action_dist.log_prob(batch_acts_act)

            dist_entropy.append(action_dist.entropy())
            log_probs.append(action_log_prob)

        if "batch_acts_nodes" in locals():
            batch_acts_nodes = self.pad_and_stack(batch_acts_nodes).squeeze(1)
            # batch_acts_nodes = torch.stack(batch_acts_nodes, dim=0).squeeze(1)
            nodes_log_prob = nodes_dist.log_prob(batch_acts_nodes) * seq_len_mask
            nodes_log_prob = nodes_log_prob.sum(dim=1)
            nodes_dist_entropies = nodes_dist.entropy() * seq_len_mask

            log_probs.append(nodes_log_prob)
            dist_entropy.append(nodes_dist_entropies.sum(dim=1))

        if "batch_acts_vals" in locals():
            batch_acts_vals_raw = [t[1] for t in batch_acts_vals]
            batch_acts_vals_raw = torch.stack(batch_acts_vals_raw, dim=0).squeeze(1)

            batch_acts_vals_raw = batch_acts_vals_raw.clamp(
                min=-1 + eps_log_clip, max=1 - eps_log_clip
            )
            values_log_prob = values_dist_tt.log_prob(batch_acts_vals_raw)
            values_log_prob = values_log_prob.sum(dim=1)
            values_dist_entropies = values_dist.entropy()

            log_probs.append(values_log_prob)
            dist_entropy.append(values_dist_entropies.sum(dim=1))

            if torch.isnan(values_log_prob).any():
                print("mean values_log_prob", values_log_prob.sum())

        log_probs = (
            torch.stack(log_probs, dim=1) if len(log_probs) > 1 else log_probs[0]
        )
        dist_entropy = (
            torch.stack(dist_entropy, dim=1)
            if len(dist_entropy) > 1
            else dist_entropy[0]
        )

        return (
            V.squeeze(),
            torch.sum(log_probs, dim=1) if log_probs.ndim > 1 else log_probs,
            torch.sum(dist_entropy, dim=1) if dist_entropy.ndim > 1 else dist_entropy,
        )

    # return V, log_probs, dist_entropy

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
        self.timesteps_per_batch = 4800  # Number of timesteps per rollout batch
        self.mini_batch_size = 100  # Number of timesteps per mini batch
        self.n_updates_per_iteration = 4  # Updates of actor/critic per iteration
        self.lr = 3e-4  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.clip = 0.2  # Recommended 0.2
        self.rnn_type = "gru"
        self.gae_lambda = 0.95  # Lambda for GAE, recommended
        self.entropy_coef = 0.01  # Coefficient for entropy loss

        # Reward normalization settings
        self.normalize_rewards = False
        self.es_patience = 10  # Early stopping patience  (iterations)
        self.es_min_delta = 0.01  # Minimum change in reward to qualify as
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

        if self.args.rl_sample_partial_trajectories:
            avg_ep_rews = self.episode_stats.get_recent_stats(10)["mean_return"]
        else:
            avg_ep_rews = self.episode_stats.get_last_rollout_mean_return()

        avg_ep_lens = np.mean(self.logger["batch_lens"])
        # Get rolling buffer statistics
        buffer_stats = self.episode_stats.get_stats()
        recent_stats = self.episode_stats.get_recent_stats(50)

        avg_actor_loss = np.mean(
            [losses.cpu().float().mean() for losses in self.logger["actor_losses"]]
        )
        avg_critic_loss = np.mean(
            [losses.cpu().float().mean() for losses in self.logger["critic_losses"]]
        )
        avg_entropy = np.mean([e.cpu().float().mean() for e in self.logger["entropy"]])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 3))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))
        avg_entropy = str(round(avg_entropy, 3))
        value_head_log_std = self.logger["value_head_log_std"]

        # Print logging statements
        f_logger.info(
            f"-------------------- Iteration #{i_so_far} --------------------"
        )
        f_logger.info(f"Batch Episodes: {len(self.logger['batch_lens'])}")
        f_logger.info(f"Average Episodic Length: {avg_ep_lens}")
        f_logger.info(f"Average Episodic Return: {avg_ep_rews}")
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
        f_logger.info(f"Average Loss: {avg_actor_loss}")
        f_logger.info(f"Timesteps So Far: {t_so_far}")
        f_logger.info(f"Episodes So Far: {e_so_far}")
        f_logger.info(f"Iteration took: {delta_t} secs")
        f_logger.info(f"------------------------------------------------------")

        # Enhanced wandb logging with buffer statistics
        log_val = {
            "avg_episode_length": float(avg_ep_lens),
            "avg_episode_return": float(avg_ep_rews),
            "avg_actor_loss": float(avg_actor_loss),
            "avg_entropy": float(avg_entropy),
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
        }

        if (
            self.args.use_wandb
            and i_so_far % eval_interval == 0
            and self.args.rl_evaluate
        ):
            wandb.log(
                {
                    "avg_evaluation_return": float(
                        self.episode_stats.get_last_eval_mean_return()
                    )
                },
                step=i_so_far - 1,
            )

        if self.args.use_wandb:
            wandb.log(log_val)

        self.logger["avg_lens"].append(float(avg_ep_lens))
        self.logger["avg_rews"].append(float(avg_ep_rews))
        self.logger["avg_actor_losses"].append(float(avg_actor_loss))
        f_logger.info("Mean episodic rewards:")
        f_logger.info(self.logger["avg_rews"])

        if self.args.rl_evaluate:
            f_logger.info("Evaluation: Mean episodic rewards:")
            # Eval reward with values rounded to 3 decimal places
            f_logger.info([round(x, 3) for x in self.episode_stats.eval_mean_returns])

        self.logger["batch_lens"] = []

        self.logger["actor_losses"] = []
        self.logger["entropy"] = []
        self.logger["critic_losses"] = []

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

        force_act_data = {"force_act": act[0].unsqueeze(0)}

        if act[1] is not None:
            num_nodes = obs["nodes"].item()
            seq_len = act[1].shape[1]

            # Create one-hot encoding more efficiently
            oneh_nodes = torch.zeros(
                num_nodes,
                self.pallete.max_len,
                dtype=torch.float32,
                device=act[1].device,
            )
            # a = act[1][0].to('cpu')
            # b = act[0].to('cpu')
            oneh_nodes[act[1][0], torch.arange(seq_len, device=act[1].device)] = 1.0

            force_act_data["force_nodes"] = oneh_nodes

            # Process node mask more efficiently
            nodes_mask = node_masks.T[:num_nodes]

            if nodes_mask.shape[1] < self.pallete.max_len:
                # Pad to target length in one operation
                pad_width = self.pallete.max_len - nodes_mask.shape[1]
                nodes_mask = F.pad(nodes_mask, (0, pad_width), value=0.0)

            force_act_data["force_nodes_mask"] = nodes_mask

        obs.update(force_act_data)
        return obs

    def update_pareto_front(self, current_front, new_solutions):
        """Update a maintained global Pareto set with newly found solutions.
        Args:
            current_front: np.ndarray or None, current non-dominated set (minimization objectives)
            new_solutions: list or array of objective vectors (minimization)
        Returns:
            (updated_front, improved): tuple of the updated Pareto set and a boolean indicating improvement.
        """
        # No new solutions provided
        if new_solutions is None or len(new_solutions) == 0:
            return current_front, False

        env_arr = np.array(new_solutions, dtype=np.float64)
        if env_arr.ndim == 1:
            env_arr = env_arr.reshape(1, -1)

        # If there is no current front, compute Pareto of new entries only
        if current_front is None or len(current_front) == 0:
            new_front = env_arr[paretoset(env_arr)]
            improved = len(new_front) > 0
            return new_front, improved

        # Merge and filter non-dominated points
        combined = np.vstack([current_front, env_arr])
        mask = paretoset(combined)
        candidate_front = combined[mask]

        # Compare sets robustly (rounded tuples)
        old_set = set(map(tuple, np.round(current_front, 8)))
        new_set = set(map(tuple, np.round(candidate_front, 8)))
        improved = new_set != old_set

        return (candidate_front if improved else current_front), improved
