"""PPO training: the training loop, the update, convergence and checkpoints."""

import json
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch

from drl.env.batched_env import BatchedRemoteEnv
from drl.env.time_feature import TIME_FEAT_DIM
from drl.main.hyperparams import Hyperparams
from drl.main.inference import run_validation, sample_batches
from drl.main.policy import Policy, model_config, read_model_config, write_model_config
from drl.main.reward_norm import RewardNormalizer
from drl.main.rollout import RolloutCollector, resolve_rollout_length, stack_actions
from drl.main.train_logging import TrainLogger
from drl.neural_model.early_stopping import EarlyStopping
from drl.utils.episode_stats import EpisodeStats
from drl.utils.pallete import PalleteEncoder

EVAL_VARIANTS = {
    "stochastic": ("stochastic",),
    "greedy": ("greedy",),
    "both": ("stochastic", "greedy"),
}


def explained_variance(returns, values):
    """1 - Var(returns - values) / Var(returns). NaN when the variance is undefined."""
    if returns.numel() < 2:
        return float("nan")
    variance = returns.var()
    if variance.item() == 0:
        return float("nan")
    residual_variance = (returns - values).var()
    return (1.0 - residual_variance / (variance + 1e-8)).item()


class PPO:
    """Trains a policy on the models the server on ``args.port`` serves."""

    def __init__(self, args, pallete, run_id=None):
        self.hp = Hyperparams.from_dict(args.hyperparams)
        print("Using normalize_rewards:", self.hp.normalize_rewards)
        self.args = args
        self.pallete = pallete
        self.pallete.masking = self.hp.masking
        self.run_id = run_id

        with open(os.path.join(args.model_path, "pallete.json"), "w") as f:
            json.dump(pallete, f, cls=PalleteEncoder)

        self._connect_envs()
        self.max_horizon = self.env.max_horizon
        self.n_steps, self.timesteps_per_batch = resolve_rollout_length(
            self.hp, self.num_envs, self.max_horizon
        )
        print(
            f"Resolved n_steps={self.n_steps} "
            f"(batch={self.timesteps_per_batch}, max horizon={self.max_horizon})"
        )

        self.episode_stats = EpisodeStats(
            buffer_size=50,
            num_envs=self.num_envs,
            instance_ids=self.env.instance_ids,
            instance_names=self.env.instance_names,
            es_warmup_episodes=self.hp.es_warmup_episodes,
        )
        self.reward_norm = RewardNormalizer(
            self.num_envs,
            len(self.env.instance_names),
            self.hp.gamma,
            self.hp.reward_clip,
            self.hp.rew_norm_freeze_after,
            self.hp.normalize_rewards,
        )

        self._create_policy()
        self._create_optimizers()
        self.collector = RolloutCollector(
            self.env,
            self.policy,
            pallete,
            self.episode_stats,
            self.reward_norm,
            self.hp.gamma,
            self.hp.gae_lambda,
            self.n_steps,
            self.timesteps_per_batch,
            args.device,
        )
        self.early_stopping = EarlyStopping(
            optimizer=self.ac_optim if self.hp.encoder == "shared" else {"actor": self.a_optim, "critic": self.c_optim},
            model=self.ac,
            path=os.path.join(args.model_path, "best_model.pth"),
            patience=self.hp.es_patience,
            min_delta=self.hp.es_min_delta,
            min_timesteps=self.hp.es_min_timesteps,
            min_delta_rel=self.hp.es_min_delta_rel,
        )
        self.train_log = TrainLogger(
            args, self.env, self.num_envs, self.episode_stats, self.eval_variants if self.validating else ()
        )

    def _connect_envs(self):
        """The training env and, if the server has a validation dataset, the validation env."""
        self._apply_saved_time_feature_config()
        self._env_kwargs = {
            "groups": self.args.env_groups,
            "time_feature": self.hp.time_feature,
            "time_scale": self.hp.time_scale,
        }
        # The server is the source of what it was started with.
        self.env = BatchedRemoteEnv.connect(
            self.args.port, self.pallete, str(self.run_id), **self._env_kwargs
        )
        self.num_envs = self.env.num_envs
        if self.hp.es_tick_episodes is None:
            self.hp.es_tick_episodes = self.num_envs

        self.eval_ports = self.args.run_info.get("eval_ports", {})
        self.validating = "val" in self.eval_ports
        self.final_datasets = {
            "none": (), "val": ("val",), "test": ("test",), "both": ("val", "test"),
        }[self.args.final_eval]
        missing = set(self.final_datasets) - set(self.eval_ports)
        if missing:
            raise ValueError(f"The server has no dataset for {sorted(missing)}")
        self._final_envs = {}
        self.final_runtimes = {}
        self.time_validation = 0.0
        if self.validating:
            if self.args.eval_interval < 1:
                raise ValueError("--eval_interval must be at least 1")
            self.eval_interval = self.args.eval_interval
            self.eval_variants = EVAL_VARIANTS[self.args.eval_mode]
            self.eval_env = self._connect_eval_env("val")
        # With --select_on val, each validation phase ticks early stopping and the best policy.
        self.select_on_val = self.args.select_on == "val"
        self._val_should_stop = False
        self._val_ticked = False
        if self.select_on_val:
            if not self.validating:
                raise ValueError("--select_on val needs a validation dataset (--val-path)")
            if "stochastic" not in self.eval_variants:
                raise ValueError("--select_on val ticks on the stochastic validation return; "
                                 "use --eval_mode stochastic or both")

    def _connect_eval_env(self, dataset):
        port = self.eval_ports[dataset]
        print(f"Setting up evaluation environment '{dataset}' (port {port})...")
        return BatchedRemoteEnv.connect(
            port, self.pallete, os.path.join(str(self.run_id), dataset), **self._env_kwargs
        )

    def _apply_saved_time_feature_config(self):
        """A checkpoint's model config decides whether the time feature is on."""
        if not self.args.ac_model:
            return
        config = read_model_config(os.path.dirname(self.args.ac_model))
        if int(config["time_feat_dim"]) <= 0:
            self.hp.time_feature = False
            self.hp.time_scale = None
            return
        self.hp.time_feature = True
        if config["time_scale"] not in (None, 0, 0.0):
            self.hp.time_scale = float(config["time_scale"])

    def _create_policy(self):
        time_scale = float(self.env.time_scale)
        if self.hp.time_feature:
            print(f"Time feature on, time_scale={time_scale}")
        else:
            print("Time feature off")
        config = model_config(
            self.hp,
            self.args.hidden_dim,
            TIME_FEAT_DIM if self.hp.time_feature else 0,
            time_scale if self.hp.time_feature else 0.0,
        )
        write_model_config(self.args.model_path, config)

        self.policy = Policy.create(config, self.pallete, self.args.device, self.hp.masking)
        self.ac = self.policy.ac
        print(f"Actor-critic parameters: {sum(p.numel() for p in self.ac.parameters()):,}")
        if self.args.no_nn_updates:
            for param in self.ac.parameters():
                param.requires_grad = False

    def _create_optimizers(self):
        if self.hp.encoder == "shared":
            self.ac_optim = Adam(self.ac.parameters(), lr=self.hp.lr, eps=1e-5)
        else:
            self.a_optim = Adam(self.ac.policy_parameters(), lr=self.hp.lr, eps=1e-5)
            self.c_optim = Adam(self.ac.value_parameters(), lr=self.hp.lr, eps=1e-5)

    def learn(self, total_timesteps):
        """Train until ``total_timesteps`` fitness evaluations or convergence, then run the final evaluation."""
        print(
            f"Learning... {self.n_steps} steps per environment, "
            f"{self.timesteps_per_batch} transitions per batch, "
            f"{self.num_envs} parallel environments "
            f"({self.hp.n_updates_per_iteration} updates of {self.hp.mini_batch_size} steps)",
            flush=True,
        )
        t_so_far = 0  # Fitness evaluations so far
        e_so_far = 0  # Episodes so far
        i_so_far = 0  # Iterations so far
        t_start = time.perf_counter()
        eval_log = None
        metrics = self.train_log.metrics

        if self.validating:
            print("Evaluating initial policy before training...", flush=True)
            eval_log = self._validate(0, 0)
            self.train_log.flush_eval_log(0, 0)

        while t_so_far < total_timesteps:
            t_rollout = time.perf_counter()
            print(
                "Using truncated rollouts!"
                if self.args.partial_trajectories
                else "Using full trajectory rollouts!",
                flush=True,
            )
            batch = self.collector.collect(full_episodes=not self.args.partial_trajectories)
            metrics.update(batch.stats)
            metrics["rollout_seconds"] = time.perf_counter() - t_rollout

            t_update = time.perf_counter()
            # Fitness evaluations: steps that returned a successor model, counted by the server.
            t_so_far = self.env.get_fe_count()
            e_so_far += batch.finished_episodes
            i_so_far += 1
            metrics["t_so_far"] = t_so_far
            metrics["e_so_far"] = e_so_far
            metrics["i_so_far"] = i_so_far
            self._update(batch)
            metrics["update_seconds"] = time.perf_counter() - t_update

            eval_log = None
            if self.validating and i_so_far % self.eval_interval == 0:
                eval_log = self._validate(i_so_far, t_so_far)

            if self._check_convergence(i_so_far, t_so_far):
                self.train_log.flush_eval_log(i_so_far, t_so_far)
                print(
                    f"Early stopping triggered after {self.early_stopping.patience} updates without improvement"
                )
                print(f"Best reward achieved: {self.early_stopping.best_reward:.3f}")
                break

            self.train_log.summarize_iteration()
            self.train_log.write_train_log(
                i_so_far, t_so_far, self.reward_norm.take_stats(), self.collector.take_step_stats()
            )
            self.train_log.log_histograms(self.ac, i_so_far)

        print(f"Training finished after {i_so_far} iterations ({t_so_far} timesteps)", flush=True)
        self._finish_training(i_so_far, t_so_far, t_start, eval_log)

    def _finish_training(self, iterations, fes, t_start, eval_log):
        """Validate the last policy, run the best one on the final datasets, and close the run."""
        if self.validating:
            if eval_log is None:
                self._validate(iterations, fes)
            self.train_log.flush_eval_log(iterations, fes)
        training_seconds = time.perf_counter() - t_start - self.time_validation

        if self.final_datasets:
            self._load_best_policy()
            for dataset in self.final_datasets:
                self.run_final_evaluation(dataset)

        if self.validating:
            self.eval_env.close()
        for env in self._final_envs.values():
            env.close()

        self.env.post_training()
        self.env.close()
        self.train_log.write_runtimes(
            training_seconds, self.time_validation, iterations, fes, self.final_runtimes
        )

    def _update(self, batch):
        """All epochs of minibatch updates on one rollout, until done or the KL target stops them."""
        device = self.args.device
        metrics = self.train_log.metrics
        obs = [o.to(device) for o in batch.obs]
        act, nodes, vals_raw = stack_actions(batch.acts, self.pallete.max_len, device)
        if vals_raw is not None and self.hp.value_head == "bins":
            metrics["value_bin_indices"] = vals_raw.detach().cpu().reshape(-1)
        old_logp = batch.log_probs.to(device)
        returns = batch.returns.to(device)
        advantages = batch.advantages.to(device)
        self._record_batch_stats(returns, advantages, batch.model_ids)

        indices = np.arange(len(obs))
        kl_early_stop = False
        epochs_run = 0
        for epoch in range(self.hp.n_updates_per_iteration):
            epochs_run = epoch + 1
            np.random.shuffle(indices)
            for start in range(0, len(obs), self.hp.mini_batch_size):
                mb = np.asarray(indices[start : start + self.hp.mini_batch_size])
                kl_early_stop = not self._update_minibatch(
                    [obs[i] for i in mb],
                    act[mb],
                    nodes[mb] if nodes is not None else None,
                    vals_raw[mb] if vals_raw is not None else None,
                    old_logp[mb],
                    returns[mb],
                    advantages[mb],
                )
                if kl_early_stop:
                    break
            if kl_early_stop:
                break

        metrics["epochs_run"] = epochs_run
        metrics["kl_early_stop"] = kl_early_stop
        optimizer = self.ac_optim if self.hp.encoder == "shared" else self.a_optim
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

    def _record_batch_stats(self, returns, advantages, model_ids):
        metrics = self.train_log.metrics
        with torch.no_grad():
            predicted_values = returns - advantages
            metrics["explained_variance"] = explained_variance(returns, predicted_values)
            metrics["explained_variance_by_instance"] = self._per_instance_explained_variance(
                returns, predicted_values, model_ids
            )
            metrics["batch_returns_mean"] = float(returns.mean().item())
            metrics["batch_returns_std"] = float(returns.std().item()) if returns.numel() > 1 else float("nan")

    def _per_instance_explained_variance(self, returns, predicted_values, model_ids):
        if not model_ids:
            return {}
        sample_instance = np.asarray(model_ids, dtype=np.int64)
        by_name = {}
        for instance_id, name in enumerate(self.env.instance_names):
            mask = torch.as_tensor(sample_instance == instance_id, device=returns.device)
            by_name[name] = explained_variance(returns[mask], predicted_values[mask])
        return by_name

    def _update_minibatch(self, obs, act, nodes, vals_raw, old_logp, returns, advantages):
        """One gradient step. Returns False, without a step, when the KL target is exceeded."""
        A_k = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        V, stats = self.policy.evaluate(Batch.from_data_list(obs), act, nodes, vals_raw)
        curr_logp = stats["logp"]
        log_ratio = curr_logp.sum(-1) - old_logp.sum(-1)
        with torch.no_grad():
            approx_kl = (-log_ratio).mean()
            approx_kl_k3 = ((log_ratio.exp() - 1) - log_ratio).mean()
            factor_kl = (old_logp - curr_logp).mean(dim=0)

        if self.hp.target_kl and approx_kl.item() > 1.5 * self.hp.target_kl:
            return False

        # The ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        ratios = torch.exp(log_ratio)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.hp.clip, 1 + self.hp.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean() - self._entropy_bonus(stats).mean()
        critic_loss = 0.5 * nn.MSELoss()(V, returns)

        self.train_log.record_minibatch(
            actor_losses=actor_loss.detach(),
            critic_losses=critic_loss.detach(),
            entropy=(stats["ent_rule"] + stats["ent_node"] + stats["ent_val"]).detach(),
            entropy_rule=stats["ent_rule"].detach(),
            entropy_node=stats["ent_node"].detach(),
            entropy_val=stats["ent_val"].detach(),
            entropy_rule_norm=stats["ent_rule_norm"].detach(),
            entropy_node_norm=stats["ent_node_norm"].detach(),
            entropy_val_norm=(
                stats["ent_val_norm"].detach()
                if stats.get("ent_val_norm") is not None
                else torch.zeros_like(stats["ent_val"].detach())
            ),
            approx_kl=approx_kl.detach(),
            approx_kl_k3=approx_kl_k3.detach(),
            approx_kl_rule=factor_kl[0].detach(),
            approx_kl_node=factor_kl[1].detach(),
            approx_kl_val=factor_kl[2].detach(),
            clip_fraction=((ratios.detach() - 1.0).abs() > self.hp.clip).float().mean(),
            ratio_min=ratios.min().detach(),
            ratio_max=ratios.max().detach(),
            adv_mean=advantages.mean().detach(),
            adv_std=advantages.std().detach(),
        )

        if not self.args.no_nn_updates:
            gn_policy, gn_critic = self._gradient_step(actor_loss, critic_loss)
            self.train_log.record_minibatch(grad_norm=gn_policy.detach(), grad_norm_critic=gn_critic.detach())
        return True

    def _entropy_bonus(self, stats):
        normalize = self.hp.entropy_normalize
        ent_rule = stats["ent_rule_norm"] if normalize else stats["ent_rule"]
        ent_node = stats["ent_node_norm"] if normalize else stats["ent_node"]
        ent_val = (
            stats["ent_val_norm"]
            if normalize and stats.get("ent_val_norm") is not None
            else stats["ent_val"]
        )
        return (
            self.hp.entropy_coef_rule * ent_rule
            + self.hp.entropy_coef_node * ent_node
            + self.hp.entropy_coef_val * ent_val
        )

    def _gradient_step(self, actor_loss, critic_loss):
        """Backpropagate and step the optimizers; returns the gradient norms before clipping."""
        gn_critic = torch.tensor(0.0)
        if self.hp.encoder == "shared":
            self.ac_optim.zero_grad()
            (actor_loss + critic_loss).backward()
            gn_policy = torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
            self.ac_optim.step()
        else:
            self.a_optim.zero_grad()
            self.c_optim.zero_grad()
            actor_loss.backward()
            gn_policy = torch.nn.utils.clip_grad_norm_(self.ac.policy_parameters(), max_norm=0.5)
            self.a_optim.step()
            critic_loss.backward()
            gn_critic = torch.nn.utils.clip_grad_norm_(self.ac.value_parameters(), max_norm=0.5)
            self.c_optim.step()
        return gn_policy, gn_critic

    def _check_convergence(self, update_step, fes):
        """Tick on the training return, the one metric behind the best policy and convergence.

        A tick needs a finite metric and enough new episodes. Returns whether to stop. With
        --select_on val the validation phases tick instead (see ``_tick_validation``).
        """
        if self.select_on_val:
            self.train_log.metrics["es_ticked"] = int(self._val_ticked)
            self._val_ticked = False
            return self._val_should_stop
        metric = self.episode_stats.get_windowed_macro_return(self.hp.es_window)
        should_stop = improved = ticked = False
        if self.episode_stats.episodes_since_tick >= self.hp.es_tick_episodes and math.isfinite(metric):
            self.episode_stats.consume_tick()
            should_stop, improved = self.early_stopping.step(metric, update_step, fes)
            ticked = True
            mean_return, std_return = self.episode_stats.get_window_return_stats(self.hp.es_window)
            self._record_checkpoint(update_step, fes, metric, mean_return, std_return, improved)
        self.train_log.metrics["es_metric"] = metric
        self.train_log.metrics["es_ticked"] = int(ticked)
        return should_stop

    def _record_checkpoint(self, update_step, fes, metric, mean_return, std_return, improved):
        """Store the policy if due, and log this tick of the selection metric."""
        file_name = None
        if improved or self.args.keep_checkpoints == "all":
            file_name = f"ac_{update_step}-{fes}-{metric:.3f}.pth"
            torch.save(self.ac.state_dict(), os.path.join(self.args.model_path, file_name))
        self.train_log.add_checkpoint_row(update_step, fes, file_name, metric, mean_return, std_return, improved)

    def _validate(self, update_step, fes):
        log = run_validation(
            self.policy,
            self.eval_env,
            update_step,
            fes,
            self.eval_variants,
            self.args.seed,
            self.args.model_path,
            self.args.eval_save_solutions,
        )
        for variant in self.eval_variants:
            self.episode_stats.add_eval_mean_return(log[f"eval_return_{variant}"], variant)
        self.time_validation += log["time_eval"]
        self.train_log.add_eval_log(log)
        if self.select_on_val:
            self._tick_validation(update_step, fes, log)
        return log

    def _tick_validation(self, update_step, fes, log):
        """One tick of early stopping and the best policy on the stochastic validation return."""
        metric = log["eval_return_stochastic"]
        should_stop, improved = self.early_stopping.step(metric, update_step, fes)
        self._val_should_stop = self._val_should_stop or should_stop
        self._val_ticked = True
        self._record_checkpoint(
            update_step, fes, metric, metric, log["eval_return_stochastic_std"], improved
        )
        self.train_log.metrics["es_metric"] = metric

    def _load_best_policy(self):
        path = os.path.join(self.args.model_path, "best_model.pth")
        if not os.path.exists(path):
            warnings.warn("No best policy was stored; using the current one.", stacklevel=2)
            return
        self.policy.load_weights(path)

    def run_final_evaluation(self, dataset):
        """Run the current policy on each model of a served dataset, one batch per model.

        Sampled batches go to final_eval/<dataset>, greedy ones to final_eval/<dataset>-greedy.
        """
        if dataset == "val":
            env = self.eval_env
        else:
            if dataset not in self._final_envs:
                self._final_envs[dataset] = self._connect_eval_env(dataset)
            env = self._final_envs[dataset]
        for variant in EVAL_VARIANTS[self.args.final_mode]:
            label = dataset if variant == "stochastic" else f"{dataset}-greedy"
            root = os.path.join(self.args.model_path, "final_eval", label)
            self.final_runtimes[label] = sample_batches(
                self.policy, env, root, 1, self.args.seed, greedy=variant == "greedy"
            )
