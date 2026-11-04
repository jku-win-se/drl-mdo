"""Collecting experience for PPO: stepping the envs, storing transitions, and GAE."""

import copy
import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from nqdm import nqdm
from torch_geometric.data import Batch

from drl.env.batched_env import StepTimers

NUM_LOGP_FACTORS = 3  # rule, node, value


def resolve_rollout_length(hp, num_envs, max_horizon):
    """``(n_steps, timesteps_per_batch)`` from 'auto', an explicit value, or timesteps_per_batch."""
    if hp.n_steps == "auto":
        if hp.timesteps_per_batch is not None:
            n_steps = math.ceil(hp.timesteps_per_batch / num_envs)
        else:
            n_steps = max_horizon
    else:
        n_steps = int(hp.n_steps)
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_steps < max_horizon:
        warnings.warn(
            f"n_steps={n_steps} is below the longest horizon {max_horizon}; "
            "some episodes will not finish inside one rollout.",
            stacklevel=2,
        )
    return n_steps, num_envs * n_steps


def compute_gae(rewards_per_env, dones_per_env, values_per_env, next_values_per_env, gamma, lam, device):
    """GAE over per-env lists of Python/numpy scalars. Empty envs are skipped."""
    all_returns = []
    all_advantages = []

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
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty

    returns_flat = torch.as_tensor(np.concatenate(all_returns), dtype=torch.float32, device=device)
    advantages_flat = torch.as_tensor(np.concatenate(all_advantages), dtype=torch.float32, device=device)
    return returns_flat, advantages_flat


def stack_actions(batch_acts, max_len, device):
    """Stacked ``(rules, nodes, raw values)`` of per-step actions, padded to ``max_len`` nodes."""
    batch_act = torch.stack(
        [a[0] if torch.is_tensor(a[0]) else torch.tensor(a[0]) for a in batch_acts]
    ).to(device=device, dtype=torch.long)
    batch_nodes = None
    batch_vals_raw = None
    if batch_acts[0][1] is not None:
        padded = [F.pad(a[1], (0, max_len - a[1].shape[1]), value=0) for a in batch_acts]
        batch_nodes = torch.stack(padded, dim=0).squeeze(1).to(device)
    if batch_acts[0][2] is not None:
        batch_vals_raw = torch.stack([a[2][1] for a in batch_acts], dim=0).squeeze(1).to(device)
    return batch_act, batch_nodes, batch_vals_raw


def with_forced_action(obs, act, node_masks, max_len):
    """A copy of ``obs`` that holds the taken action, for teacher forcing in the update.

    The copy keeps the live observation free of force_* keys: PyG collate takes keys from the
    first Data in the list, so mutating it would KeyError or drop attributes on the next Batch.
    """
    force_act_data = {"force_act": act[0].unsqueeze(0)}

    if act[1] is not None:
        num_nodes = int(obs["nodes"].item())
        seq_len = act[1].shape[1]

        oneh_nodes = torch.zeros(num_nodes, max_len, dtype=torch.float32, device=act[1].device)
        node_idx = act[1][0]
        valid = node_idx >= 0
        if valid.any():
            slots = torch.arange(seq_len, device=act[1].device)
            oneh_nodes[node_idx[valid], slots[valid]] = 1.0
        force_act_data["force_nodes"] = oneh_nodes

        nodes_mask = node_masks.T[:num_nodes]
        if nodes_mask.shape[1] < max_len:
            nodes_mask = F.pad(nodes_mask, (0, max_len - nodes_mask.shape[1]), value=0.0)
        force_act_data["force_nodes_mask"] = nodes_mask

    obs_copy = copy.copy(obs)
    obs_copy.update(force_act_data)
    return obs_copy


def _has_invalid_nodes(action):
    """An action whose node decoder found no legal node for some slot."""
    nodes = action[1] if action is not None and len(action) > 1 else None
    if nodes is None:
        return False
    if not torch.is_tensor(nodes):
        nodes = torch.as_tensor(nodes)
    return bool((nodes < 0).any())


@dataclass
class RolloutBatch:
    """Transitions of one rollout, flat over envs. ``stats`` are metrics for the logger."""

    obs: list
    acts: list
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    finished_episodes: int
    model_ids: list  # The model each transition ran on.
    stats: dict


@dataclass
class _EnvBuffers:
    """What one env collected since its last flush."""

    obs: list = field(default_factory=list)
    models: list = field(default_factory=list)
    acts: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rews: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    vals: list = field(default_factory=list)


class _Rollout:
    """State of one rollout in progress."""

    def __init__(self, num_envs, full_episodes, progress_bar):
        self.full_episodes = full_episodes
        self.progress_bar = progress_bar
        self.envs = [_EnvBuffers() for _ in range(num_envs)]
        self.completed = _EnvBuffers()  # Full-episode rollouts: transitions of finished episodes.
        self.ep_lens = []
        self.skipped_invalid_steps = 0
        self.skipped_invalid_nodes = 0
        self.episodes_per_env = np.zeros(num_envs, dtype=int)
        self.last_done = np.zeros(num_envs, dtype=np.float32)


class RolloutCollector:
    """Steps the envs with the policy, double-buffered across env groups.

    While Java applies the actions of one group, the policy runs on the next. Per-env episode
    returns accumulate across rollout boundaries, SB3 style, and only finished episodes are
    reported to ``episode_stats``.
    """

    def __init__(
        self, env, policy, pallete, episode_stats, reward_norm, gamma, gae_lambda, n_steps,
        timesteps_per_batch, device,
    ):
        self.env = env
        self.policy = policy
        self.pallete = pallete
        self.episode_stats = episode_stats
        self.reward_norm = reward_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.timesteps_per_batch = timesteps_per_batch
        self.device = device
        self.num_envs = env.num_envs
        self.timers = StepTimers()
        self.obs = env.reset()
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._steps_seen = 0
        self._steps_invalid = 0

    def collect(self, full_episodes):
        """One rollout: fixed-length segments, or only the transitions of finished episodes."""
        return self.collect_full_episodes() if full_episodes else self.collect_segments()

    def collect_segments(self):
        """A fixed-length batch per env. All transitions train, regardless of episode ends."""
        self.episode_stats.start_rollout()
        rollout = _Rollout(self.num_envs, False, nqdm(total=self.timesteps_per_batch, position=0, leave=True))

        def group_should_continue(indices):
            return any(len(rollout.envs[i].obs) < self.n_steps for i in indices)

        self._step_until_done(group_should_continue, rollout)
        rollout.progress_bar.close()
        self.env.post_rollout()
        self.episode_stats.end_rollout()

        to_bootstrap = [(i, self.obs[i]) for i in range(self.num_envs) if not rollout.last_done[i]]
        value_map = {}
        if to_bootstrap:
            indices, obs_batch = zip(*to_bootstrap)
            values = self.policy.get_value(Batch.from_data_list(obs_batch))
            value_map = {i: float(v) for i, v in zip(indices, values.detach().cpu().numpy())}

        returns, advantages = compute_gae(
            [env.rews for env in rollout.envs],
            [env.dones for env in rollout.envs],
            [env.vals for env in rollout.envs],
            [value_map.get(i, 0.0) for i in range(self.num_envs)],
            self.gamma,
            self.gae_lambda,
            self.device,
        )
        log_probs = torch.tensor(
            [lp for env in rollout.envs for lp in env.log_probs], dtype=torch.float32, device=self.device
        )
        return self._batch(
            rollout,
            obs=[obs for env in rollout.envs for obs in env.obs],
            acts=[act for env in rollout.envs for act in env.acts],
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            model_ids=[model for env in rollout.envs for model in env.models],
        )

    def collect_full_episodes(self):
        """Until the batch is full and every env finished an episode; unfinished ones don't count."""
        self.episode_stats.start_rollout()
        rollout = _Rollout(self.num_envs, True, nqdm(total=self.timesteps_per_batch, position=0, leave=True))

        def group_should_continue(_indices):
            batch_full = len(rollout.completed.obs) >= self.timesteps_per_batch
            return not (batch_full and all(count >= 1 for count in rollout.episodes_per_env))

        self._step_until_done(group_should_continue, rollout)
        rollout.progress_bar.close()
        self.env.post_rollout()
        self.episode_stats.end_rollout()

        completed = rollout.completed
        rews_per_ep, dones_per_ep, vals_per_ep = [], [], []
        start = 0
        for ep_len in rollout.ep_lens:
            end = start + ep_len
            rews_per_ep.append(completed.rews[start:end])
            dones_per_ep.append(completed.dones[start:end])
            vals_per_ep.append(completed.vals[start:end])
            start = end
        returns, advantages = compute_gae(
            rews_per_ep,
            dones_per_ep,
            vals_per_ep,
            [0.0] * len(rollout.ep_lens),
            self.gamma,
            self.gae_lambda,
            self.device,
        )
        log_probs = torch.tensor(completed.log_probs, dtype=torch.float32, device=self.device)
        return self._batch(
            rollout,
            obs=completed.obs,
            acts=completed.acts,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            model_ids=completed.models,
        )

    def _batch(self, rollout, obs, acts, log_probs, returns, advantages, model_ids):
        if log_probs.numel() == 0:
            log_probs = torch.zeros(0, NUM_LOGP_FACTORS, dtype=torch.float32, device=self.device)
        elif log_probs.ndim == 1:
            log_probs = log_probs.unsqueeze(-1)
        stats = {
            "skipped_invalid_steps": rollout.skipped_invalid_steps,
            "skipped_invalid_nodes": rollout.skipped_invalid_nodes,
            "invalid_action_rate": (
                float(rollout.skipped_invalid_steps) / float(len(obs)) if len(obs) > 0 else float("nan")
            ),
            "time_env_wait": self.timers.env_wait,
            "time_inference": self.timers.inference,
            "time_decode": self.timers.decode,
            "batch_lens": rollout.ep_lens,
        }
        if self.pallete.max_val_len > 0:
            stats["value_head_log_std"] = self.policy.value_head_log_std
        return RolloutBatch(
            obs, acts, log_probs, returns, advantages, len(rollout.ep_lens), model_ids, stats
        )

    def take_step_stats(self):
        """Share of invalid actions since the last call."""
        share = self._steps_invalid / max(self._steps_seen, 1)
        self._steps_seen = 0
        self._steps_invalid = 0
        return {"invalid_action_share": share}

    def _step_until_done(self, group_should_continue, rollout):
        self.timers = StepTimers()
        groups = self.env.groups
        pending = [None] * len(groups)
        for group_index, group in enumerate(groups):
            if group_should_continue(group):
                pending[group_index] = self._act_on_group(group_index, group)

        while any(step is not None for step in pending):
            for group_index, group in enumerate(groups):
                step = pending[group_index]
                if step is None:
                    continue
                next_obs, rew, done = self.env.timed_wait_group(group_index, self.timers)
                pending[group_index] = None
                for local_i, env_i in enumerate(step["indices"]):
                    self.record_transition(
                        env_i,
                        step["staged"][local_i],
                        step["action"][local_i],
                        step["log_prob"][local_i],
                        step["value"][local_i],
                        rew[local_i],
                        done[local_i],
                        next_obs[local_i],
                        rollout,
                    )
                if group_should_continue(group):
                    pending[group_index] = self._act_on_group(group_index, group)

    def _act_on_group(self, group_index, indices):
        """Infer actions for a group and send them; the teacher-forcing copies get built meanwhile."""
        obs_g = [self.obs[i] for i in indices]
        t0 = time.perf_counter()
        (action, log_prob, node_mask), value = self.policy.get_action_and_value(obs_g)
        self.timers.inference += time.perf_counter() - t0
        value = np.atleast_1d(np.asarray(value, dtype=np.float32).reshape(-1))
        staged = [
            with_forced_action(obs_g[i], action[i], node_mask[i], self.pallete.max_len)
            for i in range(len(obs_g))
        ]
        self.env.step_async_group(group_index, action)
        return {"indices": indices, "action": action, "log_prob": log_prob, "value": value, "staged": staged}

    def record_transition(self, i, staged_obs, action, log_prob, value, rew, done, next_obs, rollout):
        # The env's last step ran on this model, also if an auto-reset has moved the env on.
        model = int(self.env.step_models[i])
        info = self.env.step_infos[i]
        self._steps_seen += 1
        if info and info.get("invalid"):
            self._steps_invalid += 1

        if next_obs is None:
            if done:
                self.reward_norm.end_of_episode(model)
            rollout.skipped_invalid_steps += 1
            return

        if _has_invalid_nodes(action):
            rollout.skipped_invalid_nodes += 1
            self._advance_env(i, next_obs, rew, done, rollout)
            if done:
                self.reward_norm.end_of_episode(model)
                self._finish_episode(i, model, rollout)
                if rollout.full_episodes:
                    rollout.envs[i] = _EnvBuffers()
            return

        normalized_rew = self.reward_norm.normalize(i, model, rew)
        if done:
            self.reward_norm.end_of_episode(model)
        buffers = rollout.envs[i]
        buffers.obs.append(staged_obs)
        buffers.models.append(model)
        buffers.acts.append(action)
        buffers.log_probs.append(log_prob)
        buffers.rews.append(normalized_rew)
        buffers.dones.append(float(done))
        buffers.vals.append(float(value))
        self._advance_env(i, next_obs, rew, done, rollout)
        rollout.progress_bar.update(1)

        if done:
            rollout.ep_lens.append(int(self._episode_lengths[i]))
            self._finish_episode(i, model, rollout)
            if rollout.full_episodes:
                completed = rollout.completed
                completed.models.extend(buffers.models)
                completed.obs.extend(buffers.obs)
                completed.acts.extend(buffers.acts)
                completed.log_probs.extend(buffers.log_probs)
                completed.rews.extend(buffers.rews)
                completed.dones.extend(buffers.dones)
                completed.vals.extend(buffers.vals)
                rollout.envs[i] = _EnvBuffers()

    def _advance_env(self, i, next_obs, rew, done, rollout):
        rollout.last_done[i] = float(done)
        self._episode_rewards[i] += rew
        self._episode_lengths[i] += 1
        self.episode_stats.add_step_reward(rew)
        self.obs[i] = next_obs

    def _finish_episode(self, i, model, rollout):
        self.episode_stats.add_episode(
            float(self._episode_rewards[i]), int(self._episode_lengths[i]), env_id=i, instance_id=model
        )
        rollout.episodes_per_env[i] += 1
        self.reward_norm.reset_return(i)
        self._episode_rewards[i] = 0.0
        self._episode_lengths[i] = 0
