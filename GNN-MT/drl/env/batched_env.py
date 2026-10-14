import time

import numpy as np
import torch

from drl.env.compact_obs import decode_compact_obs
from drl.env.pb_client import PersistentClient
from drl.env.time_feature import (
    attach_time_feature,
    fetch_instance_names,
    instance_index_by_name,
    resolve_max_steps,
    warn_horizon_mismatch,
)
from drl.utils.pallete import SEP_INV, SPECIAL


def split_env_groups(num_envs, n_groups):
    n_groups = max(1, min(int(n_groups), num_envs))
    sizes = [num_envs // n_groups] * n_groups
    for i in range(num_envs % n_groups):
        sizes[i] += 1
    groups = []
    start = 0
    for size in sizes:
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def parse_rule_conditions(cond):
    if not cond:
        return None
    inner = cond.get("conditions", cond) if isinstance(cond, dict) else cond
    if not isinstance(inner, dict):
        return None
    return {int(k) if str(k).isdigit() else k: v for k, v in inner.items()}


class BatchedRemoteEnv:
    """Vectorized remote env using persistent connections and compact observations."""

    def __init__(
        self,
        num_envs,
        pallete,
        port=8000,
        groups=1,
        host="127.0.0.1",
        time_feature=True,
        time_scale=None,
    ):
        self.num_envs = num_envs
        self.pallete = pallete
        self.port = port
        self.host = host
        self.groups = split_env_groups(num_envs, groups)
        self.control = PersistentClient(host, port)
        self.clients = [PersistentClient(host, port) for _ in self.groups]
        # Env indices of the stepAll each group is waiting on, in result order.
        self.pending_env_indices = [None] * len(self.groups)
        self.obs = [None] * num_envs
        self.node_ids = [None] * num_envs
        self.conditions = [None] * num_envs
        self.last_decode_time = 0.0
        self.time_feature = bool(time_feature)
        self.max_steps = resolve_max_steps(self.control.request, num_envs)
        if time_scale is None:
            self.time_scale = float(max(self.max_steps))
        else:
            self.time_scale = float(time_scale)
        names = fetch_instance_names(self.control.request, num_envs)
        self.instance_ids, self.instance_names = instance_index_by_name(names)
        self.step_counts = [0] * num_envs
        self._horizon_mismatch_warned = set()

        self.control.request("configureObservation", self.build_observation_spec())
        self.store_reset_observations(self.control.request("resetAll"))

    def build_observation_spec(self):
        edge_types = {
            name: idx
            for name, idx in self.pallete.dic_edges.items()
            if not name.endswith(SEP_INV) and name != SPECIAL
        }
        return {
            "nodeTypes": dict(self.pallete.dic_nodes),
            "edgeTypes": edge_types,
            "idAttribute": self.pallete.get_node_id_attr_name(),
            "numericAttributes": list(self.pallete.additional_node_atts or []),
        }

    def store_reset_observations(self, results):
        obs = []
        t0 = time.perf_counter()
        for env_index, row in enumerate(results):
            payload, cond = row[0], row[1] if len(row) > 1 else None
            self.conditions[env_index] = parse_rule_conditions(cond)
            data, nids = decode_compact_obs(
                payload, self.pallete, self.conditions[env_index]
            )
            self.step_counts[env_index] = 0
            data = self._with_time_feature(data, env_index, steps_taken=0)
            self.obs[env_index] = data
            self.node_ids[env_index] = nids
            obs.append(data)
        self.last_decode_time = time.perf_counter() - t0
        return obs

    def encode_action_payload(self, env_index, action):
        # Keys match the Java step / stepAll contract used by RemEnv.
        rule_id = action[0].item()
        node_ids_t = []
        integer_sequence = []
        if action[1] is not None:
            node_ids_t = action[1][0]
        if action[2] is not None:
            integer_sequence = action[2][0][0]

        param_values = {}
        nids = self.node_ids[env_index]
        nparams = self.pallete.get_num_node_params_for_action(rule_id)
        for param_index, sampled_node in enumerate(node_ids_t[:nparams]):
            node_graph_idx = int(sampled_node.item())
            if (
                nids is None
                or node_graph_idx < 0
                or node_graph_idx >= len(nids)
                or nids[node_graph_idx] is None
            ):
                return {"_envId": env_index, "_skip": True}
            param_values[self.pallete.get_name_by_param_pos(rule_id, param_index)] = (
                nids[node_graph_idx]
            )

        for param_index, sampled_int in enumerate(integer_sequence):
            param_values[
                self.pallete.get_name_by_param_pos(
                    rule_id, param_index + self.pallete.max_len
                )
            ] = int(torch.round(sampled_int).item())

        return {
            "_envId": env_index,
            "_ruleName": self.pallete.get_unitname_by_id(rule_id),
            "_paramValues": param_values,
        }

    def decode_step_result(self, env_index, row):
        raw_obs = row[0]
        rew = row[1]
        done = bool(row[2])
        cond = row[3] if len(row) > 3 else None
        parsed_cond = parse_rule_conditions(cond)
        if parsed_cond is not None:
            self.conditions[env_index] = parsed_cond

        if raw_obs is None:
            return None, rew, done
        # False is a non-executable rule. Nothing is stored and the budget is unchanged,
        # matching the invalid-node skip. A done flag means the retry cap ended the episode.
        if raw_obs is False:
            if done:
                warn_horizon_mismatch(
                    self._horizon_mismatch_warned,
                    env_index,
                    self.step_counts[env_index],
                    self.max_steps[env_index],
                )
                self.step_counts[env_index] = 0
            return None, 0.0, done

        data, nids = decode_compact_obs(
            raw_obs, self.pallete, self.conditions[env_index]
        )
        steps_taken = self._advance_step_counter(env_index, done)
        data = self._with_time_feature(data, env_index, steps_taken)
        self.obs[env_index] = data
        self.node_ids[env_index] = nids
        return data, rew, done

    def _with_time_feature(self, data, env_index, steps_taken):
        if not self.time_feature:
            return data
        return attach_time_feature(
            data, steps_taken, self.max_steps[env_index], self.time_scale
        )

    def _advance_step_counter(self, env_index, done):
        """Count a stored transition. After done the returned observation is a fresh episode."""
        self.step_counts[env_index] += 1
        if not done:
            return self.step_counts[env_index]
        taken = self.step_counts[env_index]
        warn_horizon_mismatch(
            self._horizon_mismatch_warned,
            env_index,
            taken,
            self.max_steps[env_index],
        )
        self.step_counts[env_index] = 0
        return 0

    def step_async_group(
        self, group_index, group_actions, env_indices=None, auto_reset=True
    ):
        """Send one stepAll for a group, or for the subset ``env_indices`` of it.

        Returns the encoded action payloads, in the order of ``env_indices``.
        """
        if env_indices is None:
            env_indices = self.groups[group_index]
        env_indices = list(env_indices)
        actions = [
            self.encode_action_payload(env_index, action)
            for env_index, action in zip(env_indices, group_actions)
        ]
        self.pending_env_indices[group_index] = env_indices
        self.clients[group_index].send(
            "stepAll", {"_autoReset": bool(auto_reset), "_actions": actions}
        )
        return actions

    def step_wait_group(self, group_index):
        """Results of the group's pending stepAll, in the order its envs were sent."""
        results = self.clients[group_index].recv()
        env_indices = self.pending_env_indices[group_index]
        self.pending_env_indices[group_index] = None
        t0 = time.perf_counter()
        obs_out = []
        rews = []
        dones = []
        for local_index, env_index in enumerate(env_indices):
            obs, rew, done = self.decode_step_result(env_index, results[local_index])
            obs_out.append(obs)
            rews.append(rew)
            dones.append(done)
        self.last_decode_time = time.perf_counter() - t0
        return obs_out, np.array(rews), np.array(dones, dtype=bool)

    def step_async(self, actions):
        for group_index, idxs in enumerate(self.groups):
            self.step_async_group(group_index, [actions[i] for i in idxs])

    def step_wait(self):
        obs_all = [None] * self.num_envs
        rew_all = np.zeros(self.num_envs)
        done_all = np.zeros(self.num_envs, dtype=bool)
        decode_total = 0.0
        for group_index in range(len(self.groups)):
            env_indices = self.pending_env_indices[group_index]
            obs, rew, done = self.step_wait_group(group_index)
            decode_total += self.last_decode_time
            for local_index, env_index in enumerate(env_indices):
                obs_all[env_index] = obs[local_index]
                rew_all[env_index] = rew[local_index]
                done_all[env_index] = done[local_index]
        self.last_decode_time = decode_total
        return obs_all, rew_all, done_all

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        return self.store_reset_observations(self.control.request("resetAll"))

    def close(self):
        for client in self.clients:
            client.close()
        self.control.close()

    def post_rollout(self):
        self.control.request("processEnvSolutions", False)

    def post_training(self):
        return self.control.request("processSolutions", True)

    def post_evaluation(self):
        self.control.request("processSolutions", False)

    def get_env_solutions_and_clear(self, env_id):
        return self.control.request("getEnvSolutionsAndClear", env_id)

    def get_env_model_name(self, env_id):
        return self.control.request("getEnvModelName", env_id)

    def get_maxsteps(self):
        return self.control.request("getMaxSteps")
