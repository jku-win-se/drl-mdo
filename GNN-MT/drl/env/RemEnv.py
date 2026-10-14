import numpy as np
from drl.env.env_util import generate_tensors_from_graph
from drl.env.time_feature import (
    attach_time_feature,
    fetch_instance_names,
    instance_index_by_name,
    resolve_max_steps,
    warn_horizon_mismatch,
)
import procbridge as pb
from drl.utils.pallete import add_inv_edges
from drl.conversion import ConverterUtility
from drl.model2graph.model2graph import get_graph_from_resource
import networkx as nx
import torch
import multiprocess as mp


# Use multiprocess profiler for subprocess compatibility

def worker(remote, env_fn_wrapper, enable_profiling=False):
    import cProfile
    import tempfile
    import os
    from pathlib import Path
    
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        env = env_fn_wrapper()
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    finally:
        if profiler:
            profiler.disable()
            profile_dir = Path(tempfile.gettempdir()) / "worker_profiles"
            profile_dir.mkdir(exist_ok=True)
            profile_file = profile_dir / f"worker_{os.getpid()}.prof"
            profiler.dump_stats(str(profile_file))


def make_env(
    metamodel,
    pallete,
    idx,
    port=8000,
    max_steps=None,
    time_feature=False,
    time_scale=None,
):
    def _thunk():
        return RemEnv(
            idx,
            metamodel,
            pallete,
            port=port,
            max_steps=max_steps,
            time_feature=time_feature,
            time_scale=time_scale,
        )
    return _thunk

class SubprocEnv:
    def __init__(
        self,
        num_envs,
        metamodel,
        pallete,
        enable_profiling=False,
        port=8000,
        time_feature=True,
        time_scale=None,
    ):
        self.num_envs = num_envs
        self.main_remote = pb.Client('127.0.0.1', port)
        self.max_steps = resolve_max_steps(self.main_remote.request, num_envs)
        names = fetch_instance_names(self.main_remote.request, num_envs)
        self.instance_ids, self.instance_names = instance_index_by_name(names)
        self.time_feature = bool(time_feature)
        if time_scale is None:
            self.time_scale = float(max(self.max_steps))
        else:
            self.time_scale = float(time_scale)
        env_fns = [
            make_env(
                metamodel,
                pallete,
                idx,
                port,
                max_steps=self.max_steps[idx],
                time_feature=self.time_feature,
                time_scale=self.time_scale,
            )
            for idx in range(num_envs)
        ]
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_fns])
        self.ps = []
        for work_remote, env_fn in zip(self.work_remotes, env_fns):
            p = mp.Process(target=worker, args=(work_remote, env_fn, enable_profiling))
            p.daemon = True
            p.start()
            work_remote.close()
            self.ps.append(p)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones = zip(*results)
        return obs, np.array(rews), np.array(dones, dtype=bool)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()                  
    

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]
    
    

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Force terminating process {p.pid}")
                p.terminate()
                p.join()


    def post_rollout(self):
        self.main_remote.request("processEnvSolutions", False)

    def post_training(self):
        solution_set = self.main_remote.request("processSolutions", True)
        return solution_set
    
    def post_evaluation(self):
        self.main_remote.request("processSolutions", False)
    
    def get_env_solutions_and_clear(self, env_id):
        return self.main_remote.request("getEnvSolutionsAndClear", env_id)

    def get_env_model_name(self, env_id):
        return self.main_remote.request("getEnvModelName", env_id)

    def get_maxsteps(self):
        return self.main_remote.request("getMaxSteps")
    
class RemEnv:
    def __init__(
        self,
        idx,
        metamodel,
        pallete,
        mode="train",
        port=8000,
        max_steps=None,
        time_feature=False,
        time_scale=None,
    ):
      
        if mode == "train":
            self.port = port
        elif mode == "eval":
            self.port = port+1

        self.pallete = pallete
        self.metamodel = metamodel
        self.idx = idx
        self.max_steps = None if max_steps is None else int(max_steps)
        self.time_feature = bool(time_feature) and self.max_steps is not None
        self.time_scale = None if time_scale is None else float(time_scale)
        self.step_count = 0
        self._horizon_mismatch_warned = set()
        self.conv_util = ConverterUtility(metamodel)
        self.remote_env = pb.Client('127.0.0.1', self.port)

        rv = self.remote_env.request("reset", idx)

        state_xmi_str = rv[0]

        self.conditions = {int(k) if k.isdigit() else k: v for k, v in rv[1]["conditions"].items()} if rv[1] else None

        state = self.conv_util.resource_from_xmi_str("myuri" + str(idx), state_xmi_str)
        self.initial_graph = add_inv_edges(self.pallete.remove_out_of_scope(get_graph_from_resource(state)))
        self.graph = self.initial_graph
        self.step_count = 0
        self.graph_data = self._with_time_feature(
            generate_tensors_from_graph(self.initial_graph, self.pallete, self.conditions),
            0,
        )


    def reset(self, seed=None, options=None):
        """Initializes a random graph"""
        try:
            rv = self.remote_env.request("reset", self.idx)
        except Exception as e:
            print(f"Error in reset: {e}")
            raise


        if rv[1]:
            self.conditions = {int(k) if k.isdigit() else k: v for k, v in rv[1]["conditions"].items()} if rv[1] else None
        state = self.conv_util.resource_from_xmi_str("myuri" + str(self.idx), rv[0])
        self.graph = add_inv_edges(self.pallete.remove_out_of_scope(get_graph_from_resource(state)))
        self.step_count = 0
        self.graph_data = self._with_time_feature(
            generate_tensors_from_graph(self.graph, self.pallete, self.conditions),
            0,
        )
        
        return self._get_observation()

    def _with_time_feature(self, data, steps_taken):
        if not self.time_feature:
            return data
        return attach_time_feature(data, steps_taken, self.max_steps, self.time_scale)

    def _get_observation(self):
        return self.graph_data

    def step(self, action):
        rule_id = action[0].item()
        
        node_ids = []
        integer_sequence = []
        if action[1] is not None:
            node_ids = action[1][0]
        
        if action[2] is not None:
            integer_sequence = action[2][0][0]

        paramValues = {}
        id_attr = self.pallete.get_node_id_attr_name()

        for idd, sampled_node in enumerate(node_ids[:self.pallete.get_num_node_params_for_action(rule_id)]):
            node_graph_idx = sampled_node.item()
            if node_graph_idx < 0:
                print(f"Node {node_graph_idx} does not have the required attribute '{id_attr}'")
                return None, .0, False
            node_atts = self.graph.nodes[node_graph_idx]["atts"]
            if id_attr not in node_atts:
                print(f"Node {node_graph_idx} does not have the required attribute '{id_attr}'")
                return None, .0, False
            paramValues[self.pallete.get_name_by_param_pos(rule_id, idd)] = node_atts[id_attr]

        for idd, sampled_int in enumerate(integer_sequence):
            paramValues[self.pallete.get_name_by_param_pos(rule_id, idd+self.pallete.max_len)] = int(torch.round(sampled_int).item())

        r = self.remote_env.request("step", {
                            "_envId": self.idx,
                            "_ruleName": self.pallete.get_unitname_by_id(rule_id),
                            "_paramValues": paramValues})
        if r is None:
            raise ValueError("Received None from remote environment.")
            
        
        rew = r[1]
        done = bool(r[2])
        next_obs = r[0]

        # False is a non-executable rule: store nothing and leave the step counter unchanged.
        if next_obs is False:
            if done:
                warn_horizon_mismatch(
                    self._horizon_mismatch_warned,
                    self.idx,
                    self.step_count,
                    self.max_steps if self.max_steps is not None else self.step_count,
                )
                self.step_count = 0
            return None, 0.0, done

        if next_obs:
            state_xmi_str = next_obs

            self.graph = self.serialized_to_graph(state_xmi_str)
            self.step_count += 1
            steps_for_obs = self.step_count
            if done:
                horizon = self.max_steps if self.max_steps is not None else self.step_count
                warn_horizon_mismatch(
                    self._horizon_mismatch_warned,
                    self.idx,
                    self.step_count,
                    horizon,
                )
                self.step_count = 0
                steps_for_obs = 0
            self.graph_data = self._with_time_feature(
                generate_tensors_from_graph(self.graph, self.pallete, self.conditions),
                steps_for_obs,
            )

        return self._get_observation(), rew, done

    def serialized_to_graph(self, serialized_graph):
        """
        Convert a serialized graph to a NetworkX graph.
        """
        try:
            state = self.conv_util.resource_from_xmi_str("myuri" + str(self.idx), serialized_graph)
            return add_inv_edges(self.pallete.remove_out_of_scope(get_graph_from_resource(state)))
        except Exception as e:
            print(f"Error converting serialized graph: {e}")
            return None