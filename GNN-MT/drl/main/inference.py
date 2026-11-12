"""Running a policy: sampling episodes, the inference modes, validation and the final run.

``InferenceSampler`` plays episodes on a ``BatchedRemoteEnv`` with the env groups
double-buffered, as in the training rollout. While Java applies the actions of one group,
the policy runs on the next. An ``EpisodeSchedule`` decides which environments start another
episode and when sampling stops. Environments without an episode left to play are left out
of the batch, so every forward pass only covers envs that still have work.
"""

import csv
import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from paretoset import paretoset
from tqdm import tqdm

from drl.env.batched_env import BatchedRemoteEnv, StepTimers
from drl.main.policy import Policy, read_model_config
from drl.utils.episode_stats import metric_name
from utils.seed_util import seed_everything

logger = logging.getLogger(__name__)

INFERENCE_MODES = ("one-per-model", "fixed-count", "until-converged", "max-likelihood", "batches")


@contextmanager
def isolated_rng(seed):
    """Sample from ``seed`` inside the block and leave the training RNG as it was."""
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ParetoArchive:
    """Non-dominated set of objective vectors (minimization) over all episodes."""

    def __init__(self):
        self.front = None

    def add(self, solutions):
        """Merge ``solutions`` into the front and return whether it changed."""
        if solutions is None or len(solutions) == 0:
            return False

        new = np.array(solutions, dtype=np.float64)
        if new.ndim == 1:
            new = new.reshape(1, -1)

        if self.front is None or len(self.front) == 0:
            self.front = new[paretoset(new)]
            return len(self.front) > 0

        combined = np.vstack([self.front, new])
        candidate = combined[paretoset(combined)]

        # Compare the sets on rounded tuples so float noise does not count as improvement.
        old_set = set(map(tuple, np.round(self.front, 8)))
        new_set = set(map(tuple, np.round(candidate, 8)))
        improved = new_set != old_set
        if improved:
            self.front = candidate
        return improved

    def tolist(self):
        return self.front.tolist() if self.front is not None else []

    def __len__(self):
        return 0 if self.front is None else len(self.front)


@dataclass
class EpisodeRecord:
    """One episode of one environment.

    ``steps`` holds the applied transformations as ``{rule, params, reward}``.
    ``solutions`` are the objective vectors of the non-dominated states Java
    recorded during the episode. ``stalled`` marks a greedy episode that ended
    early because the most likely action did not change the model.
    """

    env_index: int
    total_reward: float = 0.0
    length: int = 0
    steps: list = field(default_factory=list)
    solutions: list = field(default_factory=list)
    stalled: bool = False


@dataclass
class SamplingResult:
    episodes: list
    pareto_front: ParetoArchive
    elapsed: float
    time_inference: float
    time_env_wait: float
    time_decode: float

    @property
    def returns(self):
        return [episode.total_reward for episode in self.episodes]

    @property
    def mean_return(self):
        return statistics.mean(self.returns) if self.episodes else float("nan")

    def describe(self):
        returns = self.returns
        if returns:
            stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
            reward_line = (
                f"Episodes: {len(returns)} | mean return {self.mean_return:.4f} "
                f"(+- {stdev:.4f}), max {max(returns):.4f}, min {min(returns):.4f}"
            )
        else:
            reward_line = "Episodes: 0"
        return (
            f"{reward_line}\n"
            f"Pareto set size: {len(self.pareto_front)}\n"
            f"Time: {self.elapsed:.3f}s total | inference {self.time_inference:.3f}s, "
            f"env wait {self.time_env_wait:.3f}s, decode {self.time_decode:.3f}s"
        )


class EpisodeSchedule:
    """Decides which environments play another episode and when sampling stops.

    ``start_episode`` is called for every env at the start and again after each
    of its episodes ends. It returns whether that env plays another episode.
    """

    # Java resets a finished env in the same stepAll when this is on.
    auto_reset = True
    # Number of episodes for the progress bar, or None if open-ended.
    progress_total = None

    @property
    def stopped(self):
        return False

    def start_episode(self, env_index):
        raise NotImplementedError

    def on_episode_end(self, improved):
        pass


class OneEpisodePerEnv(EpisodeSchedule):
    """Each of the first ``num_envs`` envs plays exactly one episode; the others stay idle."""

    auto_reset = False

    def __init__(self, num_envs):
        self.progress_total = num_envs
        self.started = set()

    def start_episode(self, env_index):
        if env_index in self.started or env_index >= self.progress_total:
            return False
        self.started.add(env_index)
        return True


class FixedEpisodeCount(EpisodeSchedule):
    """Exactly ``num_episodes`` episodes, spread over the available envs."""

    def __init__(self, num_episodes):
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        self.progress_total = num_episodes
        self.num_episodes = num_episodes
        self.started = 0

    def start_episode(self, env_index):
        if self.started >= self.num_episodes:
            return False
        self.started += 1
        return True


class UntilNoImprovement(EpisodeSchedule):
    """Episodes until ``patience`` episodes in a row leave the Pareto set unchanged."""

    def __init__(self, patience):
        if patience < 1:
            raise ValueError(f"patience must be positive, got {patience}")
        self.patience = patience
        self.episodes_since_improvement = 0

    @property
    def stopped(self):
        return self.episodes_since_improvement >= self.patience

    def start_episode(self, env_index):
        return not self.stopped

    def on_episode_end(self, improved):
        if improved:
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1


@dataclass
class PendingStep:
    """A stepAll sent for some envs of a group, waiting for its results."""

    group_index: int
    env_indices: list
    payloads: list


class InferenceSampler:
    """Plays episodes with ``policy`` on ``env`` as scheduled by ``schedule``.

    With ``greedy`` the policy decodes the most likely action, which is
    deterministic. A step that leaves the model unchanged would then repeat
    forever, so such an episode ends early and is marked ``stalled``.
    """

    def __init__(self, policy, schedule, env, greedy=False, desc="Episodes"):
        if greedy and schedule.auto_reset:
            raise ValueError(
                "Greedy decoding is deterministic; pair it with one episode per env."
            )
        self.policy = policy
        self.timers = StepTimers()
        self.env = env
        self.schedule = schedule
        self.greedy = greedy
        self.desc = desc
        self.groups = self.env.groups
        self.num_envs = self.env.num_envs

        self.archive = ParetoArchive()
        self.finished = []
        self.episodes = [EpisodeRecord(i) for i in range(self.num_envs)]
        self.active = [False] * self.num_envs
        self.progress = None

    def run(self):
        self.progress = tqdm(total=self.schedule.progress_total, desc=self.desc)
        start = time.perf_counter()

        self.active = [self.schedule.start_episode(i) for i in range(self.num_envs)]
        pending = [self.launch_group(g) for g in range(len(self.groups))]
        while any(step is not None for step in pending):
            for group_index, step in enumerate(pending):
                if step is None:
                    continue
                self.collect_group(step)
                pending[group_index] = (
                    None if self.schedule.stopped else self.launch_group(group_index)
                )
        self.discard_unfinished_episodes()

        elapsed = time.perf_counter() - start
        self.progress.close()
        return SamplingResult(
            episodes=self.finished,
            pareto_front=self.archive,
            elapsed=elapsed,
            time_inference=self.timers.inference,
            time_env_wait=self.timers.env_wait,
            time_decode=self.timers.decode,
        )

    def launch_group(self, group_index):
        """Infer and send actions for the active envs of a group, or None if it has none."""
        env_indices = [i for i in self.groups[group_index] if self.active[i]]
        if not env_indices:
            return None
        # Actions must come from the observation the env stored last for each index.
        obs = [self.env.obs[i] for i in env_indices]
        t0 = time.perf_counter()
        with torch.inference_mode():
            (actions, _, _), _ = self.policy.get_action_and_value(obs, action_only=True)
        self.timers.inference += time.perf_counter() - t0
        payloads = self.env.step_async_group(
            group_index,
            actions,
            env_indices=env_indices,
            auto_reset=self.schedule.auto_reset,
        )
        return PendingStep(group_index, env_indices, payloads)

    def collect_group(self, step):
        next_obs, rew, done = self.env.timed_wait_group(step.group_index, self.timers)
        for local_index, env_index in enumerate(step.env_indices):
            self.record_step(
                env_index,
                step.payloads[local_index],
                next_obs[local_index],
                float(rew[local_index]),
                bool(done[local_index]),
            )

    def record_step(self, env_index, payload, obs, reward, done):
        episode = self.episodes[env_index]
        # None without done: the action was skipped (invalid node) or the rule did
        # not apply. The env keeps its observation and no step is counted.
        if obs is None and not done:
            if self.greedy:
                logger.warning(
                    "Env %d: the most likely action does not change the model "
                    "after %d steps; ending its episode early.",
                    env_index,
                    episode.length,
                )
                self.finish_episode(env_index, stalled=True)
            return

        episode.total_reward += reward
        episode.length += 1
        if obs is not None and "_ruleName" in payload:
            episode.steps.append(
                {
                    "rule": payload["_ruleName"],
                    "params": payload["_paramValues"],
                    "reward": reward,
                }
            )
        if done:
            self.finish_episode(env_index)

    def finish_episode(self, env_index, stalled=False):
        # Safe: the env's group has no stepAll pending while its results are processed.
        episode = self.episodes[env_index]
        episode.solutions = self.env.get_env_solutions_and_clear(env_index)
        episode.stalled = stalled
        improved = self.archive.add(episode.solutions)
        self.finished.append(episode)
        self.schedule.on_episode_end(improved)

        self.episodes[env_index] = EpisodeRecord(env_index)
        self.active[env_index] = (
            not self.schedule.stopped and self.schedule.start_episode(env_index)
        )

        self.progress.update(1)
        postfix = {"pareto": len(self.archive)}
        if isinstance(self.schedule, UntilNoImprovement):
            postfix["no_improvement"] = self.schedule.episodes_since_improvement
        self.progress.set_postfix(postfix)

    def discard_unfinished_episodes(self):
        """Drop what Java recorded for episodes cut off when sampling stopped."""
        for env_index, episode in enumerate(self.episodes):
            if episode.length > 0:
                self.env.get_env_solutions_and_clear(env_index)


def append_csv(path, row):
    """Append a dict as a row, writing the header when the file is new."""
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def write_solutions_to_dir(
    seed, solution_set, solutions_path, file_path, sampled_episodes, time_taken, mean_rew, verbose=True
):
    os.makedirs(solutions_path, exist_ok=True)
    if verbose:
        print("Writing solutions to", solutions_path)
    with open(os.path.join(solutions_path, file_path), "w") as f:
        for sol in solution_set:
            f.write(str(sol) + "\n")
    with open(os.path.join(solutions_path, "info.txt"), "a") as f:
        f.write(
            f"Inference time (seed={seed}): {time_taken:.3f} seconds ({sampled_episodes} episodes)\n"
        )
        if mean_rew is not None:
            f.write(f"Mean reward (seed={seed}): {mean_rew:.3f}\n")


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


def write_per_model_results(result, env, model_dir, seed, extra):
    """Write the solutions and the transformation sequence of each model's episode.

    ``model_dir(model_name)`` gives the output directory, ``extra`` is added to
    each ``sequence.json``.
    """
    for model_name, episode in episodes_by_model(result, env):
        stalled_note = " (stalled)" if episode.stalled else ""
        print(
            f"Model: {model_name}, return: {episode.total_reward:.4f}, "
            f"steps: {episode.length}{stalled_note}"
        )
        solutions_path = model_dir(model_name)
        write_solutions_to_dir(
            seed, episode.solutions, solutions_path, "nn-solutions.txt", 1, result.elapsed, episode.total_reward
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


def write_merged_results(result, model_dir, seed):
    """Write the merged front of a model's episodes."""
    archive = ParetoArchive()
    for episode in result.episodes:
        archive.add(episode.solutions)
    write_solutions_to_dir(
        seed,
        archive.tolist(),
        model_dir,
        "nn-solutions.txt",
        len(result.episodes),
        result.elapsed,
        float(np.mean([e.total_reward for e in result.episodes])),
        verbose=False,
    )


def sample_batches(policy, env, root, batches, seed, greedy=False):
    """Play ``batches`` batches of one episode per env on each model, one model at a time.

    Writes ``<root>/<model>/nn-solutions.txt``: per batch an FE line, then the archive so
    far. Models run one after another, so each model's time is its own. Greedy decoding is
    deterministic, so with ``greedy`` one env plays a single batch of one episode per model.
    Returns the times.
    """
    if greedy and batches != 1:
        raise ValueError("Greedy decoding repeats the same episode; use one batch.")
    runtimes = {}
    policy.eval()
    policy.set_greedy(greedy)
    try:
        _sample_batches(policy, env, root, batches, seed, greedy, runtimes)
    finally:
        policy.set_greedy(False)
        policy.train()
    return runtimes


def _sample_batches(policy, env, root, batches, seed, greedy, runtimes):
    with isolated_rng(seed):
        for model, name in enumerate(env.instance_names):
            model_dir = os.path.join(root, name)
            os.makedirs(model_dir, exist_ok=True)
            archive = ParetoArchive()
            fe_start = env.get_fe_count()
            seconds = []
            episodes = 0
            time_inference = time_env_wait = 0.0
            with open(os.path.join(model_dir, "nn-solutions.txt"), "w") as f:
                for _ in range(batches):
                    env.reset_to_model(model)
                    result = InferenceSampler(
                        policy, OneEpisodePerEnv(1 if greedy else env.num_envs), env, greedy=greedy,
                        desc=f"{'Greedy' if greedy else 'Sample'} {name}",
                    ).run()
                    for episode in result.episodes:
                        archive.add(episode.solutions)
                    seconds.append(result.elapsed)
                    episodes += len(result.episodes)
                    time_inference += result.time_inference
                    time_env_wait += result.time_env_wait
                    f.write(f"{env.get_fe_count() - fe_start}\n")
                    for solution in archive.tolist():
                        f.write(f"{solution}\n")
            runtimes[name] = {
                "seconds": sum(seconds),
                "batch_seconds": seconds,
                "inference_seconds": time_inference,
                "env_wait_seconds": time_env_wait,
                "episodes": episodes,
                "fes": env.get_fe_count() - fe_start,
            }


def run_validation(policy, env, update_step, fes, variants, seed, model_path, save_solutions, scales):
    """Record solutions and returns of one batch per validation model.

    Each model's mean return is divided by a scale frozen in the first phase of a variant:
    the mean absolute return there, at least 1e-3. ``scales`` maps a variant to these
    scales and is filled on that first call. The mean of the normalized returns
    (``eval_macro_return_<variant>``) weighs every model equally, whatever its return
    magnitude; it can tick early stopping and the best policy (--select_on val).
    All envs play one model at a time. Sampling uses its own RNG with a fixed seed, so
    validating does not change training. Returns the metrics of this phase.
    """
    label = f"iter{update_step:05d}-t{fes}"
    log = {}
    t0 = time.perf_counter()
    policy.eval()
    try:
        for variant in variants:
            greedy = variant == "greedy"
            policy.set_greedy(greedy)
            all_returns = []
            model_returns = []
            time_env_wait = time_inference = time_decode = 0.0
            with isolated_rng(seed):
                for model, name in enumerate(env.instance_names):
                    env.reset_to_model(model)
                    # Greedy episodes on one model are identical, so one env plays it.
                    result = InferenceSampler(
                        policy,
                        OneEpisodePerEnv(1 if greedy else env.num_envs),
                        env,
                        greedy=greedy,
                        desc=f"Eval {variant} {name}",
                    ).run()
                    returns = [float(e.total_reward) for e in result.episodes]
                    all_returns += returns
                    model_returns.append(returns)
                    time_env_wait += result.time_env_wait
                    time_inference += result.time_inference
                    time_decode += result.time_decode
                    log[f"eval_return_{variant}_{metric_name(name)}"] = float(np.mean(returns))
                    if save_solutions:
                        write_merged_results(
                            result, os.path.join(model_path, "eval_solutions", variant, name, label), seed
                        )
            if variant not in scales:
                scales[variant] = [max(float(np.mean(np.abs(r))), 1e-3) for r in model_returns]
            normalized = []
            for name, returns, scale in zip(env.instance_names, model_returns, scales[variant]):
                normalized.append(float(np.mean(returns)) / scale)
                append_csv(
                    os.path.join(model_path, "val_returns.csv"),
                    {
                        "variant": variant,
                        "update": update_step,
                        "fe": fes,
                        "model": name,
                        "mean_return": float(np.mean(returns)),
                        "std_return": float(np.std(returns)),
                        "episodes": len(returns),
                        "scale": scale,
                        "normalized_return": normalized[-1],
                    },
                )
            mean_return = float(np.mean(all_returns))
            macro_return = float(np.mean(normalized))
            print(
                f"[eval {variant} @ {label}] mean return {mean_return:.4f}, macro {macro_return:.4f}",
                flush=True,
            )
            log[f"eval_return_{variant}"] = mean_return
            log[f"eval_macro_return_{variant}"] = macro_return
            log[f"eval_return_{variant}_std"] = float(np.std(all_returns))
            log[f"time_eval_{variant}_env_wait"] = time_env_wait
            log[f"time_eval_{variant}_inference"] = time_inference
            log[f"time_eval_{variant}_decode"] = time_decode
    finally:
        policy.set_greedy(False)
        policy.train()
    log["time_eval"] = time.perf_counter() - t0
    return log


def _schedule_for(mode, num_envs, num_solutions, converge_patience):
    if mode in ("one-per-model", "max-likelihood"):
        return OneEpisodePerEnv(num_envs)
    if mode == "fixed-count":
        if not num_solutions or num_solutions < 1:
            raise ValueError("fixed-count inference needs a positive num_solutions.")
        return FixedEpisodeCount(num_solutions)
    if mode == "until-converged":
        return UntilNoImprovement(converge_patience or num_envs)
    raise ValueError(f"Unknown inference mode {mode!r}; use one of {INFERENCE_MODES}.")


def _output_dir(checkpoint, group, run_label, model_name=None):
    """``<runs dir>/sampled_solutions/<group>[/<model>]/<checkpoint>--<run_label>``."""
    checkpoint = Path(checkpoint).resolve()
    parts = [str(checkpoint.parent.parent), "sampled_solutions", group]
    if model_name is not None:
        parts.append(model_name)
    parts.append(f"{checkpoint.stem}--{run_label}")
    return os.path.join(*parts)


def _connect_for_checkpoint(checkpoint, pallete, port, run_id, env_groups, time_scale):
    """The env, with the time feature as the checkpoint was trained with it."""
    config = read_model_config(os.path.dirname(checkpoint))
    time_feature = int(config["time_feat_dim"]) > 0
    if time_feature and config["time_scale"] not in (None, 0, 0.0):
        time_scale = float(config["time_scale"])
    return BatchedRemoteEnv.connect(
        port,
        pallete,
        str(run_id),
        groups=env_groups,
        time_feature=time_feature,
        time_scale=time_scale if time_feature else None,
    )


def run_inference(
    mode, checkpoint, pallete, device, seed, group, port, run_id, env_groups=1, masking=True,
    time_scale=None, num_solutions=None, converge_patience=None, num_batches=None,
):
    """Sample solutions with a checkpoint on the models the server on ``port`` serves.

    Modes (``INFERENCE_MODES``):
    - ``one-per-model``: one stochastic episode per env, i.e. per model.
    - ``fixed-count``: exactly ``num_solutions`` episodes; their Pareto set is written.
    - ``until-converged``: episodes until ``converge_patience`` episodes in a row
      (default ``num_envs``) leave the Pareto set unchanged.
    - ``max-likelihood``: one greedy episode per model, with its transformation sequence.
    - ``batches``: ``num_batches`` batches of one episode per env on each model, logged per
      batch, to see how solution quality grows with continued sampling.
    """
    env = _connect_for_checkpoint(checkpoint, pallete, port, run_id, env_groups, time_scale)
    if mode == "batches":
        if not num_batches or num_batches < 1:
            raise ValueError("batches inference needs a positive num_batches.")
        policy = Policy.from_checkpoint(checkpoint, pallete, device, masking)
        root = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "sampling_curve", group)
        runtimes = sample_batches(policy, env, root, num_batches, seed)
        with open(os.path.join(root, "runtimes.json"), "w") as f:
            json.dump(runtimes, f, indent=2)
        return None

    greedy = mode == "max-likelihood"
    schedule = _schedule_for(mode, env.num_envs, num_solutions, converge_patience)
    policy = Policy.from_checkpoint(checkpoint, pallete, device, masking)
    policy.eval()
    policy.set_greedy(greedy)

    print(f"Running {mode} inference (seed={seed})")
    seed_everything(seed)
    result = InferenceSampler(policy, schedule, env, greedy=greedy).run()
    print(result.describe())

    if isinstance(schedule, OneEpisodePerEnv):
        # Greedy episodes are deterministic, so the seed does not name the run.
        run_label = "greedy" if greedy else str(seed)
        write_per_model_results(
            result,
            env,
            lambda model_name: _output_dir(checkpoint, group, run_label, model_name),
            seed,
            {"checkpoint": checkpoint},
        )
    else:
        write_solutions_to_dir(
            seed,
            result.pareto_front.tolist(),
            _output_dir(checkpoint, group, str(seed)),
            "nn-solutions.txt",
            len(result.episodes),
            result.elapsed,
            result.mean_return,
        )
    return result
