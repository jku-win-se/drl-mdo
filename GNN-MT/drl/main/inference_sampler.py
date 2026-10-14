"""Batched solution sampling with a trained policy.

``InferenceSampler`` plays episodes on a ``BatchedRemoteEnv`` with the env
groups double-buffered, as in the training rollout. While Java applies the
actions of one group, the policy runs on the next. An ``EpisodeSchedule``
decides which environments start another episode and when sampling stops.
Environments without an episode left to play are left out of the batch, so
every forward pass only covers envs that still have work.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from paretoset import paretoset
from tqdm import tqdm

logger = logging.getLogger(__name__)

INFERENCE_MODES = ("one-per-model", "fixed-count", "until-converged", "max-likelihood")


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
    """Every env (one per model) plays exactly one episode."""

    auto_reset = False

    def __init__(self, num_envs):
        self.progress_total = num_envs
        self.started = set()

    def start_episode(self, env_index):
        if env_index in self.started:
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
    """Plays episodes with ``ppo.ac`` on ``env`` (default ``ppo.env``) as scheduled by ``schedule``.

    With ``greedy`` the policy decodes the most likely action, which is
    deterministic. A step that leaves the model unchanged would then repeat
    forever, so such an episode ends early and is marked ``stalled``.
    """

    def __init__(self, ppo, schedule, greedy=False, env=None, desc="Episodes"):
        if greedy and schedule.auto_reset:
            raise ValueError(
                "Greedy decoding is deterministic; pair it with one episode per env."
            )
        env = ppo.env if env is None else env
        if not hasattr(env, "step_async_group"):
            raise ValueError("Inference sampling needs the batched env transport.")
        self.ppo = ppo
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
        self.ppo.time_inference = 0.0
        self.ppo.time_env_wait = 0.0
        self.ppo.time_decode = 0.0
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
            time_inference=self.ppo.time_inference,
            time_env_wait=self.ppo.time_env_wait,
            time_decode=self.ppo.time_decode,
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
            (actions, _, _), _ = self.ppo.get_action_and_value(obs, action_only=True)
        self.ppo.time_inference += time.perf_counter() - t0
        payloads = self.env.step_async_group(
            group_index,
            actions,
            env_indices=env_indices,
            auto_reset=self.schedule.auto_reset,
        )
        return PendingStep(group_index, env_indices, payloads)

    def collect_group(self, step):
        next_obs, rew, done = self.ppo.wait_group_results(step.group_index, self.env)
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
        if obs is not None:
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
