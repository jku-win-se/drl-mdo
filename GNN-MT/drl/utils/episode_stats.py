"""
Episode statistics tracking module for RL training.
Provides a rolling buffer system similar to Stable-Baselines3 for monitoring training progress.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union


class EpisodeStats:
    """
    Rolling buffer for episode statistics similar to Stable-Baselines3.

    Maintains a fixed-size buffer of recent episode statistics to provide
    robust performance monitoring during RL training. When num_envs is set,
    also tracks per-environment (per-difficulty) returns so a macro-average
    can weight each instance equally regardless of episode count.
    """

    def __init__(
        self,
        buffer_size: int = 50,
        num_envs: int = 0,
        instance_ids: Optional[List[int]] = None,
        instance_names: Optional[List[str]] = None,
        es_warmup_episodes: int = 5,
    ):
        """
        Initialize the episode statistics buffer.

        Args:
            buffer_size: Maximum number of episodes to keep in the buffer
            num_envs: Number of parallel environments (0 disables per-env tracking)
            instance_ids: Dense instance id of each environment. Replicas of one
                model share an id. Defaults to one instance per environment.
            instance_names: Name of each instance, aligned with the dense ids.
            es_warmup_episodes: Episodes per instance used to freeze the return scale.
        """
        if es_warmup_episodes < 1:
            raise ValueError(
                f"es_warmup_episodes must be at least 1, got {es_warmup_episodes}"
            )
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.es_warmup_episodes = es_warmup_episodes
        if instance_ids is None:
            self.instance_ids = list(range(num_envs))
        else:
            if len(instance_ids) != num_envs:
                raise ValueError(
                    f"instance_ids has length {len(instance_ids)}, expected {num_envs}"
                )
            self.instance_ids = [int(instance_id) for instance_id in instance_ids]
        self.num_instances = (max(self.instance_ids) + 1) if self.instance_ids else 0
        if self.instance_ids and sorted(set(self.instance_ids)) != list(range(self.num_instances)):
            raise ValueError(
                "instance_ids must be dense and start at 0, "
                f"got {sorted(set(self.instance_ids))}"
            )
        if instance_names is None:
            self.instance_names = [str(i) for i in range(self.num_instances)]
        else:
            self.instance_names = [str(name) for name in instance_names]
        if len(self.instance_names) != self.num_instances:
            raise ValueError(
                f"instance_names has length {len(self.instance_names)}, "
                f"expected {self.num_instances}"
            )
        self.returns = deque(maxlen=buffer_size)
        self.lengths = deque(maxlen=buffer_size)
        self.rollout_mean_returns = []
        self.rollout_macro_returns: List[float] = []
        self.rollout_pooled_returns: List[float] = []
        self.rollout_env_means: List[np.ndarray] = []
        self.rollout_per_step_returns: List[float] = []
        self.eval_mean_returns: Dict[str, List[float]] = {}
        self.total_episodes = 0
        self.new_episodes = 0
        self.episodes_since_tick = 0

        self.env_returns: List[deque] = [
            deque(maxlen=buffer_size) for _ in range(num_envs)
        ]
        self.env_lengths: List[deque] = [
            deque(maxlen=buffer_size) for _ in range(num_envs)
        ]
        self.instance_returns: List[List[float]] = [
            [] for _ in range(self.num_instances)
        ]
        self.instance_return_scale: List[Optional[float]] = [
            None
        ] * self.num_instances
        self._init_rollout_accumulators()
        self._last_env_stats: Dict[str, np.ndarray] = {
            "means": np.full(num_envs, np.nan, dtype=np.float64),
            "stds": np.zeros(num_envs, dtype=np.float64),
            "counts": np.zeros(num_envs, dtype=np.int32),
            "mean_lengths": np.full(num_envs, np.nan, dtype=np.float64),
        }
        self._last_instance_stats: Dict[str, np.ndarray] = {
            "means": np.full(self.num_instances, np.nan, dtype=np.float64),
            "stds": np.zeros(self.num_instances, dtype=np.float64),
            "counts": np.zeros(self.num_instances, dtype=np.int32),
            "mean_lengths": np.full(self.num_instances, np.nan, dtype=np.float64),
        }
        self._fresh_instance_count = 0

    def _init_rollout_accumulators(self) -> None:
        self._rollout_env_returns: List[List[float]] = [[] for _ in range(self.num_envs)]
        self._rollout_env_lengths: List[List[int]] = [[] for _ in range(self.num_envs)]
        self._rollout_instance_returns: List[List[float]] = [
            [] for _ in range(self.num_instances)
        ]
        self._rollout_instance_lengths: List[List[int]] = [
            [] for _ in range(self.num_instances)
        ]
        self._rollout_reward_sum = 0.0
        self._rollout_steps = 0

    def start_rollout(self) -> None:
        """Clear per-rollout accumulators before collecting a new batch."""
        self._init_rollout_accumulators()
        self.new_episodes = 0

    def consume_tick(self) -> int:
        """Return episodes finished since the last tick and reset that counter."""
        completed = self.episodes_since_tick
        self.episodes_since_tick = 0
        return completed

    def fresh_instance_count(self) -> int:
        """Number of instances that finished at least one episode in the last rollout."""
        return self._fresh_instance_count

    def _freeze_instance_scale(self, instance_id: int) -> None:
        if self.instance_return_scale[instance_id] is not None:
            return
        returns = self.instance_returns[instance_id]
        if len(returns) < self.es_warmup_episodes:
            return
        warmup = returns[: self.es_warmup_episodes]
        mean_absolute_return = float(np.mean(np.abs(warmup)))
        self.instance_return_scale[instance_id] = max(mean_absolute_return, 1e-3)

    def get_windowed_macro_return(self, window: int) -> float:
        """Mean over instances of (mean of the last ``window`` returns) / frozen scale.

        NaN until every instance has a frozen scale and at least ``window`` returns.
        The scale stays fixed so the metric can be compared across iterations.
        """
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        if self.num_instances == 0:
            return float("nan")
        normalized_means = []
        for instance_id in range(self.num_instances):
            scale = self.instance_return_scale[instance_id]
            returns = self.instance_returns[instance_id]
            if scale is None or len(returns) < window:
                return float("nan")
            window_mean = float(np.mean(returns[-window:]))
            normalized_means.append(window_mean / scale)
        return float(np.mean(normalized_means))

    def add_step_reward(self, raw_rew: float) -> None:
        """Accumulate a raw step reward for the composition-invariant per-step return."""
        self._rollout_reward_sum += float(raw_rew)
        self._rollout_steps += 1

    def add_eval_mean_return(self, mean_return: float, variant: str) -> None:
        """Record the mean return of one evaluation phase of ``variant`` (stochastic/greedy)."""
        self.eval_mean_returns.setdefault(variant, []).append(float(mean_return))

    def add_rollout_mean_return(self, mean_return: float) -> None:
        """Alias of the pooled rollout mean for backward compatibility."""
        self.rollout_mean_returns.append(float(mean_return))
        if not self.rollout_pooled_returns or self.rollout_pooled_returns[-1] != float(
            mean_return
        ):
            self.rollout_pooled_returns.append(float(mean_return))

    def get_last_rollout_mean_return(self) -> float:
        """Pooled mean return of the last recorded rollout (alias)."""
        return self.get_last_rollout_pooled_return()

    def get_last_rollout_pooled_return(self) -> float:
        """Mean return over all episodes completed in the last rollout."""
        if not self.rollout_pooled_returns:
            return 0.0
        return self.rollout_pooled_returns[-1]

    def get_last_rollout_macro_return(self) -> float:
        """Mean of per-env mean returns from the last rollout (equal weight per env)."""
        if not self.rollout_macro_returns:
            return 0.0
        return self.rollout_macro_returns[-1]

    def get_last_rollout_per_step_return(self) -> float:
        """Sum of raw rewards divided by number of stored steps in the last rollout."""
        if not self.rollout_per_step_returns:
            return 0.0
        return self.rollout_per_step_returns[-1]

    def get_last_rollout_env_stats(self) -> Dict[str, np.ndarray]:
        """Per-env means, stds, counts and mean lengths from the last rollout."""
        return {
            "means": self._last_env_stats["means"].copy(),
            "stds": self._last_env_stats["stds"].copy(),
            "counts": self._last_env_stats["counts"].copy(),
            "mean_lengths": self._last_env_stats["mean_lengths"].copy(),
        }

    def get_last_rollout_instance_stats(self) -> Dict[str, np.ndarray]:
        """Per-instance means, stds, counts and mean lengths from the last rollout."""
        return {
            "means": self._last_instance_stats["means"].copy(),
            "stds": self._last_instance_stats["stds"].copy(),
            "counts": self._last_instance_stats["counts"].copy(),
            "mean_lengths": self._last_instance_stats["mean_lengths"].copy(),
        }

    def _record_instance_rollout(self) -> None:
        means = np.full(self.num_instances, np.nan, dtype=np.float64)
        stds = np.zeros(self.num_instances, dtype=np.float64)
        counts = np.zeros(self.num_instances, dtype=np.int32)
        mean_lengths = np.full(self.num_instances, np.nan, dtype=np.float64)
        for instance_id in range(self.num_instances):
            returns = self._rollout_instance_returns[instance_id]
            lengths = self._rollout_instance_lengths[instance_id]
            counts[instance_id] = len(returns)
            if not returns:
                continue
            means[instance_id] = float(np.mean(returns))
            stds[instance_id] = float(np.std(returns)) if len(returns) > 1 else 0.0
            mean_lengths[instance_id] = float(np.mean(lengths))
        self._last_instance_stats = {
            "means": means,
            "stds": stds,
            "counts": counts,
            "mean_lengths": mean_lengths,
        }
        self._fresh_instance_count = int(np.sum(counts > 0))

    def add_episode(
        self,
        episode_return: Union[float, int],
        episode_length: int,
        env_id: Optional[int] = None,
    ) -> None:
        """
        Add a completed episode to the buffer.

        Args:
            episode_return: Total reward for the completed episode
            episode_length: Number of steps in the completed episode
            env_id: Environment index (required for per-env tracking)
        """
        ret = float(episode_return)
        length = int(episode_length)
        self.returns.append(ret)
        self.lengths.append(length)
        self.total_episodes += 1
        if env_id is not None and 0 <= env_id < self.num_envs:
            self.env_returns[env_id].append(ret)
            self.env_lengths[env_id].append(length)
            self._rollout_env_returns[env_id].append(ret)
            self._rollout_env_lengths[env_id].append(length)
            instance_id = self.instance_ids[env_id]
            self.instance_returns[instance_id].append(ret)
            self._rollout_instance_returns[instance_id].append(ret)
            self._rollout_instance_lengths[instance_id].append(length)
            self._freeze_instance_scale(instance_id)
            self.new_episodes += 1
            self.episodes_since_tick += 1

    def end_rollout(self) -> Dict[str, float]:
        """
        Finalize the current rollout: compute per-env, pooled, macro and per-step
        returns. Envs that completed no episode this rollout fall back to the mean
        of their rolling deque (nan if they have never finished).
        """
        env_means = np.full(self.num_envs, np.nan, dtype=np.float64)
        env_stds = np.zeros(self.num_envs, dtype=np.float64)
        env_counts = np.zeros(self.num_envs, dtype=np.int32)
        env_mean_lengths = np.full(self.num_envs, np.nan, dtype=np.float64)
        all_returns: List[float] = []

        for i in range(self.num_envs):
            rollout_rets = self._rollout_env_returns[i]
            rollout_lens = self._rollout_env_lengths[i]
            env_counts[i] = len(rollout_rets)
            if rollout_rets:
                env_means[i] = float(np.mean(rollout_rets))
                env_stds[i] = (
                    float(np.std(rollout_rets)) if len(rollout_rets) > 1 else 0.0
                )
                env_mean_lengths[i] = float(np.mean(rollout_lens))
                all_returns.extend(rollout_rets)
            elif self.env_returns[i]:
                env_means[i] = float(np.mean(self.env_returns[i]))
                env_stds[i] = (
                    float(np.std(self.env_returns[i]))
                    if len(self.env_returns[i]) > 1
                    else 0.0
                )
                env_mean_lengths[i] = float(np.mean(self.env_lengths[i]))

        pooled_return = float(np.mean(all_returns)) if all_returns else float("nan")
        if self.num_envs > 0 and np.any(~np.isnan(env_means)):
            macro_return = float(np.nanmean(env_means))
        else:
            macro_return = pooled_return
        per_step_return = (
            float(self._rollout_reward_sum / self._rollout_steps)
            if self._rollout_steps > 0
            else 0.0
        )

        self._last_env_stats = {
            "means": env_means,
            "stds": env_stds,
            "counts": env_counts,
            "mean_lengths": env_mean_lengths,
        }
        self._record_instance_rollout()
        self.rollout_macro_returns.append(macro_return)
        self.rollout_pooled_returns.append(pooled_return)
        self.rollout_mean_returns.append(pooled_return)
        self.rollout_env_means.append(env_means.copy())
        self.rollout_per_step_returns.append(per_step_return)

        return {
            "macro_return": macro_return,
            "pooled_return": pooled_return,
            "per_step_return": per_step_return,
        }

    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive statistics from all episodes in the buffer.

        Returns:
            Dictionary containing mean, std, min, max statistics for returns and lengths,
            plus episode counts
        """
        if len(self.returns) == 0:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "min_return": 0.0,
                "max_return": 0.0,
                "num_episodes": 0,
                "total_episodes": self.total_episodes,
            }

        return {
            "mean_return": float(np.mean(self.returns)),
            "std_return": float(np.std(self.returns)),
            "mean_length": float(np.mean(self.lengths)),
            "std_length": float(np.std(self.lengths)),
            "min_return": float(np.min(self.returns)),
            "max_return": float(np.max(self.returns)),
            "num_episodes": len(self.returns),
            "total_episodes": self.total_episodes,
        }

    def get_recent_stats(self, n: int = 10) -> Dict[str, float]:
        """
        Get statistics from the most recent n episodes.

        Args:
            n: Number of recent episodes to analyze

        Returns:
            Dictionary containing statistics for the most recent episodes
        """
        if len(self.returns) == 0:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "mean_length": 0.0,
                "num_episodes": 0,
            }

        # Get the most recent n episodes (or all if fewer than n)
        recent_returns = list(self.returns)[-n:]
        recent_lengths = list(self.lengths)[-n:]

        return {
            "mean_return": float(np.mean(recent_returns)),
            "std_return": (
                float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
            ),
            "mean_length": float(np.mean(recent_lengths)),
            "num_episodes": len(recent_returns),
        }

    def get_percentiles(
        self, percentiles: list = [25, 50, 75, 90, 95]
    ) -> Dict[str, float]:
        """
        Get percentile statistics for episode returns.

        Args:
            percentiles: List of percentiles to compute

        Returns:
            Dictionary mapping percentile names to values
        """
        if len(self.returns) == 0:
            return {f"return_p{p}": 0.0 for p in percentiles}

        return {
            f"return_p{p}": float(np.percentile(self.returns, p)) for p in percentiles
        }

    def is_improving(self, window_size: int = 10, threshold: float = 0.01) -> bool:
        """
        Check if performance is improving based on recent episodes.

        Args:
            window_size: Size of the window to compare
            threshold: Minimum improvement threshold (relative)

        Returns:
            True if recent performance is better than earlier performance
        """
        if len(self.returns) < 2 * window_size:
            return False

        recent_mean = np.mean(list(self.returns)[-window_size:])
        earlier_mean = np.mean(list(self.returns)[-2 * window_size : -window_size])

        if earlier_mean == 0:
            return recent_mean > 0

        improvement = (recent_mean - earlier_mean) / abs(earlier_mean)
        return improvement > threshold

    def reset(self) -> None:
        """Reset the buffer while keeping total episode count."""
        self.returns.clear()
        self.lengths.clear()
        for env_buf in self.env_returns:
            env_buf.clear()
        for env_buf in self.env_lengths:
            env_buf.clear()
        for instance_returns in self.instance_returns:
            instance_returns.clear()
        self.instance_return_scale = [None] * self.num_instances
        self.new_episodes = 0
        self.episodes_since_tick = 0
        self._fresh_instance_count = 0
        self._init_rollout_accumulators()

    def __len__(self) -> int:
        """Return the number of episodes currently in the buffer."""
        return len(self.returns)

    def __repr__(self) -> str:
        """String representation of the buffer state."""
        stats = self.get_stats()
        return (
            f"EpisodeStats(buffer_size={self.buffer_size}, "
            f"num_envs={self.num_envs}, "
            f"episodes={stats['num_episodes']}, "
            f"total={stats['total_episodes']}, "
            f"mean_return={stats['mean_return']:.3f})"
        )
