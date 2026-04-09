"""
Episode statistics tracking module for RL training.
Provides a rolling buffer system similar to Stable-Baselines3 for monitoring training progress.
"""

import numpy as np
from collections import deque
from typing import Dict, Union


class EpisodeStats:
    """
    Rolling buffer for episode statistics similar to Stable-Baselines3.

    Maintains a fixed-size buffer of recent episode statistics to provide
    robust performance monitoring during RL training.
    """

    def __init__(self, buffer_size: int = 50):
        """
        Initialize the episode statistics buffer.

        Args:
            buffer_size: Maximum number of episodes to keep in the buffer
        """
        self.buffer_size = buffer_size
        self.returns = deque(maxlen=buffer_size)
        self.lengths = deque(maxlen=buffer_size)
        self.rollout_mean_returns = []
        self.eval_mean_returns = []
        self.total_episodes = 0

    def add_eval_mean_return(self, mean_return: float) -> None:
        """
        Add the mean return of an evaluation phase to the tracking list.

        Args:
            mean_return: Mean return of the evaluation phase
        """
        self.eval_mean_returns.append(float(mean_return))

    def get_last_eval_mean_return(self) -> float:
        """
        Get the mean return of the last recorded evaluation phase.

        Returns:
            Mean return of the last evaluation phase, or 0.0 if none recorded
        """
        if not self.eval_mean_returns:
            return 0.0
        return self.eval_mean_returns[-1]

    def add_rollout_mean_return(self, mean_return: float) -> None:
        """
        Add the mean return of a rollout to the tracking list.

        Args:
            mean_return: Mean return of the rollout
        """
        self.rollout_mean_returns.append(float(mean_return))

    def get_last_rollout_mean_return(self) -> float:
        """
        Get the mean return of the last recorded rollout.

        Returns:
            Mean return of the last rollout, or 0.0 if none recorded
        """
        if not self.rollout_mean_returns:
            return 0.0
        return self.rollout_mean_returns[-1]

    def add_episode(
        self, episode_return: Union[float, int], episode_length: int
    ) -> None:
        """
        Add a completed episode to the buffer.

        Args:
            episode_return: Total reward for the completed episode
            episode_length: Number of steps in the completed episode
        """
        self.returns.append(float(episode_return))
        self.lengths.append(int(episode_length))
        self.total_episodes += 1

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

    def __len__(self) -> int:
        """Return the number of episodes currently in the buffer."""
        return len(self.returns)

    def __repr__(self) -> str:
        """String representation of the buffer state."""
        stats = self.get_stats()
        return (
            f"EpisodeStats(buffer_size={self.buffer_size}, "
            f"episodes={stats['num_episodes']}, "
            f"total={stats['total_episodes']}, "
            f"mean_return={stats['mean_return']:.3f})"
        )
