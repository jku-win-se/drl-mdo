"""
Balanced mini-batch sampling utilities for ensuring equal representation
from each environment during PPO training.
"""

import numpy as np


class BalancedMiniBatchSampler:
    """
    Creates mini-batches that ensure equal representation from each environment.
    Recycles samples from environments with fewer samples to maintain balance.
    """

    def __init__(self, num_envs):
        """
        Args:
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.samples_per_env = None
        self.env_sample_indices = None

    def has_tracking_info(self):
        """
        Check if tracking information for samples per environment is available.

        Returns:
            True if tracking info is set, False otherwise.
        """
        return self.samples_per_env is not None and self.env_sample_indices is not None

    def set_environment_tracking(self, samples_per_env, env_sample_indices):
        """
        Set the tracking information for samples per environment.

        Args:
            samples_per_env: List of sample counts per environment
            env_sample_indices: List of sample indices per environment
        """
        self.samples_per_env = samples_per_env
        self.env_sample_indices = env_sample_indices

    def create_minibatch_indices(self, batch_size, mini_batch_size):
        """
        Create mini-batch indices that ensure equal representation from each environment.

        Args:
            batch_size: Total number of samples in the batch
            mini_batch_size: Size of each mini-batch
            strategy: "recycle" (original) or "proportional" (proportional sampling)

        Returns:
            List of mini-batch indices where each mini-batch has balanced environment representation
        """

        return self._create_recycling_minibatch_indices(batch_size, mini_batch_size)

    def _create_recycling_minibatch_indices(self, batch_size, mini_batch_size):
        """
        Create mini-batches using recycling (original method).
        """

        max_samples = max(self.samples_per_env) if self.samples_per_env else 0
        if max_samples == 0:
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            return [
                indices[i : i + mini_batch_size]
                for i in range(0, batch_size, mini_batch_size)
            ]

        # Create balanced sample pool by recycling samples from environments with fewer samples
        balanced_indices = []

        for env_idx in range(self.num_envs):
            env_indices = self.env_sample_indices[env_idx].copy()
            if len(env_indices) == 0:
                continue

            while len(env_indices) < max_samples:
                recycled_count = min(
                    len(self.env_sample_indices[env_idx]),
                    max_samples - len(env_indices),
                )
                env_indices.extend(self.env_sample_indices[env_idx][:recycled_count])

            np.random.shuffle(env_indices)
            balanced_indices.extend(
                env_indices[:max_samples]
            )  # Take exactly max_samples

        np.random.shuffle(balanced_indices)

        # Ensure we don't exceed the original batch size
        mini_batches = []
        for i in range(0, len(balanced_indices), mini_batch_size):
            mini_batch = balanced_indices[i : i + mini_batch_size]

            mini_batch = [idx for idx in mini_batch if idx < batch_size]
            if len(mini_batch) > 0:
                mini_batches.append(mini_batch)

        return mini_batches
