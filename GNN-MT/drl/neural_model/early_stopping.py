import numpy as np
import torch


class EarlyStopping(object):
    def __init__(
        self, optimizer, model, path, patience=5, min_delta=0.01, min_timesteps=0
    ):
        """
        Early stopping for reinforcement learning based on mean reward improvement.

        Args:
            optimizer: The optimizer to save
            model: The model to save
            path: Path to save the checkpoint
            patience: Number of updates without reward improvement before stopping
            min_delta: Minimum improvement in reward to be considered significant
        """
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.min_delta = min_delta
        self.patience = patience
        self.best_reward = None
        self.num_bad_updates = 0
        self.min_timesteps = min_timesteps

        if patience == 0:
            self.step = lambda a, b: False

    def step(self, mean_reward, update_step, current_timesteps):
        """
        Check if training should stop based on mean reward improvement.

        Args:
            mean_reward: Mean reward from the most recent rollout
            update_step: Current update step number

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Handle NaN
        if np.isnan(mean_reward):
            print("Stop: NaN reward encountered.")
            return True

        # First update
        if self.best_reward is None:
            self.best_reward = mean_reward
            self._save_checkpoint(update_step, mean_reward)
            return False

        # Check if current reward is better than best (higher reward is better)
        if mean_reward > self.best_reward + self.min_delta:
            # Reward improved
            self.num_bad_updates = 0
            self.best_reward = mean_reward
            self._save_checkpoint(update_step, mean_reward)
        else:
            # No improvement
            self.num_bad_updates += 1

        # Check if patience exceeded
        if (
            current_timesteps >= self.min_timesteps
            and self.num_bad_updates >= self.patience
        ):
            print("Stop: No improvement in reward.")
            return True

        return False

    def _save_checkpoint(self, update_step, mean_reward):
        """Save model checkpoint with current state."""
        torch.save(
            {
                "update_step": update_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "mean_reward": mean_reward,
            },
            self.path,
        )

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
