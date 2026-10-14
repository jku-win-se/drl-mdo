import numpy as np
import torch


class EarlyStopping(object):
    def __init__(
        self,
        optimizer,
        model,
        path,
        patience=5,
        min_delta=0.01,
        min_timesteps=0,
        min_delta_rel=0.01,
    ):
        """
        Early stopping for reinforcement learning based on mean reward improvement.

        Args:
            optimizer: The optimizer to save
            model: The model to save
            path: Path to save the checkpoint
            patience: Number of updates without reward improvement before stopping
            min_delta: Minimum absolute improvement in reward to be considered significant
            min_delta_rel: Minimum improvement as a fraction of the best reward so far
        """
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.min_delta = min_delta
        self.min_delta_rel = min_delta_rel
        self.patience = patience
        self.best_reward = None
        self.num_bad_updates = 0
        self.min_timesteps = min_timesteps

        if patience == 0:
            def never_stop(mean_reward, update_step, current_timesteps):
                return False, False

            self.step = never_stop

    def step(self, mean_reward, update_step, current_timesteps):
        """
        Check whether training should stop and whether this metric is a new best.

        Returns:
            (should_stop, improved)
        """
        if np.isnan(mean_reward):
            print("Stop: NaN reward encountered.")
            return True, False

        if self.best_reward is None:
            self.best_reward = mean_reward
            self._save_checkpoint(update_step, mean_reward)
            return False, True

        margin = max(self.min_delta, self.min_delta_rel * abs(self.best_reward))
        if mean_reward > self.best_reward + margin:
            self.num_bad_updates = 0
            self.best_reward = mean_reward
            self._save_checkpoint(update_step, mean_reward)
            return False, True

        self.num_bad_updates += 1
        if (
            current_timesteps >= self.min_timesteps
            and self.num_bad_updates >= self.patience
        ):
            print("Stop: No improvement in reward.")
            return True, False

        return False, False

    def _save_checkpoint(self, update_step, mean_reward):
        """Save model checkpoint with current state."""
        if isinstance(self.optimizer, dict):
            optimizer_state_dict = {
                k: o.state_dict() for k, o in self.optimizer.items()
            }
        else:
            optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                "update_step": update_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer_state_dict,
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
