"""Reward scaling per model, for training on models of very different return scales."""

import math

import numpy as np

from drl.utils.running_mean_std import RunningMeanStd

REWARD_NORM_EPS = 1e-8


def scale_reward(
    normalizer, discounted_return, raw_reward, reward_clip, update_stats, scale_from=None
):
    """Divide a reward by the instance return scale, in SB3 VecNormalize order.

    The running variance is updated with the discounted return first. The raw
    reward is then divided by sqrt(variance + eps) and clipped. It is not
    mean-centered. ``scale_from`` takes the variance from another normalizer,
    while ``normalizer`` still updates.
    """
    if update_stats:
        normalizer.update(np.array([discounted_return], dtype=np.float64))
    source = normalizer if scale_from is None else scale_from
    variance = float(np.asarray(source.var, dtype=np.float64).reshape(-1)[0])
    scaled = float(raw_reward) / math.sqrt(variance + REWARD_NORM_EPS)
    if scaled > reward_clip:
        return float(reward_clip)
    if scaled < -reward_clip:
        return float(-reward_clip)
    return scaled


class RewardNormalizer:
    """One return scale per model, and a pooled one until a model finished its first episode.

    The discounted return is tracked per env, like OpenAI Baselines, and must be reset with
    ``reset_return`` when an env's episode ends. With ``enabled`` off, rewards pass through.
    """

    def __init__(self, num_envs, num_models, gamma, reward_clip, freeze_after=-1, enabled=True):
        self.enabled = enabled
        self.gamma = gamma
        self.reward_clip = reward_clip
        self.freeze_after = freeze_after
        self.returns = np.zeros(num_envs)
        self.normalizers = [RunningMeanStd() for _ in range(num_models)]
        # The unit prior keeps this scale near 1 at the very start, where a tiny count would
        # inflate rewards.
        self.pooled = RunningMeanStd(epsilon=1.0)
        self.warmed_up = np.zeros(num_models, dtype=bool)
        self.updates = np.zeros(num_models, dtype=np.int64)
        self._scaled = 0
        self._clipped = 0

    def normalize(self, env_i, model, raw_rew):
        """Scale the reward an env just got for a step on ``model``."""
        if not self.enabled:
            return float(raw_rew)
        self.returns[env_i] = self.returns[env_i] * self.gamma + float(raw_rew)
        update_stats = self.freeze_after < 0 or self.updates[model] < self.freeze_after
        if update_stats:
            self.pooled.update(np.array([self.returns[env_i]], dtype=np.float64))
        scaled = scale_reward(
            self.normalizers[model],
            self.returns[env_i],
            raw_rew,
            self.reward_clip,
            update_stats,
            scale_from=None if self.warmed_up[model] else self.pooled,
        )
        if update_stats:
            self.updates[model] += 1
        self._scaled += 1
        self._clipped += int(abs(scaled) >= self.reward_clip)
        return scaled

    def end_of_episode(self, model):
        """From now on, ``model`` uses its own scale."""
        self.warmed_up[model] = True

    def reset_return(self, env_i):
        self.returns[env_i] = 0.0

    def take_stats(self):
        """Stats since the last call."""
        stats = {
            "models_on_pooled_scale": int((~self.warmed_up).sum()),
            "clipped_reward_share": self._clipped / max(self._scaled, 1),
        }
        self._scaled = 0
        self._clipped = 0
        return stats
