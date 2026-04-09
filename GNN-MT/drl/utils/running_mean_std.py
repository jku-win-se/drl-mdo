"""
Running mean and standard deviation tracker for reward normalization.
Based on OpenAI Baselines implementation.
"""

import numpy as np


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, update_stats=True):
        """
        Normalize values using current statistics (OpenAI Baselines style)
        Args:
            x: Values to normalize
            update_stats: Whether to update running statistics with these values
            clip_range: Range to clip normalized values
        Returns:
            Normalized and clipped values
        """
        if update_stats:
            self.update(x)
        normalized = x / np.sqrt(self.var + 1e-8)
        #  return np.clip(normalized, -10.0, 10.0)
        return normalized
