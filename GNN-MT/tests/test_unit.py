"""Tests without a server. Run from GNN-MT/: python -m unittest tests.test_unit"""

import os
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "scripts"))

import analysis_common as common  # noqa: E402
from drl.main.ppo import PPO  # noqa: E402
from drl.neural_model.early_stopping import EarlyStopping  # noqa: E402
from drl.utils.episode_stats import EpisodeStats  # noqa: E402
from drl.utils.running_mean_std import RunningMeanStd  # noqa: E402


class RewardNormalizationTests(unittest.TestCase):
    def make_ppo(self, num_models=3, num_envs=1):
        ppo = PPO.__new__(PPO)
        ppo.normalize_rewards = True
        ppo.gamma = 0.99
        ppo.reward_clip = 10.0
        ppo.rew_norm_freeze_after = -1
        ppo.returns = np.zeros(num_envs)
        ppo._instance_reward_normalizers = [RunningMeanStd() for _ in range(num_models)]
        ppo._pooled_reward_normalizer = RunningMeanStd(epsilon=1.0)
        ppo._model_warmed_up = np.zeros(num_models, dtype=bool)
        ppo._model_norm_updates = np.zeros(num_models, dtype=np.int64)
        ppo._rewards_scaled = ppo._rewards_clipped = ppo._steps_seen = ppo._steps_invalid = 0
        ppo.env = types.SimpleNamespace(step_models=[0] * num_envs, instance_ids=[0] * num_envs)
        return ppo

    def test_no_clipping_from_variance_startup(self):
        ppo = self.make_ppo()
        scaled = [ppo._normalize_reward(0, 1.0) for _ in range(5)]
        self.assertTrue(all(abs(s) < 10.0 for s in scaled), scaled)
        self.assertEqual(ppo.take_reward_stats()["clipped_reward_share"], 0.0)

    def test_first_visit_uses_pooled_scale_then_its_own(self):
        ppo = self.make_ppo()
        for _ in range(50):  # Model 0 has large rewards.
            ppo._normalize_reward(0, 100.0)
        ppo.returns[0] = 0.0
        ppo.env.step_models[0] = 1
        pooled_std = np.sqrt(ppo._pooled_reward_normalizer.var)
        first = ppo._normalize_reward(0, 1.0)
        self.assertAlmostEqual(first, 1.0 / pooled_std, delta=1e-2 / pooled_std + 1e-3)

        for _ in range(50):
            ppo._normalize_reward(0, 1.0)
        ppo._end_of_episode(0)
        after = ppo._normalize_reward(0, 1.0)
        own_std = np.sqrt(ppo._instance_reward_normalizers[1].var)
        self.assertAlmostEqual(after, 1.0 / own_std, places=6)

    def test_each_reward_updates_the_model_it_came_from(self):
        ppo = self.make_ppo(num_models=4)
        ppo.env.step_models[0] = 2
        ppo._normalize_reward(0, 5.0)
        self.assertEqual(ppo._model_norm_updates.tolist(), [0, 0, 1, 0])

    def test_last_reward_of_episode_uses_that_episodes_model(self):
        # The row with the last reward also moved the env on: its instance id names the next model.
        ppo = self.make_ppo(num_models=4)
        ppo.env.step_models[0] = 1
        ppo.env.instance_ids[0] = 3
        ppo._normalize_reward(0, 5.0)
        self.assertEqual(ppo._model_norm_updates.tolist(), [0, 1, 0, 0])

    def test_freeze_counts_per_model(self):
        ppo = self.make_ppo()
        ppo.rew_norm_freeze_after = 2
        for _ in range(5):
            ppo._normalize_reward(0, 1.0)
        ppo.env.step_models[0] = 1
        ppo._normalize_reward(0, 1.0)
        self.assertEqual(ppo._model_norm_updates.tolist(), [2, 1, 0])

    def test_pooled_model_count_is_reported(self):
        ppo = self.make_ppo()
        ppo._end_of_episode(0)
        self.assertEqual(ppo.take_reward_stats()["models_on_pooled_scale"], 2)


class WindowStatsTests(unittest.TestCase):
    def test_window_return_stats_use_the_last_returns_of_every_model(self):
        stats = EpisodeStats(buffer_size=10, num_envs=2, instance_ids=[0, 1], instance_names=["a", "b"])
        for ret in (100.0, 1.0, 3.0):
            stats.add_episode(ret, 5, env_id=0, instance_id=0)
        stats.add_episode(5.0, 5, env_id=1, instance_id=1)
        mean, std = stats.get_window_return_stats(2)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, np.std([1.0, 3.0, 5.0]))


class BestPolicyTests(unittest.TestCase):
    def make(self, **kwargs):
        stopping = EarlyStopping.__new__(EarlyStopping)
        stopping.min_delta = kwargs.get("min_delta", 0.0)
        stopping.min_delta_rel = kwargs.get("min_delta_rel", 0.0)
        stopping.patience = kwargs.get("patience", 0)
        stopping.best_reward = None
        stopping.num_bad_updates = 0
        stopping.min_timesteps = 0
        stopping._save_checkpoint = lambda *_: None
        return stopping

    def test_patience_zero_tracks_the_best_and_never_stops(self):
        stopping = self.make()
        outcomes = [stopping.step(metric, 0, 10**6) for metric in [1.0, 0.5, 0.5, 1.0001, 0.2]]
        self.assertEqual([improved for _, improved in outcomes], [True, False, False, True, False])
        self.assertFalse(any(stop for stop, _ in outcomes))

    def test_patience_still_stops(self):
        stopping = self.make(patience=1)
        self.assertEqual(stopping.step(1.0, 0, 0), (False, True))
        self.assertEqual(stopping.step(0.5, 0, 0), (True, False))


class AnalysisTests(unittest.TestCase):
    def test_hv_on_known_fronts(self):
        self.assertAlmostEqual(common.hypervolume(np.array([[0.0, 0.0]])), 1.21)
        self.assertAlmostEqual(common.hypervolume(np.array([[0.0, 1.0], [1.0, 0.0]])), 0.21)
        self.assertAlmostEqual(common.hypervolume(np.array([[0.0, 0.0, 0.0]])), 1.331)

    def test_bounds_normalize_per_model_and_handle_flat_objectives(self):
        bounds = common.Bounds()
        bounds.add("m", np.array([[0.0, 5.0], [10.0, 5.0]]))
        bounds.add("n", np.array([[100.0, 1.0]]))
        np.testing.assert_allclose(bounds.normalize("m", np.array([[5.0, 5.0]])), [[0.5, 0.0]])

    def test_a12_with_ties(self):
        self.assertEqual(common.a12([2, 2], [1, 1]), 1.0)
        self.assertEqual(common.a12([1, 1], [2, 2]), 0.0)
        self.assertEqual(common.a12([1.0], [1.0 + 1e-9]), 0.5)

    def test_holm_by_hand(self):
        np.testing.assert_allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_ranks_within_blocks(self):
        table = pd.DataFrame([
            {"setting": s, "model": m, "seed": 1, "hv": hv}
            for s, m, hv in [("a", "x", 0.9), ("b", "x", 0.5), ("a", "y", 0.2), ("b", "y", 0.4)]
        ])
        _, ranks = common.rank_within_blocks(table)
        self.assertEqual(ranks.mean().to_dict(), {"a": 1.5, "b": 1.5})

    def test_primary_objective(self):
        points = np.array([[-3.0, 1.0, 7.0], [-5.0, 2.0, 8.0]])
        self.assertEqual(common.best_primary("cra", points), 5.0)  # Stored negated, maximized.
        self.assertEqual(common.best_primary("slb", points), -5.0)
        self.assertEqual(common.best_primary("sp", points), -3.0)

    def test_curves_share_one_grid_and_hold_their_last_value(self):
        rows = [
            ("a", 1, np.array([10, 20]), np.array([0.5, 0.6])),
            ("a", 2, np.array([10, 100]), np.array([0.1, 0.9])),
        ]
        table = common.curves_at_grid(rows)
        grids = table.groupby("seed")["fe"].apply(list)
        self.assertEqual(grids[1], grids[2])
        last = table[table["fe"] == table["fe"].max()].set_index("seed")["value"]
        self.assertEqual(last.to_dict(), {1: 0.6, 2: 0.9})  # Seed 1 ended early and holds 0.6.
        curves = common.summarize_curves(table.dropna())
        self.assertEqual(curves["seeds"].iloc[-1], 2)

    def test_read_solutions_with_and_without_fe_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "f.txt")
            with open(path, "w") as f:
                f.write("\n100\n[1.0, 2.0]\n\n200\n[0.5, 2.0]\n[1.0, 1.5]\n")
            blocks = common.read_solutions(path)
            self.assertEqual([fe for fe, _ in blocks], [100, 200])
            self.assertEqual(blocks[1][1].shape, (2, 2))
            with open(path, "w") as f:
                f.write("[1.0, 2.0]\n")
            self.assertEqual(common.read_solutions(path)[0][0], 0)


if __name__ == "__main__":
    unittest.main()
