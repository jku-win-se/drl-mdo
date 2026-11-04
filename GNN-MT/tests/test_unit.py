"""Tests without a server. Run from GNN-MT/: python -m unittest tests.test_unit"""

import dataclasses
import glob
import os
import re
import subprocess
import sys
import types
import warnings
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "scripts"))

import analysis_common as common  # noqa: E402
from drl.main.hyperparams import Hyperparams  # noqa: E402
from drl.main.inference import (  # noqa: E402
    FixedEpisodeCount,
    OneEpisodePerEnv,
    ParetoArchive,
    UntilNoImprovement,
    isolated_rng,
)
from drl.main.reward_norm import RewardNormalizer  # noqa: E402
from drl.main.rollout import (  # noqa: E402
    RolloutCollector,
    _Rollout,
    compute_gae,
    resolve_rollout_length,
)
from drl.neural_model.early_stopping import EarlyStopping  # noqa: E402
from drl.utils.episode_stats import EpisodeStats  # noqa: E402


class RewardNormalizationTests(unittest.TestCase):
    def make(self, num_models=3, freeze_after=-1):
        return RewardNormalizer(1, num_models, gamma=0.99, reward_clip=10.0, freeze_after=freeze_after)

    def test_no_clipping_from_variance_startup(self):
        norm = self.make()
        scaled = [norm.normalize(0, 0, 1.0) for _ in range(5)]
        self.assertTrue(all(abs(s) < 10.0 for s in scaled), scaled)
        self.assertEqual(norm.take_stats()["clipped_reward_share"], 0.0)

    def test_first_visit_uses_pooled_scale_then_its_own(self):
        norm = self.make()
        for _ in range(50):  # Model 0 has large rewards.
            norm.normalize(0, 0, 100.0)
        norm.reset_return(0)
        pooled_std = np.sqrt(norm.pooled.var)
        first = norm.normalize(0, 1, 1.0)
        self.assertAlmostEqual(first, 1.0 / pooled_std, delta=1e-2 / pooled_std + 1e-3)

        for _ in range(50):
            norm.normalize(0, 1, 1.0)
        norm.end_of_episode(1)
        after = norm.normalize(0, 1, 1.0)
        self.assertAlmostEqual(after, 1.0 / np.sqrt(norm.normalizers[1].var), places=6)

    def test_each_reward_updates_the_model_it_came_from(self):
        norm = self.make(num_models=4)
        norm.normalize(0, 2, 5.0)
        self.assertEqual(norm.updates.tolist(), [0, 0, 1, 0])

    def test_freeze_counts_per_model(self):
        norm = self.make(freeze_after=2)
        for _ in range(5):
            norm.normalize(0, 0, 1.0)
        norm.normalize(0, 1, 1.0)
        self.assertEqual(norm.updates.tolist(), [2, 1, 0])

    def test_pooled_model_count_is_reported(self):
        norm = self.make()
        norm.end_of_episode(0)
        self.assertEqual(norm.take_stats()["models_on_pooled_scale"], 2)

    def test_disabled_passes_rewards_through(self):
        norm = RewardNormalizer(1, 1, gamma=0.99, reward_clip=10.0, enabled=False)
        self.assertEqual(norm.normalize(0, 0, 123.0), 123.0)


class RolloutTests(unittest.TestCase):
    def collector(self, step_model=1, next_model=3):
        env = types.SimpleNamespace(
            reset=lambda: [None], num_envs=1, step_models=[step_model], instance_ids=[next_model], step_infos=[None]
        )
        stats = types.SimpleNamespace(
            add_step_reward=lambda _: None, add_episode=lambda *_, **__: None
        )
        norm = RewardNormalizer(1, 4, gamma=0.99, reward_clip=10.0)
        return RolloutCollector(env, None, None, stats, norm, 0.99, 0.95, 1, 1, "cpu"), norm

    def record(self, collector, rollout, done, next_obs="next", action=("rule", None, None)):
        collector.record_transition(0, "staged", action, [0.0, 0.0, 0.0], 0.5, 5.0, done, next_obs, rollout)

    def test_last_reward_of_episode_belongs_to_that_episodes_model(self):
        # The row with the last reward also moved the env on: its instance id names the next model.
        collector, norm = self.collector()
        rollout = _Rollout(1, False, types.SimpleNamespace(update=lambda _: None))
        self.record(collector, rollout, True)
        self.assertEqual(norm.updates.tolist(), [0, 1, 0, 0])
        self.assertEqual(norm.warmed_up.tolist(), [False, True, False, False])
        self.assertEqual(rollout.envs[0].models, [1])
        self.assertEqual(rollout.ep_lens, [1])

    def test_full_episode_rollout_keeps_only_finished_episodes(self):
        collector, _ = self.collector()
        rollout = _Rollout(1, True, types.SimpleNamespace(update=lambda _: None))
        self.record(collector, rollout, False)
        self.assertEqual(rollout.completed.obs, [])
        self.record(collector, rollout, True)
        self.assertEqual(rollout.completed.obs, ["staged", "staged"])
        self.assertEqual(rollout.envs[0].obs, [])

    def test_skipped_step_stores_nothing_and_keeps_the_observation(self):
        collector, _ = self.collector()
        rollout = _Rollout(1, False, types.SimpleNamespace(update=lambda _: None))
        self.record(collector, rollout, False, next_obs=None)
        self.assertEqual(rollout.envs[0].obs, [])
        self.assertEqual(rollout.skipped_invalid_steps, 1)
        self.assertEqual(collector.obs, [None])

    def test_invalid_node_action_counts_but_is_not_stored(self):
        collector, _ = self.collector()
        rollout = _Rollout(1, False, types.SimpleNamespace(update=lambda _: None))
        self.record(collector, rollout, False, action=("rule", torch.tensor([[-1]]), None))
        self.assertEqual(rollout.envs[0].obs, [])
        self.assertEqual(rollout.skipped_invalid_nodes, 1)
        self.assertEqual(collector.obs, ["next"])

    def test_invalid_action_share_resets(self):
        collector, _ = self.collector()
        collector.env.step_infos[0] = {"invalid": True}
        rollout = _Rollout(1, False, types.SimpleNamespace(update=lambda _: None))
        self.record(collector, rollout, False, next_obs=None)
        self.assertEqual(collector.take_step_stats()["invalid_action_share"], 1.0)
        self.assertEqual(collector.take_step_stats()["invalid_action_share"], 0.0)

    def test_gae_by_hand(self):
        # One env, two steps, no terminal: gamma 0.5, lambda 1, next value 4.
        returns, advantages = compute_gae([[1.0, 2.0]], [[0.0, 0.0]], [[1.0, 1.0]], [4.0], 0.5, 1.0, "cpu")
        # t=1: 2 + 0.5*4 - 1 = 3. t=0: 1 + 0.5*1 - 1 + 0.5*3 = 2.
        self.assertEqual(advantages.tolist(), [2.0, 3.0])
        self.assertEqual(returns.tolist(), [3.0, 4.0])

    def test_gae_stops_at_a_finished_episode(self):
        _, advantages = compute_gae([[1.0, 2.0]], [[1.0, 0.0]], [[0.0, 0.0]], [0.0], 0.5, 1.0, "cpu")
        self.assertEqual(advantages.tolist(), [1.0, 2.0])

    def test_rollout_length(self):
        self.assertEqual(resolve_rollout_length(Hyperparams(), 4, 30), (30, 120))
        self.assertEqual(resolve_rollout_length(Hyperparams(timesteps_per_batch=100), 4, 25), (25, 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(resolve_rollout_length(Hyperparams(n_steps=10), 2, 30), (10, 20))


class InferenceTests(unittest.TestCase):
    def test_running_a_policy_does_not_need_the_training_code(self):
        code = (
            "import sys; import drl.main.inference; "
            "assert 'drl.main.ppo' not in sys.modules and 'drl.main.rollout' not in sys.modules"
        )
        root = os.path.join(os.path.dirname(__file__), "..")
        subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True, cwd=root)

    def test_archive_only_reports_changes_of_the_front(self):
        archive = ParetoArchive()
        self.assertTrue(archive.add([[1.0, 2.0]]))
        self.assertFalse(archive.add([[1.0, 2.0], [2.0, 3.0]]))  # A repeat and a dominated point.
        self.assertTrue(archive.add([[0.5, 3.0]]))
        self.assertEqual(len(archive), 2)

    def test_schedules(self):
        once = OneEpisodePerEnv(2)
        self.assertEqual([once.start_episode(0), once.start_episode(0)], [True, False])
        fixed = FixedEpisodeCount(1)
        self.assertEqual([fixed.start_episode(0), fixed.start_episode(1)], [True, False])
        patient = UntilNoImprovement(2)
        patient.on_episode_end(False)
        patient.on_episode_end(True)
        patient.on_episode_end(False)
        self.assertFalse(patient.stopped)
        patient.on_episode_end(False)
        self.assertTrue(patient.stopped)

    def test_isolated_rng_repeats_and_leaves_the_outer_stream_alone(self):
        torch.manual_seed(1)
        expected = torch.rand(2)
        torch.manual_seed(1)
        with isolated_rng(7):
            first = torch.rand(1)
        with isolated_rng(7):
            second = torch.rand(1)
        self.assertEqual(first.item(), second.item())
        self.assertTrue(torch.equal(torch.rand(2), expected))


class HyperparamsTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "es_patienc"):
            Hyperparams.from_dict({"es_patienc": 15})

    def test_values_parse_as_exec_did(self):
        hp = Hyperparams.from_dict(
            {"masking": "False", "time_scale": "None", "target_kl": "None", "lr": 1e-3,
             "graph_encoder": "gatv2", "n_steps": "auto", "entropy_normalize": "yes", "value_ordinal": 0}
        )
        self.assertIs(hp.masking, False)
        self.assertIsNone(hp.time_scale)
        self.assertEqual(hp.target_kl, 0.0)
        self.assertEqual(hp.lr, 1e-3)
        self.assertEqual(hp.graph_encoder, "gatv2")
        self.assertEqual(hp.n_steps, "auto")
        self.assertIs(hp.entropy_normalize, True)
        self.assertIs(hp.value_ordinal, False)
        self.assertEqual(Hyperparams.from_dict({"n_steps": 100}).n_steps, 100)

    def test_entropy_coefficients_default_to_the_shared_one(self):
        hp = Hyperparams.from_dict({"entropy_coef": 0.5, "entropy_coef_node": 0.1})
        self.assertEqual((hp.entropy_coef_rule, hp.entropy_coef_node, hp.entropy_coef_val), (0.5, 0.1, 0.5))

    def test_unsupported_choice_is_rejected(self):
        for key in ("encoder", "value_head", "node_decoder", "graph_encoder"):
            with self.assertRaisesRegex(ValueError, key):
                Hyperparams.from_dict({key: "nope"})

    def test_every_key_in_the_experiment_scripts_is_known(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        known = {field.name for field in dataclasses.fields(Hyperparams)} | {"hidden_dim"}
        pattern = re.compile(r"\b([a-z_0-9]+)=[^\s\"']+")
        paths = glob.glob(os.path.join(root, "experiments", "*.sh")) + [
            os.path.join(root, "server-executables", "py_train_scripts.sh")
        ]
        for path in paths:
            with open(path) as f:
                # Only hyperparameter strings: lines inside BASE_HP, or passed to rq1/--hyperparams.
                for line in f:
                    if "--hyperparams" in line or line.startswith(("BASE_HP", "entropy_coef", "target_kl")) or re.match(r"\s*(N=\S+ )?rq1 ", line):
                        keys = set(pattern.findall(line)) - {"BASE_HP", "N", "FES"}
                        self.assertEqual(keys - known, set(), f"{path}: {line}")


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
