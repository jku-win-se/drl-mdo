"""Tests against the real server, which must be built. Takes several minutes.

Run from GNN-MT/: python -m unittest tests.test_runs
Uses port 8000, like train_agent.sh, so no other run may be active.
"""

import csv
import filecmp
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import procbridge as pb
import torch

GNN_MT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(GNN_MT)
sys.path.insert(0, os.path.join(ROOT, "evaluation", "scripts"))

import analysis_common as common  # noqa: E402
from drl.env.batched_env import BatchedRemoteEnv  # noqa: E402
from drl.utils.pallete import Pallete  # noqa: E402
from main import CASE_FILES, Config  # noqa: E402

HP = (
    "hidden_dim=16 n_steps=25 mini_batch_size=32 n_updates_per_iteration=1 lr=3e-4 value_head=gaussian "
    "normalize_rewards=True time_feature=False target_kl=0 graph_encoder=sage node_decoder=rnn "
    "encoder_layers=1 es_patience=0 es_min_delta=0 es_min_delta_rel=0"
)
PREFIX = f"test-{os.getpid()}"
SERVER_PORT = 8310
WORK = tempfile.mkdtemp(prefix="drl-tests-")


def load_env():
    env = dict(os.environ)
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line.startswith("export "):
                key, value = line[len("export "):].split("=", 1)
                env[key] = value.strip('"')
    return env


ENV = load_env()
CLASSPATH = f"{ROOT}/server-executables/build/classes:{ROOT}/server-executables/build/lib/*"


def tearDownModule():
    for pattern in (
        f"GNN-MT/models/rl/*/{PREFIX}*",
        f"models/generated/*/{PREFIX}*",
        f"server-executables/output/nn_logs/*/{PREFIX}*",
        f"server-executables/output/nn_logs/*/generalization/{PREFIX}*",
        f"server-executables/logs/*{PREFIX}*",
    ):
        for path in glob.glob(os.path.join(ROOT, pattern)):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    shutil.rmtree(WORK, ignore_errors=True)


def train(case, name, *flags, runs=1, check=True):
    result = subprocess.run(
        ["bash", os.path.join(ROOT, "train_agent.sh"), "--case", case, "--runs", str(runs),
         "--experiment-name", f"{PREFIX}-{name}", "--partial_trajectories", *flags],
        capture_output=True, text=True, env=ENV, cwd=ROOT,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"train_agent failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return result


def base_flags(fes=200, envs=2, hp=HP):
    # Tick on every finished episode, so short runs have a training metric to track.
    hp = f"{hp} es_window=1 es_warmup_episodes=1 es_tick_episodes=1"
    return ["--fes", str(fes), "--n-envs", str(envs), "--env-assignment", "cycle", "--eval-k", "2",
            "--hyperparams", hp]


def run_dirs(case, name):
    return sorted(glob.glob(os.path.join(GNN_MT, "models", "rl", case, f"{PREFIX}-{name}", "*-*")))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def generate(case, out, seed=1, train_sizes="2,4", val=2, test=1, sizes=None):
    folder = {"scrum": "sp"}.get(case, case)
    sizes = sizes or {"cra": "16-20", "slb": "9-12", "scrum": "37-42"}[case]
    subprocess.run(
        [ENV["JAVA_BIN"], "-cp", CLASSPATH, "learnapi.ModelGenerator", "--case", case, "--sizes", sizes,
         "--train", train_sizes, "--val", str(val), "--test", str(test), "--seed", str(seed),
         "--exclude", os.path.join(ROOT, "models", "benchmark", folder), "--out", out],
        check=True, capture_output=True, cwd=ROOT,
    )
    return out


class GeneratorTests(unittest.TestCase):
    def test_splits_nesting_and_determinism(self):
        for case in ("cra", "slb", "scrum"):
            a = generate(case, f"{WORK}/gen/{case}-a", seed=7)
            b = generate(case, f"{WORK}/gen/{case}-b", seed=7)
            c = generate(case, f"{WORK}/gen/{case}-c", seed=8)
            self.assertEqual(read_csv(f"{a}/manifest.csv"), read_csv(f"{b}/manifest.csv"))
            self.assertNotEqual(read_csv(f"{a}/manifest.csv"), read_csv(f"{c}/manifest.csv"))
            rows = read_csv(f"{a}/manifest.csv")
            self.assertEqual(len({row["hash"] for row in rows}), len(rows))  # Disjoint splits.
            self.assertEqual(sorted({row["split"] for row in rows}), ["test", "train", "val"])
            small, large = (set(os.listdir(f"{a}/train-{n}")) for n in (2, 4))
            self.assertEqual((len(small), len(large)), (2, 4))
            self.assertTrue(small < large)
            self.assertTrue(os.path.islink(f"{a}/train-2/{sorted(small)[0]}"))

    def test_sizes_stay_in_range(self):
        out = generate("slb", f"{WORK}/gen/range", train_sizes="20", val=5, test=5, sizes="9-11")
        sizes = {int(row["size"]) for row in read_csv(f"{out}/manifest.csv")}
        self.assertTrue(sizes <= {9, 10, 11}, sizes)


class ServerTests(unittest.TestCase):
    """Drive the env directly, to compare the FE counter and cycling with an independent count."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate("slb", f"{WORK}/gen/server", seed=3, train_sizes="2,3,4")
        case_config, metamodel, root_object = CASE_FILES["slb"]
        with open(os.path.join(GNN_MT, case_config)) as f:
            conf = json.loads(f.read(), object_hook=lambda d: Config(**d) if "maxNodeParams" in d else d)
        cls.pallete = Pallete(os.path.join(GNN_MT, metamodel), root_object, conf=conf)

    def start(self, envs, train, assignment, masking=True):
        self.pallete.masking = masking
        log = open(f"{WORK}/server.log", "w")
        process = subprocess.Popen(
            [ENV["JAVA_BIN"], "-cp", CLASSPATH, "learnapi.MultiModelEnvServer", "--case", "slb", "--n_envs", str(envs),
             "--models_path", train, "--env_assignment", assignment, "--out_path", f"slb/{PREFIX}-server",
             "--port", str(SERVER_PORT)],
            stdout=log, stderr=subprocess.STDOUT, cwd=os.path.join(ROOT, "server-executables"),
        )
        self.addCleanup(process.wait)
        self.addCleanup(process.terminate)
        for _ in range(120):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", SERVER_PORT)) == 0:
                    break
            time.sleep(0.5)
        pb.Client("127.0.0.1", SERVER_PORT).request("resetInstance", "run")
        env = BatchedRemoteEnv(envs, self.pallete, port=SERVER_PORT, time_feature=False)
        self.addCleanup(env.close)
        return env

    def act(self, env, env_index, amount):
        """shiftLeft from Stack_1 to its left neighbour Stack_0; a huge amount cannot be applied."""
        rule_id = next(i for i in sorted(self.pallete.unitIdToRequiredNodeTypes)
                       if self.pallete.get_unitname_by_id(i).endswith("shiftLeft"))
        params = {
            self.pallete.get_name_by_param_pos(rule_id, 0): "Stack_1",
            self.pallete.get_name_by_param_pos(rule_id, 1): "Stack_0",
            self.pallete.get_name_by_param_pos(rule_id, self.pallete.max_len): amount,
        }
        return {"_envId": env_index, "_ruleName": self.pallete.get_unitname_by_id(rule_id), "_paramValues": params}

    def step(self, env, env_indices, amount):
        env.pending_env_indices[0] = list(env_indices)
        env.clients[0].send("stepAll", {"_autoReset": True,
                                        "_actions": [self.act(env, i, amount) for i in env_indices]})
        return env.step_wait_group(0)

    def test_fe_counts_steps_with_a_successor_only(self):
        env = self.start(2, f"{self.dataset}/train-2", "fixed")
        self.assertEqual(env.get_fe_count(), 0)
        expected = 0
        for amount in (1, 10**6, 1, 10**6, 1):
            obs, _, _ = self.step(env, [0, 1], amount)
            expected += sum(o is not None for o in obs)
            self.assertEqual(env.get_fe_count(), expected)
        self.assertGreater(expected, 0)
        env.reset()  # Evaluates the initial models again, without counting.
        self.assertEqual(env.get_fe_count(), expected)

    def test_masking_off_invalid_actions_cost_steps_but_no_fe(self):
        env = self.start(1, f"{self.dataset}/train-2/{sorted(os.listdir(self.dataset + '/train-2'))[0]}",
                         "fixed", masking=False)
        horizon = env.max_steps[0]
        for step in range(horizon):
            obs, rew, done = self.step(env, [0], 10**6)
            self.assertIsNotNone(obs[0])  # An ordinary transition, with the unchanged model.
            self.assertEqual(float(rew[0]), 0.0)
            self.assertEqual(bool(done[0]), step == horizon - 1)
            self.assertTrue(env.step_infos[0]["invalid"])
        self.assertEqual(env.get_fe_count(), 0)

    def play_episodes(self, env, env_index, episodes):
        """Models the env plays, in order; invalid steps are repeated until the episode ends."""
        models = [env.instance_ids[env_index]]
        while len(models) <= episodes:
            _, _, done = self.step(env, [env_index], 10**6)
            self.assertEqual(env.step_models[env_index], models[-1])
            if done[0]:
                models.append(env.instance_ids[env_index])
        return models

    def test_all_envs_can_be_pointed_at_one_model(self):
        env = self.start(2, f"{self.dataset}/train-4", "cycle", masking=False)
        for model in (2, 0, 3):
            env.reset_to_model(model)
            self.assertEqual(env.instance_ids, [model, model])
            self.assertEqual(self.play_episodes_no_cycle(env, model), model)

    def play_episodes_no_cycle(self, env, model):
        obs, _, _ = self.step(env, [0, 1], 10**6)
        self.assertEqual(env.step_models, [model, model])
        return env.step_models[0]

    def test_cycle_follows_the_formula_and_visits_every_model(self):
        env = self.start(2, f"{self.dataset}/train-4", "cycle", masking=False)
        self.assertEqual(len(env.instance_names), 4)
        self.assertEqual(self.play_episodes(env, 0, 4), [0, 2, 0, 2, 0])
        self.assertEqual(self.play_episodes(env, 1, 4), [1, 3, 1, 3, 1])

    def test_cycle_equals_fixed_when_models_divide_envs(self):
        env = self.start(2, f"{self.dataset}/train-2", "cycle", masking=False)
        self.assertEqual(self.play_episodes(env, 0, 3), [0, 0, 0, 0])
        self.assertEqual(self.play_episodes(env, 1, 3), [1, 1, 1, 1])

    def test_cycle_reaches_every_model_with_a_coprime_count(self):
        env = self.start(2, f"{self.dataset}/train-3", "cycle", masking=False)
        self.assertEqual(self.play_episodes(env, 0, 3), [0, 2, 1, 0])


class RunTests(unittest.TestCase):
    """Short training runs through train_agent.sh, sharing one dataset per experiment family."""

    @classmethod
    def setUpClass(cls):
        cls.gen = ["--gen-sizes", "9-12", "--gen-train", "2,4", "--train-size", "4", "--gen-test", "2",
                   "--gen-seed", "5", "--gen-name", f"{PREFIX}-shared"]
        cls.flags = base_flags(fes=300) + cls.gen + ["--final-eval", "test"]
        train("slb", "all", *cls.flags, "--keep-checkpoints", "all", runs=2)
        cls.runs = run_dirs("slb", "all")

    def seed42(self):
        return [r for r in self.runs if os.path.basename(r).startswith("42-")][0]

    def test_runs_use_the_first_stored_seeds_and_write_every_artifact(self):
        self.assertEqual(sorted(os.path.basename(r).split("-")[0] for r in self.runs), ["17", "42"])
        for run in self.runs:
            for name in ("best_model.pth", "checkpoints.csv", "train_log.jsonl", "run_args.json",
                         "runtimes.json", "pallete.json", "model_config.json"):
                self.assertTrue(os.path.exists(os.path.join(run, name)), name)
            self.assertFalse(os.path.exists(os.path.join(run, "val_returns.csv")))
            with open(os.path.join(run, "run_args.json")) as f:
                info = json.load(f)
            self.assertEqual(info["run_info"]["case"], "slb")
            self.assertEqual(sorted(info["run_info"]["datasets"]), ["test", "train"])
            self.assertIn("commit", info)
            self.assertNotIn("wandb_config", info)
            logs = os.path.join(ROOT, "server-executables", "output", "nn_logs", "slb", f"{PREFIX}-all",
                                os.path.basename(run), "running-solutions.txt")
            self.assertTrue(os.path.exists(logs), logs)

    def test_too_many_runs_are_refused(self):
        result = train("slb", "many", *self.flags, runs=31, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("more seeds", result.stderr)

    def test_budget_counts_server_fes(self):
        for run in self.runs:
            with open(os.path.join(run, "runtimes.json")) as f:
                times = json.load(f)
            self.assertGreaterEqual(times["fes"], 300)
            self.assertLess(times["fes"], 300 + 2 * 25)  # One rollout of overshoot at most.
            self.assertLessEqual(int(read_csv(os.path.join(run, "checkpoints.csv"))[-1]["fe"]), times["fes"])

    def test_checkpoint_log_marks_one_best_equal_to_best_model(self):
        for run in self.runs:
            rows = read_csv(os.path.join(run, "checkpoints.csv"))
            self.assertGreater(len(rows), 1)
            self.assertTrue(all(row["file"] for row in rows))  # keep_checkpoints all
            self.assertTrue(all(row["std_return"] != "" for row in rows))
            best = [row for row in rows if row["is_best"] == "True"]
            self.assertEqual(len(best), 1)
            self.assertEqual(float(best[0]["metric"]), max(float(r["metric"]) for r in rows))
            named = torch.load(os.path.join(run, best[0]["file"]), map_location="cpu")
            stored = torch.load(os.path.join(run, "best_model.pth"), map_location="cpu")["model_state_dict"]
            for key, value in named.items():
                self.assertTrue(torch.equal(value, stored[key]), key)

    def test_keep_best_still_writes_the_full_log(self):
        train("slb", "best", *self.flags)
        run = run_dirs("slb", "best")[0]
        rows = read_csv(os.path.join(run, "checkpoints.csv"))
        stored = [row["file"] for row in rows if row["file"]]
        self.assertGreater(len(rows), 1)
        self.assertEqual(sorted(stored), sorted(os.path.basename(p) for p in glob.glob(f"{run}/ac_*.pth")))
        self.assertEqual(sum(row["is_best"] == "True" for row in rows), 1)

    def test_final_run_is_one_batch_of_k_on_each_model(self):
        for run in self.runs:
            curves = common.final_curves(common.Run(run, "s", 0, "slb"), "test")
            self.assertEqual(len(curves), 2)
            with open(os.path.join(run, "runtimes.json")) as f:
                times = json.load(f)["final_eval"]["test"]
            for model, blocks in curves.items():
                self.assertEqual(len(blocks), 1)
                self.assertEqual(times[model]["episodes"], 2)
                self.assertEqual(times[model]["fes"], blocks[0][0])
                self.assertEqual(blocks[0][0], 2 * int(int(model.split("n")[1]) * 2.5))
                self.assertTrue((blocks[0][1][:, -1] > 0).all())  # No length-0 states.

    def test_evaluation_server_allocates_k_envs_in_total(self):
        with open(os.path.join(ROOT, "server-executables", "server_output.log")) as f:
            self.assertTrue(any("'test'" in line and "2 envs" in line for line in f))

    def test_same_flags_repeat_and_test_data_has_no_influence(self):
        # Same training data with a different test set: the selection must not change.
        train("slb", "again", *self.flags)
        other_test = base_flags(fes=300) + self.gen[:-4] + ["--gen-seed", "5", "--gen-name", f"{PREFIX}-shared",
                                                            "--test-path", "models/benchmark/slb", "--final-eval", "test"]
        train("slb", "other-test", *other_test)
        for name in ("again", "other-test"):
            self.assertEqual(self.selection(run_dirs("slb", name)[0]), self.selection(self.seed42()), name)

    @staticmethod
    def selection(run):
        columns = ("update", "fe", "metric", "mean_return", "std_return", "is_best")
        return [tuple(row[c] for c in columns) for row in read_csv(os.path.join(run, "checkpoints.csv"))]

    def test_validation_records_without_changing_training(self):
        dataset = os.path.join(ROOT, "models", "generated", "slb", f"{PREFIX}-shared", "seed-5", "test")
        train("slb", "validated", *self.flags, "--val-path", dataset, "--test-path", "models/benchmark/slb",
              "--eval-interval", "2", "--eval-save-solutions")
        run = run_dirs("slb", "validated")[0]
        rows = read_csv(os.path.join(run, "val_returns.csv"))
        self.assertEqual({row["episodes"] for row in rows}, {"2"})
        self.assertEqual(len({row["model"] for row in rows}), 2)
        stored = glob.glob(os.path.join(run, "eval_solutions", "stochastic", "*", "iter*", "nn-solutions.txt"))
        self.assertGreater(len(stored), 0)
        self.assertEqual(self.selection(run), self.selection(self.seed42()))

    def test_convergence_stops_training_early(self):
        hp = HP.replace("es_patience=0", "es_patience=2") + " es_min_timesteps=0 es_window=1 es_warmup_episodes=1 es_tick_episodes=1"
        flags = ["--fes", "20000", "--n-envs", "2", "--env-assignment", "cycle", "--eval-k", "2", "--hyperparams", hp]
        train("slb", "converge", *flags, *self.gen, "--final-eval", "test")
        run = run_dirs("slb", "converge")[0]
        with open(os.path.join(run, "runtimes.json")) as f:
            times = json.load(f)
        self.assertLess(times["fes"], 20000)
        self.assertTrue(os.path.isdir(os.path.join(run, "final_eval", "test")))

    def test_analysis_scripts_write_their_files(self):
        experiment = os.path.dirname(self.runs[0])
        out = os.path.join(WORK, "analysis")
        for script, args in (
            ("rank_configs.py", ["--dataset", "test"]),
            ("compare_hv.py", ["--dataset", "test"]),
            ("reward_over_fes.py", []),
            ("runtimes.py", []),
        ):
            result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "scripts", script),
                                     "--runs", experiment, "--out", out, *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        for name in ("ranks.csv", "ranks.tex", "hv_test.csv", "hv_test.tex", "reward_over_fes.pdf", "runtimes.csv"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)

    def test_sampling_curve_logs_each_batch_and_grows(self):
        result = subprocess.run(
            ["bash", os.path.join(ROOT, "run_policy.sh"), "--case", "slb", "--mode", "batches", "--n-batches", "3",
             "--n-envs", "2", "--label", "heldout", "--checkpoint", os.path.join(self.seed42(), "best_model.pth"),
             "--models", f"models/generated/slb/{PREFIX}-shared/seed-5/test"],
            capture_output=True, text=True, env=ENV, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-1500:] + result.stdout[-1500:])
        curves = common.final_curves(common.Run(self.seed42(), "s", 42, "slb"), "heldout", "sampling_curve")
        self.assertEqual(len(curves), 2)
        for model, blocks in curves.items():
            self.assertEqual(len(blocks), 3)
            bounds = common.Bounds()
            for _, points in blocks:
                bounds.add(model, points)
            values = [bounds.hv(model, points) for _, points in blocks]
            self.assertEqual(values, sorted(values))
        out = os.path.join(WORK, "analysis-sampling")
        result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "scripts", "hv_over_fes.py"), "--runs",
                                 os.path.dirname(self.seed42()), "--source", "sampling", "--dataset", "heldout",
                                 "--out", out], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertTrue(os.path.exists(os.path.join(out, "hv_over_fes_sampling.pdf")))

    def test_second_launch_is_refused(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 8000))
            listener.listen()
            result = train("slb", "second", *self.flags, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another run", result.stderr)

    def test_finished_seeds_are_skipped(self):
        before = run_dirs("slb", "all")
        result = train("slb", "all", *self.flags, "--keep-checkpoints", "all", runs=2)
        self.assertEqual(result.stdout.count("already complete"), 2)
        self.assertEqual(run_dirs("slb", "all"), before)

    def test_generated_dataset_is_reused(self):
        manifest = os.path.join(ROOT, "models", "generated", "slb", f"{PREFIX}-shared", "seed-5", "manifest.csv")
        before = os.stat(manifest).st_mtime_ns
        train("slb", "reuse", *self.flags)
        self.assertEqual(os.stat(manifest).st_mtime_ns, before)

    def test_test_path_equal_to_train_needs_an_explicit_flag(self):
        model = "models/benchmark/slb/model_ten_stacks.xmi"
        flags = base_flags(fes=50, envs=1) + ["--train-path", model, "--test-path", model]
        result = train("slb", "same", *flags, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must differ", result.stderr)
        train("slb", "same", *flags, "--allow-train-as-test", "--final-eval", "test")
        self.assertTrue(os.path.isdir(os.path.join(run_dirs("slb", "same")[0], "final_eval", "test", "model_ten_stacks")))


class PerSeedDatasetTests(unittest.TestCase):
    def test_each_seed_gets_its_own_dataset_named_after_it(self):
        train("slb", "perseed", *base_flags(fes=50), "--gen-sizes", "9-10", "--gen-train", "2", runs=2)
        root = os.path.join(ROOT, "models", "generated", "slb", f"{PREFIX}-perseed")
        self.assertEqual(sorted(os.listdir(root)), ["seed-17", "seed-42"])
        self.assertNotEqual(read_csv(f"{root}/seed-17/manifest.csv"), read_csv(f"{root}/seed-42/manifest.csv"))
        for run in run_dirs("slb", "perseed"):
            seed = os.path.basename(run).split("-")[0]
            with open(os.path.join(run, "run_args.json")) as f:
                self.assertIn(f"seed-{seed}", json.load(f)["run_info"]["datasets"]["train"])


class AllCasesTests(unittest.TestCase):
    def policy(self, case, sizes, hidden_dim):
        hp = HP.replace("hidden_dim=16", f"hidden_dim={hidden_dim}")
        train(case, "policy", *base_flags(fes=250, hp=hp), "--gen-sizes", sizes, "--gen-train", "2")
        return run_dirs(case, "policy")[0]

    def run_policy(self, case, checkpoint, models, *flags):
        result = subprocess.run(
            ["bash", os.path.join(ROOT, "run_policy.sh"), "--case", case, "--checkpoint", checkpoint,
             "--models", models, *flags],
            capture_output=True, text=True, env=ENV, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-1500:] + result.stdout[-1500:])

    def test_train_and_run_policy_in_every_case(self):
        for case, sizes, hidden_dim, benchmark in (
            ("cra", "16-19", 32, "CRA_18.xmi"),
            ("slb", "9-11", 16, "model_ten_stacks.xmi"),
            ("sp", "37-40", 32, "sprint-planning-model-4-stakeholders-42-items.xmi"),
        ):
            run = self.policy(case, sizes, hidden_dim)
            model = os.path.join("models", "benchmark", case, benchmark)
            # Network shape and case come from the checkpoint and the server, not from flags.
            for flags in (("--mode", "one-per-model"), ("--mode", "fixed-count", "--n-solutions", "2", "--n-envs", "2")):
                self.run_policy(case, os.path.join(run, "best_model.pth"), model, *flags)
            sampled = glob.glob(os.path.join(os.path.dirname(run), "..", "**", "nn-solutions.txt"), recursive=True)
            self.assertTrue(sampled, case)


if __name__ == "__main__":
    unittest.main()
