import glob
import logging
import os
import json
from types import SimpleNamespace
from collections import namedtuple
import random
from argparse import ArgumentParser, ArgumentTypeError
import cProfile

import numpy as np
import torch
from prettytable import PrettyTable
from drl.utils.pallete import Pallete
from drl.model2graph.model2graph import get_graph_from_model
from drl.main.rl_agent import train, sample_trajectories
from drl.main.inference_sampler import INFERENCE_MODES
#from drl.env.RemEnv import SubprocEnv
from utils.seed_util import seed_everything
import wandb
from time import time

logger = logging.getLogger(__name__)
Sample = namedtuple("Sample", "graph ruleApplication isLast modelPath")
# SampleDescriptor = namedtuple("SampleDescriptor", "appState isLast modelPath")


class SampleDescriptor(
    namedtuple("SampleDescriptor", ["appState", "isLast", "modelPath"])
):
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        else:
            return super().__getitem__(key)


class Config(
    namedtuple(
        "Config",
        [
            "unitPrefix",
            "nodeIdentifierAttributeName",
            "maxNodeParams",
            "maxValueParams",
            "units",
            "additionalNodeEncodingAttributes",
        ],
    )
):
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        else:
            return super().__getitem__(key)


def init_wandb_run(args, ts):
    return wandb.init(
        project="momot-rl",
        config=args,
        name=f"{args.wandb_config['group']}-{args.seed}-{ts}",
        **args.wandb_config,
    )


def run_inference(pallete, args, ts):
    """Sample with a trained checkpoint, using the palette saved next to it."""
    pallete_path = os.path.join(os.path.dirname(args.ac_model), "pallete.json")
    if not os.path.exists(pallete_path):
        raise SystemExit(
            f"Palette file not found at {pallete_path}. Pass a checkpoint from a training run."
        )
    print(f"Loading palette from {pallete_path}")
    with open(pallete_path) as f:
        pallete.from_json(json.load(f))

    if len(args.seeds) > 1:
        logger.warning(
            f"Multiple seeds given for inference; using the first seed {args.seeds[0]}"
        )
    args.seed = args.seeds[0]

    if args.use_wandb:
        run = init_wandb_run(args, ts)
    sample_trajectories(pallete, args, ts)
    if args.use_wandb:
        run.finish()


def main(args, conf, ts):

    pallete = get_pallete(args, conf)

    if args.use_wandb:
        wandb.login(key=args.wandb_config['key'])
        args.wandb_config.pop('key')

    if args.infer:
        run_inference(pallete, args, ts)
        return

    if args.train:

        if args.rl:
            file_handler = None
            for seed in args.seeds:
                detach_file_logger(file_handler)
                args.seed = seed
                seed_everything(seed)
                ts = time()
                args.model_path = os.path.join(
                    args.base_model_path, f"{seed}-{ts}"
                )
                file_handler = attach_file_logger(args.model_path)
                log_configuration(args)

                if args.use_wandb:
                    run = init_wandb_run(args, ts)

                logger.info(f"Starting RL run {ts} (seed={args.seed})")
                train(pallete, args, ts)

                if args.use_wandb:
                    run.finish()

            detach_file_logger(file_handler)

        else:
            with open(
                os.path.join(args.training_dataset, args.sample_description), "r"
            ) as f:
                data_samples = json.loads(
                    f.read(),
                    object_hook=lambda d: (
                        SampleDescriptor(**d)
                        if "isLast" in d and "modelPath" in d
                        else d
                    ),
                )

            training_samples = [
                Sample(
                    pallete.remove_out_of_scope(
                        get_graph_from_model(sample.modelPath, [args.metamodel])
                    ),
                    sample.appState,
                    sample.isLast,
                    sample.modelPath,
                )
                for sample in data_samples
            ]
            logger.info(f"Loaded dataset of {len(training_samples)} elements")
            # train_generator(training_samples, pallete, args)

    else:
        raise ValueError("--train, --inference, or --evaluate should be provided.")


def get_pallete(args, conf):
    pallete = Pallete(
        path_metamodel=args.metamodel, root_element=args.root_object, conf=conf
    )

    return pallete


def create_directory_and_file(model_path, filename):
    # Join the model_path with the filename
    file_path = os.path.join(model_path, filename)

    # Create the directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w"):
            pass  # This just creates an empty file


def smart_cast(value):
    """Automatically converts strings to int, float, or keeps as string if neither."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Keep as string if not a number


def parse_seeds(value):
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ArgumentTypeError("--seed must contain at least one integer")
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ArgumentTypeError(
            f"Invalid --seed {value!r}; expected integers, optionally comma-separated"
        )


def attach_file_logger(model_path):
    create_directory_and_file(model_path, "info_500.log")
    file_handler = logging.FileHandler(os.path.join(model_path, "info_500.log"))
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return file_handler


def detach_file_logger(file_handler):
    if file_handler is None:
        return
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def log_configuration(args):
    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info("Configuration:\n{}".format(config_table))


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training a M2 generator")
    parser.add_argument("--case_config", default="config.json")
    parser.add_argument("--metamodel", default="./data/ecore_simplified.ecore")
    parser.add_argument("--root_object", default="EPackage")
    parser.add_argument(
        "--hidden_dim",
        help="Hidden dimension of the neural model.",
        type=int,
        default=128,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--model_path", default="models/ecore_github")
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument(
        "--output_path",
        help="Output folder where the models will be generated",
        default="generated_models/ecore_exercise",
    )
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--train", help="If training phase", action="store_true")
    parser.add_argument(
        "--rl", default="false", help="use reinforcement learning", action="store_true"
    )

    parser.add_argument(
        "--rl_hyperparams", nargs="+", help="Hyperparameters as key-value pairs"
    )
    parser.add_argument("--ac_model", default="", help="model path")
    parser.add_argument(
        "--rl_evaluate",
        help="Evaluate policy on server test models",
        action="store_true",
    )
    parser.add_argument(
        "--rl_eval_mode",
        choices=["stochastic", "greedy", "both"],
        default="stochastic",
        help="Evaluation episodes per test model: one sampled (stochastic), one most "
        "likely (greedy), or both, logged and saved separately.",
    )
    parser.add_argument(
        "--rl_eval_interval",
        type=int,
        default=4,
        help="Run an evaluation phase after every N rollouts.",
    )
    parser.add_argument(
        "--rl_es_metric",
        choices=["rollout", "eval_stochastic", "eval_greedy"],
        default="rollout",
        help="Early stopping metric. rollout (default): windowed macro return of the "
        "training rollouts. eval_stochastic / eval_greedy: per-model return of that "
        "evaluation variant, normalized by the initial evaluation; needs --rl_evaluate "
        "with a matching --rl_eval_mode, and patience then counts evaluation phases.",
    )
    parser.add_argument(
        "--rl_eval_save_solutions",
        action="store_true",
        help="Store the solutions of every evaluation phase per test model under "
        "<model_path>/eval_solutions/<variant>/<model>/<phase>.",
    )
    parser.add_argument(
        "--wandb_config", nargs="+", help="Wandb configuration as key-value pairs"
    )
    parser.add_argument(
        "--rl_tsteps", type=int, default=1, help="Total steps for RL training"
    )
    parser.add_argument(
        "--rl_sample_partial_trajectories",
        help="Sample partial trajectories for RL training",
        action="store_true",
    )
    parser.add_argument(
        "--no_nn_updates", help="Perform no neural network updates", action="store_true"
    )
    parser.add_argument(
        "--seed",
        type=parse_seeds,
        default="12",
        help="Random seed, or comma-separated list of seeds for sequential training runs",
    )
    parser.add_argument(
        "--infer",
        choices=INFERENCE_MODES,
        default=None,
        help=(
            "Sample solutions with the checkpoint --ac_model instead of training. "
            "one-per-model: one stochastic episode per model, all models in one batch. "
            "fixed-count: --num_solutions episodes of the served model, Pareto set written. "
            "until-converged: episodes until --converge_patience episodes in a row do not "
            "improve the Pareto set. "
            "max-likelihood: one greedy (most likely) episode per model, with its "
            "transformation sequence."
        ),
    )
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=None,
        help="Number of episodes for --infer fixed-count.",
    )
    parser.add_argument(
        "--converge_patience",
        type=int,
        default=None,
        help="Episodes without Pareto improvement that end --infer until-converged "
        "(default: num_envs).",
    )
    parser.add_argument("--port", help="Port for the server.", type=int, default=8000)
    parser.add_argument(
        "--env_transport",
        choices=["batched", "subproc"],
        default="batched",
        help="Vectorized env transport: batched (persistent/compact) or subproc (legacy XMI).",
    )
    parser.add_argument(
        "--env_groups",
        type=int,
        default=1,
        help="Number of env groups for double-buffered stepping (batched transport).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    if args.infer and not args.ac_model:
        parser.error("--infer needs a checkpoint via --ac_model")
    if args.infer == "fixed-count" and (args.num_solutions is None or args.num_solutions < 1):
        parser.error("--infer fixed-count needs a positive --num_solutions")

    args.rl_hyperparams = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in args.rl_hyperparams[0].split(" ")
    }
    args.wandb_config = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in (args.wandb_config or [""])[0].split(" ")
        if arg
    }

    args.use_wandb = hasattr(args, "wandb_config") and "key" in args.wandb_config

    with open(args.case_config, "r") as f:
        conf = json.loads(
            f.read(), object_hook=lambda d: Config(**d) if "maxNodeParams" in d else d
        )

    args.seeds = args.seed
    args.base_model_path = args.model_path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Categorical and Normal argument checks sync on every sample. Keep them
    # only while hunting NaNs; rollout and the update both build distributions.
    torch.distributions.Distribution.set_default_validate_args(
        bool(getattr(args, "debug_nan", False))
    )
    timestamp = time()

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    is_rl_train = args.train and args.rl and not args.infer
    if not is_rl_train:
        args.seed = args.seeds[0]
        seed_everything(args.seed)
        if not args.infer:
            args.model_path = os.path.join(
                args.base_model_path, f"{args.seed}-{timestamp}"
            )
        attach_file_logger(args.model_path)
        log_configuration(args)

    main(args, conf, timestamp)
