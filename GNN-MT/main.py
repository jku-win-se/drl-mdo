import glob
import logging
import os
import json
from types import SimpleNamespace
from collections import namedtuple
import random
from argparse import ArgumentParser
import cProfile

import numpy as np
import torch
from prettytable import PrettyTable
from drl.utils.pallete import Pallete
from drl.model2graph.model2graph import get_graph_from_model
from drl.main.rl_agent import train, sample_trajectories
from drl.env.RemEnv import SubprocEnv
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


def main(args, conf, ts):

    pallete = get_pallete(args, conf)

    if args.use_wandb:
        wandb.login()

    if args.train:

        if args.rl:

            if args.rl_sample_n_trajectories:

                parent_dir = os.path.dirname(args.ac_model)
                if not os.path.exists(f"{parent_dir}/pallete.json"):
                    print(
                        f"Pallete file not found in {parent_dir}/pallete.json. Please provide a valid model path."
                    )
                    exit(1)
                else:
                    with open(f"{parent_dir}/pallete.json") as f:
                        print(f"Loading pallete from {parent_dir}/pallete.json")
                        pallete.from_json(json.load(f))

            if args.use_wandb:
                project_name = "momot-rl"

                if not os.path.exists(
                    os.path.join(args.wandb_config["group"], "wandb")
                ):
                    os.makedirs(os.path.join(args.wandb_config["group"], "wandb"))
                run = wandb.init(
                    project=project_name,
                    config=args,
                    name=f"{args.wandb_config['group']}-{args.seed}-{ts}",
                    **args.wandb_config,
                )

            if args.rl_sample_n_trajectories:

                sample_trajectories(pallete, args, ts)
            else:
                logger.info(f"Starting RL run {ts} (seed={args.seed})")
                train(pallete, args, ts)

            if args.use_wandb:
                run.finish()

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
        "--rl_autoreg",
        default=False,
        help="use reinforcement learning",
        action="store_true",
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
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument(
        "--rl_sample_n_trajectories",
        type=int,
        default=None,
        help="Sample trajectories for RL training",
    )
    parser.add_argument(
        "--infer_set_single", help="Evaluate of generated models", action="store_true"
    )
    parser.add_argument("--port", help="Port for the server.", type=int, default=8000)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    args.rl_hyperparams = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in args.rl_hyperparams[0].split(" ")
    }
    args.wandb_config = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in args.wandb_config[0].split(" ")
    }

    args.use_wandb = hasattr(args, "wandb_config") and "key" in args.wandb_config

    with open(args.case_config, "r") as f:
        conf = json.loads(
            f.read(), object_hook=lambda d: Config(**d) if "maxNodeParams" in d else d
        )

    seed_everything(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time()
    if not args.rl_sample_n_trajectories:
        args.model_path = os.path.join(args.model_path, f"{args.seed}-{timestamp}")

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    create_directory_and_file(args.model_path, "info_500.log")
    file = logging.FileHandler(os.path.join(args.model_path, "info_500.log"))
    file.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s"
    )
    file.setFormatter(formatter)
    logger.addHandler(file)

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info("Configuration:\n{}".format(config_table))

    main(args, conf, timestamp)
