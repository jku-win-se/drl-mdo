import logging
import os
import json
import platform
import subprocess
from collections import namedtuple
from argparse import ArgumentParser, ArgumentTypeError

import procbridge as pb
import torch
from prettytable import PrettyTable
from drl.utils.pallete import Pallete
from drl.main.rl_agent import train, sample_trajectories
from drl.main.inference_sampler import INFERENCE_MODES
#from drl.env.RemEnv import SubprocEnv
from utils.seed_util import seed_everything
import wandb
from time import time

logger = logging.getLogger(__name__)

# Files of each case the server can run: case config, metamodel, root object.
AGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
CASE_FILES = {
    "cra": ("datasets/cra/config.json", "data/cra.ecore", "ClassModel"),
    "slb": ("datasets/slb/config.json", "data/stack.ecore", "StackModel"),
    "scrum": ("datasets/sp/config.json", "data/sprintplanning.ecore", "Plan"),
    "scrum-uc": ("datasets/sp/config.json", "data/sprintplanning.ecore", "Plan"),
}


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

    if not args.train:
        raise ValueError("--train or --infer should be provided.")

    file_handler = None
    for seed in args.seeds:
        detach_file_logger(file_handler)
        args.seed = seed
        seed_everything(seed)
        ts = time()
        run_id = f"{seed}-{ts}"
        args.model_path = os.path.join(args.base_model_path, run_id)
        file_handler = attach_file_logger(args.model_path)
        log_configuration(args)
        write_run_args(args)

        if args.use_wandb:
            run = init_wandb_run(args, ts)

        logger.info(f"Starting RL run {run_id}")
        train(pallete, args, run_id)

        if args.use_wandb:
            run.finish()

    detach_file_logger(file_handler)


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


def git_output(*git_args):
    try:
        return subprocess.run(
            ["git", "-C", AGENT_ROOT, *git_args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_args(args):
    """What determines this run: its arguments, datasets, code version and hardware."""
    skipped = {"wandb_config", "base_model_path"}  # the W&B config can hold an API key
    info = {key: value for key, value in vars(args).items() if key not in skipped}
    info["commit"] = git_output("rev-parse", "HEAD")
    info["dirty"] = bool(git_output("status", "--porcelain"))
    info["hardware"] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(args.model_path, "run_args.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def log_configuration(args):
    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info("Configuration:\n{}".format(config_table))


if __name__ == "__main__":
    parser = ArgumentParser(description="Train or sample with an RL policy on a running env server")
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
        "--hyperparams", nargs="+", help="Hyperparameters as key-value pairs"
    )
    parser.add_argument("--ac_model", default="", help="model path")
    parser.add_argument(
        "--eval_mode",
        choices=["stochastic", "greedy", "both"],
        default="stochastic",
        help="Validation episodes per env: sampled (stochastic), most likely (greedy), or "
        "both, logged and saved separately. Validation runs if the server has a validation "
        "dataset. It records solutions and returns and picks nothing.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=4,
        help="Run a validation phase after every N PPO iterations.",
    )
    parser.add_argument(
        "--eval_save_solutions",
        action="store_true",
        help="Store the solutions of every validation phase per model under "
        "<model_path>/eval_solutions/<variant>/<model>/<phase>.",
    )
    parser.add_argument(
        "--keep_checkpoints",
        choices=["best", "all"],
        default="best",
        help="best: store a checkpoint when the best policy improves. all: at every tick of the "
        "training metric. best_model.pth always holds the best policy, and checkpoints.csv "
        "lists every tick.",
    )
    parser.add_argument(
        "--final_eval",
        choices=["none", "val", "test", "both"],
        default="none",
        help="After training, sample the best policy on these datasets of the server: one batch "
        "per model.",
    )
    parser.add_argument(
        "--wandb_config", nargs="+", help="Wandb configuration as key-value pairs"
    )
    parser.add_argument(
        "--fes", type=int, default=1, help="Training budget in fitness evaluations"
    )
    parser.add_argument(
        "--partial_trajectories",
        help="Use truncated rollouts for training updates instead of full episodes",
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
            "transformation sequence. "
            "batches: --num_batches batches of one episode per env on each model, logged per batch."
        ),
    )
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=None,
        help="Number of episodes for --infer fixed-count.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches per model for --infer batches.",
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

    args.hyperparams = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in " ".join(args.hyperparams or []).split()
    }
    args.hyperparams.setdefault("encoder", "shared")
    args.hidden_dim = args.hyperparams.pop("hidden_dim", args.hidden_dim)
    args.wandb_config = {
        arg.split("=")[0]: smart_cast(arg.split("=")[1])
        for arg in (args.wandb_config or [""])[0].split(" ")
        if arg
    }

    args.use_wandb = hasattr(args, "wandb_config") and "key" in args.wandb_config

    # Case and datasets are what the server was started with.
    args.run_info = pb.Client("127.0.0.1", args.port).request("getRunInfo") or {}
    case_id = args.run_info.get("case")
    if case_id not in CASE_FILES:
        raise SystemExit(f"Unknown case {case_id!r} from the server on port {args.port}.")
    case_config, metamodel, args.root_object = CASE_FILES[case_id]
    args.metamodel = os.path.join(AGENT_ROOT, metamodel)

    with open(os.path.join(AGENT_ROOT, case_config), "r") as f:
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

    if args.infer or not args.train:
        args.seed = args.seeds[0]
        seed_everything(args.seed)
        if not args.infer:
            args.model_path = os.path.join(
                args.base_model_path, f"{args.seed}-{timestamp}"
            )
        attach_file_logger(args.model_path)
        log_configuration(args)

    main(args, conf, timestamp)
