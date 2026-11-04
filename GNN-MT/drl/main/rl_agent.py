from drl.main.hyperparams import Hyperparams
from drl.main.inference import run_inference
from drl.main.ppo import PPO


def train(pallete, args, run_id):
    """Train a policy, starting from the checkpoint ``args.ac_model`` if there is one."""
    model = PPO(args=args, pallete=pallete, run_id=run_id)
    if args.ac_model:
        print(f"Loading in {args.ac_model}", flush=True)
        model.policy.load_weights(args.ac_model)
    model.learn(total_timesteps=args.fes)
    model.train_log.close()


def sample_trajectories(pallete, args, run_id):
    """Run ``args.infer`` with the checkpoint ``args.ac_model``."""
    print(f"Inference ({args.infer})", flush=True)
    hp = Hyperparams.from_dict(args.hyperparams)
    pallete.masking = hp.masking
    run_inference(
        args.infer,
        args.ac_model,
        pallete,
        args.device,
        args.seed,
        args.wandb_config.get("group", "inference"),
        args.port,
        run_id,
        env_groups=args.env_groups,
        masking=hp.masking,
        time_scale=hp.time_scale,
        num_solutions=args.num_solutions,
        converge_patience=args.converge_patience,
        num_batches=args.num_batches,
    )
