import torch
from drl.main.ppo import PPO
from drl.neural_model.gnn_predictor import SharedEncoderAC, SeparateEncoderAC


def train(pallete, args, run_id):
    """
    Trains the model.

    Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

    Return:
            None
    """

    # Create a model for PPO.

    if args.rl_hyperparams["encoder"] == "shared":
        nn_class = SharedEncoderAC
    elif args.rl_hyperparams["encoder"] == "separate":
        nn_class = SeparateEncoderAC
    else:
        raise ValueError(f"Unknown encoder type: {args.rl_hyperparams['encoder']}")

    model = PPO(gnn_actor_critic=nn_class, args=args, pallete=pallete, run_id=run_id)

    if args.ac_model and args.ac_model != "":
        ac_model = args.ac_model
        print(f"Loading in {ac_model}", flush=True)

        ## get parameters from model.ac for later comparison after loading
        model.ac.load_state_dict(torch.load(ac_model))

        print(f"Successfully loaded model")

    model.learn(total_timesteps=args.rl_tsteps)
    model.writer.close()


def sample_trajectories(pallete, args, run_id):
    print(f"Inference", flush=True)

    if args.rl_hyperparams["encoder"] == "shared":
        nn_class = SharedEncoderAC
    elif args.rl_hyperparams["encoder"] == "separate":
        nn_class = SeparateEncoderAC
    else:
        raise ValueError(f"Unknown encoder type: {args.rl_hyperparams['encoder']}")

    model = PPO(gnn_actor_critic=nn_class, args=args, pallete=pallete, run_id=run_id)

    # FIX: Add missing model loading logic
    if args.ac_model and args.ac_model != "":
        ac_model = args.ac_model
        print(f"Loading in {ac_model}", flush=True)
        model.ac.load_state_dict(torch.load(ac_model))
        print(f"Successfully loaded model")

    model.inference_sampling(
        args.rl_sample_n_trajectories, args.seed, args.infer_set_single, nn_class
    )
