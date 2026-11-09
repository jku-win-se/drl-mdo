# DRL-MDO - Deep reinforcement learning for model-driven optimization
DRL-MDO is an MDO framework using deep neural networks to transform models under guidance of fitness and constraint functions.  

## Description
The prototype consists of two modules communicating via TCP sockets: 
- The RL agent (Python, Client): Implements an ML model to determine the next transformation rule call for a given input model
- The RL environment (Java, Server): Hosts the interpreter engine and evaluation utilities. It supports an API following the conventional RL interaction pattern: A "step" function to execute a received rule call (i.e., the action) on the current model state, returning a response object with the reward and transformed model (i.e., the successor state).

## Repository Structure

Key directories:

- `evaluation`: Test cases results and analysis scripts
- `GNN-MT/`: RL agent (GNN-based ML model)
- `models/`: Training/Test models
- `momot-rl-env-api/`: RL environment module (source)
- `server-executables/`: RL environment build and launch scripts
- `experiments/`: Scripts and instructions for the experiments


## 🛠 Requirements
The prototype was tested using Python 3.8 and 3.9 with the required dependencies and a Java 17 or newer JDK. 

### RL agent
For Python, we recommend setting up a conda environment with the listed dependencies ([GNN-MT/environment.yml](./GNN-MT/environment.yml)) as follows:

```
conda create -n <env_name> python=3.9
conda activate <env_name>
conda env update -f environment.yml
``` 

The RL agent consists of a GNN-based ML model implemented with PyTorch and utilities from PyTorch Geometric. Training was executed on an NVIDIA GeForce RTX 3060 with CUDA 12.8. The following packages are system-dependent and  suitable versions must be installed for your hardware (listed versions reflect our setup):

- `torch==2.7.1`
- `torch-geometric==2.6.1`
- `torch_cluster==1.6.3`
- `torch_scatter==2.1.2`
- `torch_sparse==0.6.18`
- `torch_spline_conv==1.2.2`

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

> **Note:** The path to the conda environment needs to be set in `.env` in the root directory (see provided `.env.example`).


### RL environment
The Java server is built from source with `server-executables/build.sh`. The server will be automatically spinned up when running the prepared scripts `train_agent.sh` and `run_policy.sh`. 

> **Note:** The path to the runtime needs to be set in `.env` in the root directory (see provided `.env.example`).

## 🚀 Train a policy and run it to optimize model instances
Build the RL environment once. This needs `ECLIPSE_PLUGINS` in `.env`, the plugins folder of an Eclipse Modeling install with Henshin:

```sh
server-executables/build.sh
```

`train_agent.sh` starts the server and trains one setup for N seeds, one seed at a time, using the first N stored seeds:

```sh
train_agent.sh \
  --case <case> \
  --runs <runs> \
  --experiment-name <name> \
  --train-path <path_to_training_models> \
  --val-path <path_to_validation_models> \
  --test-path <path_to_test_models> \
  --partial_trajectories
```

- `case`: one of `slb`, `cra`, `sp`, `sp_uc`.
- `runs`: the number of seeds to run.
- `name`: name of the experiment. Results go to `GNN-MT/models/rl/<case>/<name>/<seed>-<timestamp>/`.
- `train-path`, `val-path`, `test-path`: folders of models or a single `.xmi`, relative to the repository root. Setting `val-path` turns periodic validation on, which records solutions and returns. Setting `test-path` turns the final run on: after training, the best policy samples one batch per test model. Only `train-path` is required.
- `partial_trajectories` (optional): use truncated rollouts in training updates; otherwise full episodes are used.

`train_agent.sh` never generates models. `generate_dataset.sh` does, beforehand:

```sh
generate_dataset.sh --case <case> [--sizes MIN-MAX] --train 20,40 [--val N] [--test N] --seed <seed> --out <folder>
```

It writes `train-<n>/` per training size (smaller sets are contained in larger ones), `val/`, `test/` and `manifest.csv`. Models identical to a benchmark of the case (`models/benchmark/<case>`, plus any `--exclude` folder) or to another generated model in any split are rejected and redrawn. It skips a folder that already holds a dataset. Without `--sizes`, the size range of the case applies; that range and the other model characteristics are constants at the top of [ModelGenerator.java](./momot-rl-env-api/examples/org.learnapi/src/learnapi/ModelGenerator.java). All flags are at the top of [train_agent.sh](./train_agent.sh), and [experiments/README.md](./experiments/README.md) shows complete setups with their artifacts and analysis.

`run_policy.sh` samples solutions with stored policies:

```sh
run_policy.sh \
  --case <case> \
  --mode <mode> \
  --models <path_to_models> \
  (--policy-dir <experiment_name> | --checkpoint <file>)
```

Run `run_policy.sh --help` for the modes and options.
