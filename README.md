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
- `server-executables/`: RL evironment module (standalone)


## 🛠 Requirements
The prototype was tested using Python 3.9 with the required dependencies and a Java 17 runtime. 

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
For the Java server application, executable jar's are provided to start the server for training and running policies. The server will be automatically spinned up when running the prepared scripts `train_agent.sh` and `run_policy.sh`. 

> **Note:** The path to the runtime needs to be set in `.env` in the root directory (see provided `.env.example`).

## 🚀 Train a policy and run it to optimize model instances
To train and use policies, a bash script is provided to start the RL environment (server) and agent (client) application. Run the following:


```sh
train_agent.sh \
  --case <case> \
  --runs <runs> \
  --train-path <path_to_training_models> \
  --test-path <path_to_test_models> \
  --experiment-name <name> \
  --partial_trajectories
```

- `case`: one of `slb`,`cra`,`sp`,`sp_uc`.
- `runs`: the number of training runs.
- `train-path`: relative path to models for training, e.g., `"models/datasets/slb/train-N30-25.-50."`
- `name`: Name of the experiment and location to store policies during/after training (stored in ./GNN-MT/models/rl/\<case\>/\<name\>/)
- `test_path` (optional): relative path to models for evaluating current policy, e.g., `"models/datasets/slb/test-N30-25.-50."`
- `partial_trajectories` (optional): Flag to use truncated rollouts in training updates. Otherwise, full episodes are used only

We used truncated rollouts (`--partial_trajectories`) to keep track of performance increments with fixed-size updates, but use only full episodes to obtain the best performance. 

With run_policy.sh, stored policies can be used to transform problem models provided in a folder. Of each run, it automatically loads the checkpoint of a policy that achieved the highest reward. 

```sh
run_policy.sh \
  --case <case> \
  --policy-dir <path_to_policies> \
  --problem-model-dir <path_to_models> \
  --n-solutions <solutions> \
```

- `case`: one of `slb`,`cra`,`sp`,`sp_uc`.
- `policy-dir`: relative path to policies (stored in) relative to project folder "GNN-MT/models/rl", e.g., `"slb/<experiment_name>"`.
- `problem-model-dir`: relative path to folder with models to run policy on, e.g., `"models/datasets/slb/train-N30-25.-50."`
- `n-solutions`: number of solutions to generate with batch-wise sampling, e.g., `10` or multiple of 10 for batch-wise, `-1` for sampling until no improvement
