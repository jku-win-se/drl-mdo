#!/bin/bash --login

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="$(cd "$SCRIPT_DIR/../../../GNN-MT" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

source "$SCRIPT_DIR/paths.sh"
source "$SCRIPT_DIR/py_train_scripts.sh"
source "$SCRIPT_DIR/java_environment_server.sh"

# Defaults
RUNS=30
TEST_MODELS_PATH="NONE"
EVALUATE_FLAG=""
PARTIAL_TRAJECTORIES_FLAG=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --case) CASE="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --train-path) TRAINING_MODELS_PATH="$2"; shift ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift ;;
    --test-path) TEST_MODELS_PATH="$2"; shift ;;
    --evaluate) EVALUATE_FLAG="--evaluate" ;;
    --partial_trajectories) PARTIAL_TRAJECTORIES_FLAG="--sample_partial_trajectories" ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

echo "CASE=$CASE"
echo "RUNS=$RUNS"
echo "TRAINING_MODELS_PATH=$TRAINING_MODELS_PATH"
echo "TEST_MODELS_PATH=$TEST_MODELS_PATH"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "EVALUATE_FLAG=$EVALUATE_FLAG - Policies will be run frequently during training to evaluate performance on test set"
echo "PARTIAL_TRAJECTORIES_FLAG=$PARTIAL_TRAJECTORIES_FLAG - Using partial trajectories for updates during training"

if [ "$RUNS" -gt 30 ]; then
    echo "Error: RUNS must be 30 or less (got $RUNS)" >&2
    exit 1
fi

SEEDS=(42 17 92 5 88 101 7 66 23 55 19 4 78 29 12 84 31 9 40 64 90 73 38 26 13 97 3 21 60 15)
FES=100000 	
LOGFILE_NAME="$CASE_$(date '+%Y%m%d_%H%M%S')"
LOGFILE="$LOGFILE_NAME.log"
N_ENVS=10
 
for ((i = 1; i <= $RUNS; i++)); do
    echo "=== Run $i of $RUNS ==="
    echo "=> Checkpoints will be stored at $AGENT_ROOT/models/rl/$CASE/$EXPERIMENT_NAME"

	  start_server_trainingset $CASE $N_ENVS $TRAINING_MODELS_PATH $TEST_MODELS_PATH $LOGFILE_NAME
	  run_training $CASE ${SEEDS[$i-1]} "num_envs=$N_ENVS" $FES $EXPERIMENT_NAME "models/rl/$CASE/$EXPERIMENT_NAME" $EVALUATE_FLAG $PARTIAL_TRAJECTORIES_FLAG >> $LOGFILE 2>&1 
	  cleanup >> $LOGFILE 2>&1
done
