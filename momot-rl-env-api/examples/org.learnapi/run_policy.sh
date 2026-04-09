#!/bin/bash --login

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="$(cd "$SCRIPT_DIR/../../../GNN-MT" && pwd)"

source "$SCRIPT_DIR/paths.sh"
source "$SCRIPT_DIR/py_train_scripts.sh"
source "$SCRIPT_DIR/java_environment_server.sh"
source "$SCRIPT_DIR/get_best_checkpoints.sh"

N_ENVS=10
#CASE="$1"
#POLICY_DIR=$2
#PROBLEM_MODEL_DIR=$3
N_SOLUTIONS=-1
PORT=8000


while [[ "$#" -gt 0 ]]; do
  case $1 in
    --case) CASE="$2"; shift ;;
    --policy-dir) POLICY_DIR="$2"; shift ;;
    --problem-model-dir) PROBLEM_MODEL_DIR="$2"; shift ;;
    --n-solutions) N_SOLUTIONS="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

MODEL_DIR_NAME=$(basename "$PROBLEM_MODEL_DIR")


echo "CASE=$CASE"
echo "POLICY_DIR=$POLICY_DIR"
echo "PROBLEM_MODEL_DIR=$PROBLEM_MODEL_DIR"
echo "N_SOLUTIONS=$N_SOLUTIONS"

get_filenames_no_ext() {
    local dir="$1"

    for path in "$dir"/*; do
        [[ -f "$path" ]] || continue
        basename "${path%.*}"
    done
}

for subdir in "$AGENT_ROOT/models/rl/$CASE/$POLICY_DIR"/*/; do
     if [[ "$(basename "$subdir")" == "sampled_solutions" ]]; then
        continue
    fi

    subdir_name=$(basename "$subdir")

    [[ "$subdir_name" == solutions* ]] && continue

    read -r seed best_overall best_pre100k < <(get_best_from_subdir "$subdir")

     if [[ $N_SOLUTIONS -eq 1 ]]; then
        echo "Running single-solution sampling for $CASE with policy $best_overall on problem models in $PROBLEM_MODEL_DIR"
        num_models=$(find "$SCRIPT_DIR/$PROBLEM_MODEL_DIR" -maxdepth 1 -type f | wc -l)

        start_server_trainingset $CASE $num_models $PROBLEM_MODEL_DIR NONE "generalization-$CASE"
        run_inference $CASE "num_envs=$num_models encoder=shared rnn_type=gru" "$CASE/$POLICY_DIR/$best_pre100k" 1 "$seed" "$(basename "$PROBLEM_MODEL_DIR")/single-solution" 8000 --infer-single-solution
        cleanup
    else
        echo "Running sampling for $CASE with policy $best_overall on problem models in $PROBLEM_MODEL_DIR, generating $N_SOLUTIONS MT sequences per model"
        for name in $(get_filenames_no_ext "$SCRIPT_DIR/$PROBLEM_MODEL_DIR"); do
            start_server $CASE -1 $N_ENVS "$PROBLEM_MODEL_DIR/$name" generalization-$CASE $PORT
            run_inference $CASE "num_envs=$N_ENVS encoder=shared rnn_type=gru" "$CASE/$POLICY_DIR/$best_pre100k" $N_SOLUTIONS "$seed" "$name/$N_SOLUTIONS-solutions"
            cleanup  "benchmark/$name/conv" 
	    done
    fi
done  
		