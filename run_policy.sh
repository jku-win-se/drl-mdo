#!/bin/bash --login
# Sample model transformation sequences with trained policies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/server-executables"
AGENT_ROOT="$(cd "$SCRIPT_DIR/GNN-MT" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_policy.sh --case CASE --models PATH --mode MODE
                (--policy-dir DIR | --checkpoint FILE)
                [--n-solutions N] [--n-envs N] [--env-groups N] [--seed S] [--port P]

Modes:
  one-per-model    One stochastic episode per model. All models of PATH run in one batch.
  max-likelihood   One greedy (most likely) episode per model, written with its
                   transformation sequence (sequence.json). PATH may be a folder or one model.
  fixed-count      Exactly --n-solutions episodes per model; their Pareto set is written.
  until-converged  Episodes per model until --n-envs episodes in a row leave the
                   Pareto set unchanged.

Options:
  --case CASE          slb | cra | sp | ...
  --models PATH        Folder of problem models or a single model file (.xmi),
                       relative to server-executables/ or absolute.
  --policy-dir DIR     Training run folder under GNN-MT/models/rl/CASE/. Every seed
                       subfolder runs with its best checkpoint within 100k evaluations.
  --checkpoint FILE    A single ac_*.pth, absolute or relative to GNN-MT/models/rl/.
  --n-solutions N      Episodes per model for fixed-count.
  --n-envs N           Parallel envs for fixed-count / until-converged (default 10).
  --env-groups N       Env groups whose inference overlaps Java stepping (default 1).
  --seed S             Sampling seed (default: the seed of the checkpoint's run folder).
  --port P             Environment server port (default 8000).

Examples:
  run_policy.sh --case cra --mode max-likelihood --policy-dir my-run \
                --models models/datasets/cra/test-N30-d0.15-.25
  run_policy.sh --case cra --mode fixed-count --n-solutions 100 \
                --checkpoint cra/my-run/12-1700000000/ac_10-50000-1.23.pth \
                --models models/datasets/cra/test-N30-d0.15-.25/100-0.180.xmi
EOF
}

CASE=""
MODELS=""
MODE=""
POLICY_DIR=""
CHECKPOINT=""
N_SOLUTIONS=""
N_ENVS=10
ENV_GROUPS=1
SEED=""
PORT=8000

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --case) CASE="$2"; shift ;;
    --models) MODELS="$2"; shift ;;
    --mode) MODE="$2"; shift ;;
    --policy-dir) POLICY_DIR="$2"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift ;;
    --n-solutions) N_SOLUTIONS="$2"; shift ;;
    --n-envs) N_ENVS="$2"; shift ;;
    --env-groups) ENV_GROUPS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

fail() {
  echo "Error: $1" >&2
  echo "Run with --help for usage." >&2
  exit 1
}

[[ -n "$CASE" ]] || fail "--case is required"
[[ -n "$MODELS" ]] || fail "--models is required"
case "$MODE" in
  one-per-model|max-likelihood|fixed-count|until-converged) ;;
  "") fail "--mode is required" ;;
  *) fail "unknown mode '$MODE'" ;;
esac
if [[ -n "$POLICY_DIR" && -n "$CHECKPOINT" ]] || [[ -z "$POLICY_DIR" && -z "$CHECKPOINT" ]]; then
  fail "give exactly one of --policy-dir and --checkpoint"
fi
if [[ "$MODE" == "fixed-count" ]] && ! [[ "$N_SOLUTIONS" =~ ^[1-9][0-9]*$ ]]; then
  fail "fixed-count needs --n-solutions N with N >= 1"
fi

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
  echo "Error: .env not found in $SCRIPT_DIR" >&2
  exit 1
fi

set -a
source "$SCRIPT_DIR/.env"
set +a

source "$PROJECT_DIR/py_train_scripts.sh"
source "$PROJECT_DIR/java_environment_server.sh"
source "$PROJECT_DIR/get_best_checkpoints.sh"
source "$PROJECT_DIR/init.sh"

# The Java server runs from server-executables/, so relative model paths resolve there.
if [[ "$MODELS" = /* ]]; then
  MODELS_ABS="$MODELS"
else
  MODELS_ABS="$PROJECT_DIR/$MODELS"
fi
[[ -e "$MODELS_ABS" ]] || fail "models path not found: $MODELS_ABS"

# Model instance names: file names without extension.
model_names() {
  if [[ -f "$MODELS_ABS" ]]; then
    local file
    file=$(basename "$MODELS_ABS")
    echo "${file%.*}"
    return
  fi
  local path
  for path in "$MODELS_ABS"/*; do
    [[ -f "$path" ]] || continue
    local file
    file=$(basename "$path")
    echo "${file%.*}"
  done
}

# Folder containing the model instances, as passed to the Java server.
if [[ -f "$MODELS_ABS" ]]; then
  MODELS_DIR=$(dirname "$MODELS")
  MODELS_LABEL=$(basename "${MODELS%.*}")
else
  MODELS_DIR="${MODELS%/}"
  MODELS_LABEL=$(basename "$MODELS_DIR")
fi

# Run MODE with one checkpoint on all requested models.
sample_with_checkpoint() {
  local checkpoint="$1"
  local seed="$2"
  local hyperparams="encoder=shared rnn_type=gru"

  case "$MODE" in
    one-per-model|max-likelihood)
      # One env per model, all models in one batch.
      local num_models
      num_models=$(model_names | wc -l)
      if [[ -f "$MODELS_ABS" ]]; then
        start_server "$CASE" -1 1 "$MODELS_DIR/$MODELS_LABEL" "generalization-$CASE" "$PORT"
      else
        start_server_trainingset "$CASE" "$num_models" "$MODELS_DIR" NONE "generalization-$CASE" "$PORT"
      fi
      run_inference "$CASE" "num_envs=$num_models $hyperparams" "$checkpoint" "$MODE" \
        "$seed" "$MODELS_LABEL/$MODE" "$PORT" "" "$ENV_GROUPS"
      cleanup
      ;;
    fixed-count|until-converged)
      # One server per model; N_ENVS envs sample that model in parallel.
      local name label
      for name in $(model_names); do
        if [[ "$MODE" == "fixed-count" ]]; then
          label="$name/$N_SOLUTIONS-solutions"
        else
          label="$name/until-converged"
        fi
        start_server "$CASE" -1 "$N_ENVS" "$MODELS_DIR/$name" "generalization-$CASE" "$PORT"
        run_inference "$CASE" "num_envs=$N_ENVS $hyperparams" "$checkpoint" "$MODE" \
          "$seed" "$label" "$PORT" "$N_SOLUTIONS" "$ENV_GROUPS"
        cleanup "benchmark/$name/conv"
      done
      ;;
  esac
}

echo "CASE=$CASE MODE=$MODE MODELS=$MODELS"

if [[ -n "$CHECKPOINT" ]]; then
  # Run folders are named <seed>-<timestamp>.
  run_dir=$(basename "$(dirname "$CHECKPOINT")")
  seed="${SEED:-${run_dir%%-*}}"
  [[ "$seed" =~ ^[0-9]+$ ]] || fail "cannot infer the seed from '$run_dir'; pass --seed"
  echo "Sampling with checkpoint $CHECKPOINT (seed $seed)"
  sample_with_checkpoint "$CHECKPOINT" "$seed"
  exit 0
fi

POLICY_ROOT="$AGENT_ROOT/models/rl/$CASE/$POLICY_DIR"
[[ -d "$POLICY_ROOT" ]] || fail "policy folder not found: $POLICY_ROOT"

for subdir in "$POLICY_ROOT"/*/; do
  subdir_name=$(basename "$subdir")
  [[ "$subdir_name" == "sampled_solutions" || "$subdir_name" == solutions* ]] && continue

  read -r run_seed best_overall best_pre100k < <(get_best_from_subdir "$subdir")
  if [[ -z "$best_pre100k" ]]; then
    echo "Skipping $subdir_name: no checkpoint within 100k evaluations" >&2
    continue
  fi
  seed="${SEED:-$run_seed}"
  echo "Sampling with $best_pre100k (seed $seed)"
  sample_with_checkpoint "$CASE/$POLICY_DIR/$best_pre100k" "$seed"
done
