#!/bin/bash --login
# Train one setup for N seeds, one seed at a time. See experiments/README.md for examples.
# Datasets are given, never generated here (see generate_dataset.sh): --train-path is required,
# --val-path turns on periodic validation and --test-path the final run on the test dataset.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$SCRIPT_DIR/server-executables"
AGENT_ROOT="$(cd "$SCRIPT_DIR/GNN-MT" && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: .env not found in $SCRIPT_DIR" >&2
    exit 1
fi

set -a
source "$SCRIPT_DIR/.env"
set +a

source "$PROJECT_DIR/py_train_scripts.sh"
source "$PROJECT_DIR/java_environment_server.sh"

# --runs N uses the first N of these seeds, --run I only the I-th. Append to the list for more runs.
SEEDS=(42 17 92 5 88 101 7 66 23 55 19 4 78 29 12 84 31 9 40 64 90 73 38 26 13 97 3 21 60 15)

# Defaults
RUNS=30
RUN=""
FES=100000
N_ENVS=10
HYPERPARAMS=""
TRAIN_PATH=""
VAL_PATH=""
TEST_PATH=""
EVAL_K=1
ENV_ASSIGNMENT=fixed
PY_FLAGS=()
ALLOW_TRAIN_AS_TEST=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --case) CASE="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --run) RUN="$2"; shift ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift ;;
    --fes) FES="$2"; shift ;;
    --n-envs) N_ENVS="$2"; shift ;;
    --hyperparams) HYPERPARAMS="$2"; shift ;;
    --train-path) TRAIN_PATH="$2"; shift ;;
    --val-path) VAL_PATH="$2"; shift ;;
    --test-path) TEST_PATH="$2"; shift ;;
    --eval-k) EVAL_K="$2"; shift ;;
    --env-assignment) ENV_ASSIGNMENT="$2"; shift ;;
    --partial_trajectories) PY_FLAGS+=(--partial_trajectories) ;;
    --eval-mode) PY_FLAGS+=(--eval_mode "$2"); shift ;;
    --eval-interval) PY_FLAGS+=(--eval_interval "$2"); shift ;;
    --eval-save-solutions) PY_FLAGS+=(--eval_save_solutions) ;;
    --keep-checkpoints) PY_FLAGS+=(--keep_checkpoints "$2"); shift ;;
    --allow-train-as-test) ALLOW_TRAIN_AS_TEST=1 ;;
    *) echo "Unknown parameter: $1" >&2; exit 1 ;;
  esac
  shift
done

fail() {
  echo "Error: $1" >&2
  exit 1
}

[[ -n "$CASE" ]] || fail "--case is required"
[[ -n "$EXPERIMENT_NAME" ]] || fail "--experiment-name is required"
[[ -n "$TRAIN_PATH" ]] || fail "--train-path is required"
if [[ -n "$RUN" ]]; then
  [[ "$RUN" -ge 1 && "$RUN" -le "${#SEEDS[@]}" ]] ||
    fail "--run $RUN is outside the ${#SEEDS[@]} stored seeds; append to SEEDS"
  FIRST=$((RUN - 1))
  RUNS=$RUN
else
  FIRST=0
  [[ "$RUNS" -le "${#SEEDS[@]}" ]] ||
    fail "--runs $RUNS asks for more seeds than the ${#SEEDS[@]} stored; append to SEEDS"
fi
[[ -n "$TEST_PATH" ]] && PY_FLAGS+=(--final_eval test)
# Runs share the machine with nothing else, so their runtimes stay clean.
if nc -z localhost 8000 2>/dev/null; then
  fail "port 8000 is in use, so another run seems active; runs execute one at a time"
fi

# Dataset paths are relative to the repository root; the server runs from server-executables/.
absolute() {
  [[ -z "$1" || "$1" = /* ]] && echo "$1" || echo "$SCRIPT_DIR/$1"
}

mkdir -p "$PROJECT_DIR/logs"
RUN_DIR="$AGENT_ROOT/models/rl/$CASE/$EXPERIMENT_NAME"
LOGFILE="$PROJECT_DIR/logs/${CASE}_${EXPERIMENT_NAME}_$(date '+%Y%m%d_%H%M%S').log"
echo "CASE=$CASE RUNS=$RUNS FES=$FES N_ENVS=$N_ENVS => $RUN_DIR (log: $LOGFILE)"

train="$(absolute "$TRAIN_PATH")"
val="$(absolute "$VAL_PATH")"
test="$(absolute "$TEST_PATH")"
for path in "$train" "$val" "$test"; do
  [[ -z "$path" || -e "$path" ]] || fail "dataset $path does not exist"
done
if [[ -n "$test" && -z "$ALLOW_TRAIN_AS_TEST" ]]; then
  for other in "$train" "$val"; do
    [[ -n "$other" && "$(realpath -m "$other")" == "$(realpath -m "$test")" ]] &&
      fail "the test dataset must differ from the train and validation datasets (--allow-train-as-test)"
  done
fi

for ((i = FIRST; i < RUNS; i++)); do
  SEED=${SEEDS[$i]}
  if compgen -G "$RUN_DIR/$SEED-*/runtimes.json" > /dev/null; then
    echo "=== Run $((i + 1)) of $RUNS (seed $SEED): already complete, skipping ==="
    continue
  fi
  echo "=== Run $((i + 1)) of $RUNS (seed $SEED) ==="

  start_server "$CASE" "$N_ENVS" "$train" "${val:-NONE}" "${test:-NONE}" "$CASE/$EXPERIMENT_NAME" \
    8000 "$EVAL_K" "$ENV_ASSIGNMENT"
  run_training "$CASE" "$SEED" "$HYPERPARAMS" "$FES" "$EXPERIMENT_NAME" \
    "models/rl/$CASE/$EXPERIMENT_NAME" "${PY_FLAGS[@]}" >> "$LOGFILE" 2>&1
  status=$?
  cleanup >> "$LOGFILE" 2>&1
  [[ "$status" -eq 0 ]] || fail "seed $SEED failed, see $LOGFILE"
done
