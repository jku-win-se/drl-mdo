#!/bin/bash

# run_training CASE SEED HYPERPARAMS FES GROUP MODEL_PATH [main.py flags...]
run_training() {
  local case="$1"
  local seed="$2"
  local custom_hyperparams="$3"
  local fes="${4:-20000}"
  local wandb_group="$5"
  local model_path="${6:-models/rl/$case/$wandb_group}"
  shift 6

  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CONDA_ENV/bin/python" "${AGENT_ROOT}/main.py" \
    --train \
    "$@" \
    --hyperparams "normalize_rewards=False encoder=shared rnn_type=gru entropy_coef=0.01 lr=1e-4 clip=0.2 timesteps_per_batch=2048 gamma=0.99 gae_lambda=0.95 mini_batch_size=64 n_updates_per_iteration=10 hidden_dim=64 $custom_hyperparams" \
    --fes "$fes" \
    --wandb_config "dir=$case group=$wandb_group" \
    --model_path "${AGENT_ROOT}/${model_path}" \
    --seed "$seed"
}

# run_inference CASE CHECKPOINT MODE SEED LABEL [PORT] [NUM_SOLUTIONS] [ENV_GROUPS] [NUM_BATCHES]
#   MODE        one-per-model | fixed-count | until-converged | max-likelihood | batches
#   CHECKPOINT  ac_*.pth or best_model.pth, absolute or relative to $AGENT_ROOT/models/rl
#   LABEL       output group: solutions go to <runs dir>/sampled_solutions/LABEL/...
run_inference() {
  local case="$1"
  local checkpoint="$2"
  local mode="$3"
  local seed="$4"
  local label="$5"
  local port="${6:-8000}"
  local num_solutions="${7:-}"
  local env_groups="${8:-1}"
  local num_batches="${9:-}"

  [[ "$checkpoint" = /* ]] || checkpoint="${AGENT_ROOT}/models/rl/$checkpoint"
  local model_path="models/rl/$case/$label"

  local mode_args=(--infer "$mode")
  [[ -n "$num_solutions" ]] && mode_args+=(--num_solutions "$num_solutions")
  [[ -n "$num_batches" ]] && mode_args+=(--num_batches "$num_batches")

  echo "mode: $mode${num_solutions:+ ($num_solutions solutions)}"
  echo "checkpoint: $checkpoint"
  echo "seed: $seed | port: $port | env groups: $env_groups"
  echo "output label: $label"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CONDA_ENV/bin/python" "${AGENT_ROOT}/main.py" \
    "${mode_args[@]}" \
    --seed "$seed" \
    --hyperparams "" \
    --wandb_config "dir=$case group=$label" \
    --model_path "${AGENT_ROOT}/${model_path}" \
    --ac_model "$checkpoint" \
    --env_groups "$env_groups" \
    --port "$port"
}
