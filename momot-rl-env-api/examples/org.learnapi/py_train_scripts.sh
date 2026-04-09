#!/bin/bash

run_training_partial() {
  local case="$1"
  local seed=$2
  local custom_hyperparams="$3"
  local rl_tsteps="${4:-20000}"
  local wandb_group="${5}"
  local model_path="${6:-models/rl/$case/$wandb_group}"
  local rl_evaluate="${7:""}"

  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")

  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CONDA_ENV/bin/python" "${AGENT_ROOT}/main.py" \
    --train \
    --rl \
    --rl_autoreg \
    --rl_sample_partial_trajectories True \ 
    --rl_hyperparams "normalize_rewards=False encoder=shared rnn_type=gru entropy_coef=0.01 lr=1e-4 clip=0.2 timesteps_per_batch=2048 gamma=0.99 gae_lambda=0.95 mini_batch_size=64 n_updates_per_iteration=10 render=False render_every_i=10 $custom_hyperparams" \
    --rl_tsteps "$rl_tsteps" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${AGENT_ROOT}/datasets/${case}/config.json" \
    --metamodel "${AGENT_ROOT}/${metamodel_path}" \
    --root_object $root_object \
    --model_path "${AGENT_ROOT}/${model_path}" \
    --batch_size 8 \
    --hidden_dim 64 \
    --epochs 200 \
    --seed $seed
}

run_training() {
  local case="$1"
  local seed=$2
  local custom_hyperparams="$3"
  local rl_tsteps="${4:-20000}"
  local wandb_group="${5}"
  local model_path="${6:-models/rl/$case/$wandb_group}"

  shift 6
  local sample_partial_trajectories=false
  local rl_evaluate=false
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --sample_partial_trajectories) sample_partial_trajectories=true ;;
            --evaluate) rl_evaluate=true ;;
        esac
        shift
    done

  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")

  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CONDA_ENV/bin/python" "${AGENT_ROOT}/main.py" \
    --train \
    --rl \
    --rl_autoreg \
    $($sample_partial_trajectories && echo "--rl_sample_partial_trajectories") \
     $($rl_evaluate && echo "--rl_evaluate") \
   --rl_hyperparams "normalize_rewards=False encoder=shared rnn_type=gru entropy_coef=0.01 lr=1e-4 clip=0.2 timesteps_per_batch=2048 gamma=0.99 gae_lambda=0.95 mini_batch_size=64 n_updates_per_iteration=10 render=False render_every_i=10 $custom_hyperparams" \
    --rl_tsteps "$rl_tsteps" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${AGENT_ROOT}/datasets/${case}/config.json" \
    --metamodel "${AGENT_ROOT}/${metamodel_path}" \
    --root_object $root_object \
    --model_path "${AGENT_ROOT}/${model_path}" \
    --hidden_dim 64 \
    --seed $seed
}

run_inference() {
  local case="$1"
  local custom_hyperparams="$2"
  local ac_model="$3"
  local sample_n_episodes=$4
  local seed=$5
  local wandb_group="${6}"
  local port="${7:-8000}"

  local model_path="models/rl/$case/$wandb_group"
  
  
  shift 7
  local infer_set_single=false
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --infer-single-solution) infer_set_single=true ;;
        esac
        shift
    done

  
  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")

  echo "infer_set_single: $infer_set_single"
  echo "wandb_group: $wandb_group"
  echo "model_path: $model_path"
  echo "ac_model: $ac_model"
  echo "sample_n_episodes: $sample_n_episodes"
  echo "seed: $seed"
  echo "port: $port"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CONDA_ENV/bin/python" "${AGENT_ROOT}/main.py" \
    --train \
    --rl \
     $($infer_set_single && echo "--infer_set_single") \
    --rl_autoreg \
    --rl_sample_n_trajectories "$sample_n_episodes" \
     --seed $seed \
    --rl_hyperparams "encoder=shared rnn_type=gru num_envs=8 render=False render_every_i=10 $custom_hyperparams" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${AGENT_ROOT}/datasets/${case}/config.json" \
    --metamodel "${AGENT_ROOT}/${metamodel_path}" \
    --model_path "${AGENT_ROOT}/${model_path}" \
    --root_object $root_object \
    --hidden_dim 64 \
    --ac_model "${AGENT_ROOT}/models/rl/$ac_model" \
    --port $port
}

get_metamodel_path() {
  local case="$1"
  case "$case" in
    slb) echo "data/stack.ecore" ;;
    cra) echo "data/cra.ecore" ;;
    refactoring) echo "data/refactoring.ecore" ;;
    sp) echo "data/sprintplanning.ecore" ;;
    sp_uc) echo "data/sprintplanning.ecore" ;;
    *) echo "" ;;
  esac
}

get_root_object() {
  local case="$1"
  case "$case" in
    slb) echo "StackModel" ;;
    cra) echo "ClassModel" ;;
    refactoring) echo "RefactoringModel" ;;
    sp) echo "Plan" ;;
    sp_uc) echo "Plan" ;;
    *) echo "" ;;
  esac
}

