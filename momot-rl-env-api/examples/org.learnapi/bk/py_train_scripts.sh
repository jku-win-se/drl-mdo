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
  
  local basepath="/home/win-se-mlcuda01/Desktop/projects/GNN-MT"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/win-semlcuda01/envs/M239/bin/python "${basepath}/main.py" \
    --train \
    --rl \
    --rl_autoreg \
    --rl_evaluate \
    --rl_sample_partial_trajectories True \ 
    --rl_hyperparams "normalize_rewards=False encoder=shared rnn_type=gru entropy_coef=0.01 lr=1e-4 clip=0.2 timesteps_per_batch=2048 gamma=0.99 gae_lambda=0.95 mini_batch_size=64 n_updates_per_iteration=10 render=False render_every_i=10 $custom_hyperparams" \
    --rl_tsteps "$rl_tsteps" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${basepath}/datasets/${case}/config.json" \
    --metamodel "${basepath}/${metamodel_path}" \
    --root_object $root_object \
    --model_path "${basepath}/${model_path}" \
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
  local rl_evaluate="${7:""}"

  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")
  
  local basepath="/home/win-se-mlcuda01/Desktop/projects/GNN-MT"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/win-semlcuda01/envs/M239/bin/python "${basepath}/main.py" \
    --train \
    --rl \
    --rl_autoreg \
    $rl_evaluate \
    --rl_hyperparams "normalize_rewards=False encoder=shared rnn_type=gru timesteps_per_batch=576 mini_batch_size=144 num_envs=8 gamma=0.99 gae_lambda=0.95 entropy_coef=0.001 n_updates_per_iteration=10 lr=1e-4 clip=0.2 render=False render_every_i=10 $custom_hyperparams" \
    --rl_tsteps "$rl_tsteps" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${basepath}/datasets/${case}/config.json" \
    --metamodel "${basepath}/${metamodel_path}" \
    --root_object $root_object \
    --model_path "${basepath}/${model_path}" \
    --batch_size 8 \
    --hidden_dim 64 \
    --epochs 200 \
    --seed $seed
}

get_metamodel_path() {
  local case="$1"
  case "$case" in
    slb) echo "data/stack.ecore" ;;
    cra) echo "data/cra.ecore" ;;
    refactoring) echo "data/refactoring.ecore" ;;
    scrum) echo "data/sprintplanning.ecore" ;;
    scrum_uc) echo "data/sprintplanning.ecore" ;;
    *) echo "" ;;
  esac
}

get_root_object() {
  local case="$1"
  case "$case" in
    slb) echo "StackModel" ;;
    cra) echo "ClassModel" ;;
    refactoring) echo "RefactoringModel" ;;
    scrum) echo "Plan" ;;
    scrum_uc) echo "Plan" ;;
    *) echo "" ;;
  esac
}

