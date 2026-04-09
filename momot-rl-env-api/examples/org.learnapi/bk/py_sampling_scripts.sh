#!/bin/bash

run_inference() {
  local case="$1"
  local custom_hyperparams="$2"
  local ac_model="$3"
  local sample_n_episodes=$4
  local sampling_seeds=$5
  local wandb_group="${6}"
  local port="${7:-8000}"
  local model_path="${8:-models/rl/$case/$wandb_group}"
  

  
  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")
  
  local basepath="/home/win-se-mlcuda01/Desktop/projects/GNN-MT"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/win-semlcuda01/envs/M239/bin/python "${basepath}/main.py" \
    --train \
    --rl \
    --rl_autoreg \
     --rl_sample_n_trajectories "$sample_n_episodes" \
     --sampling_seed $sampling_seeds \
    --rl_hyperparams "encoder=shared rnn_type=gru num_envs=8 render=False render_every_i=10 $custom_hyperparams" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${basepath}/datasets/${case}/config.json" \
    --metamodel "${basepath}/${metamodel_path}" \
    --model_path "${basepath}/${model_path}" \
    --root_object $root_object \
    --hidden_dim 64 \
    --ac_model "${basepath}/models/rl/$ac_model" \
    --port $port
}

run_inference_single() {
  local case="$1"
  local custom_hyperparams="$2"
  local ac_model="$3"
  local sample_n_episodes=$4
  local sampling_seeds=$5
  local wandb_group="${6}"
  local port="${7:-8000}"
  local model_path="${8:-models/rl/$case/$wandb_group}"
  

  
  local metamodel_path
  metamodel_path=$(get_metamodel_path "$case")
  
  local root_object
  root_object=$(get_root_object "$case")
  
  local basepath="/home/win-se-mlcuda01/Desktop/projects/GNN-MT"

  CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/win-semlcuda01/envs/M239/bin/python "${basepath}/main.py" \
    --train \
    --rl \
    --infer_set_single \
    --rl_autoreg \
     --rl_sample_n_trajectories "$sample_n_episodes" \
     --sampling_seed $sampling_seeds \
    --rl_hyperparams "encoder=shared rnn_type=gru num_envs=8 render=False render_every_i=10 $custom_hyperparams" \
    --wandb_config "dir=$case group=$wandb_group" \
    --case_config "${basepath}/datasets/${case}/config.json" \
    --metamodel "${basepath}/${metamodel_path}" \
    --model_path "${basepath}/${model_path}" \
    --root_object $root_object \
    --hidden_dim 64 \
    --ac_model "${basepath}/models/rl/$ac_model" \
    --port $port
}

get_metamodel_path() {
  local case="$1"
  case "$case" in
    slb) echo "data/stack.ecore" ;;
    cra) echo "data/cra.ecore" ;;
    refactoring) echo "data/refactoring.ecore" ;;
    scrum) echo "data/sprintplanning.ecore" ;;
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
    *) echo "" ;;
  esac
}

