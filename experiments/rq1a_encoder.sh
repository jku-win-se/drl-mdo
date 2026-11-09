#!/bin/bash
# RQ1a: encoder. Usage: experiments/rq1a_encoder.sh <cra|slb|sp>
# Needs the RQ1 dataset, generated once beforehand (see README.md).
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

# Shared by every run below: 5 seeds to convergence, at most 500k FEs, on the RQ1 dataset.
ARGS=(--case "$c" --runs 2 --fes 500000 --n-envs 20 --env-assignment cycle --eval-k 100 --partial_trajectories
  --train-path "models/generated/$c/rq1/seed-12/train-20" --test-path "models/generated/$c/rq1/seed-12/test")
HP="hidden_dim=64 n_steps=100 mini_batch_size=128 n_updates_per_iteration=4 lr=3e-4 clip=0.2 entropy_coef=0.01 \
gamma=0.99 gae_lambda=0.95 value_head=gaussian normalize_rewards=True time_feature=False target_kl=0 \
graph_encoder=sage node_decoder=rnn \
es_tick_episodes=40 es_patience=10"

./train_agent.sh "${ARGS[@]}" --experiment-name rq1a-sage --hyperparams "$HP graph_encoder=sage"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1a-gatv2 --hyperparams "$HP graph_encoder=gatv2"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1a-transformer --hyperparams "$HP graph_encoder=transformer"
