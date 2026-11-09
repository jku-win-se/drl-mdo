#!/bin/bash
# RQ1b: hyperparameter sensitivity, from the encoder chosen in RQ1a.
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

# Chosen after the earlier subexperiments; edit by hand.
case $c in
  cra) SEL="" ;;
  slb) SEL="" ;;
  sp) SEL="" ;;
esac

# Shared by every run below: 5 seeds to convergence, at most 500k FEs, on the RQ1 dataset.
ARGS=(--case "$c" --runs 5 --fes 500000 --n-envs 20 --env-assignment cycle --eval-k 100 --partial_trajectories
  --train-path "models/generated/$c/rq1/seed-1/train-40" --test-path "models/generated/$c/rq1/seed-1/test")
HP="hidden_dim=64 n_steps=100 mini_batch_size=128 n_updates_per_iteration=4 lr=3e-4 clip=0.2 entropy_coef=0.01 \
gamma=0.99 gae_lambda=0.95 value_head=gaussian normalize_rewards=True time_feature=False target_kl=0 \
graph_encoder=sage node_decoder=rnn \
es_tick_episodes=40 es_patience=10 $SEL"

./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-h32 --hyperparams "$HP hidden_dim=32"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-h128 --hyperparams "$HP hidden_dim=128"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-l2 --hyperparams "$HP encoder_layers=2"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-l4 --hyperparams "$HP encoder_layers=4"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-mb64 --hyperparams "$HP mini_batch_size=64"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-mb256 --hyperparams "$HP mini_batch_size=256"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-ep10 --hyperparams "$HP n_updates_per_iteration=10"
./train_agent.sh "${ARGS[@]}" --experiment-name rq1b-ep20 --hyperparams "$HP n_updates_per_iteration=20"
