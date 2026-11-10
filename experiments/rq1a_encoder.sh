#!/bin/bash
# RQ1a: encoder (sage, gatv2) x training-set size (20, 40, 60). Usage: experiments/rq1a_encoder.sh <cra|slb|sp>
# Needs the RQ1 dataset, generated once beforehand (see README.md).
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

# Shared by every run below: 5 seeds, each until 10 validation phases in a row bring no better
# validation reward (cap 1M FEs). Validation every 10 iterations with 50 episodes per model, its
# solutions saved; the best policy by validation reward then runs on the test set, sampled
# (50 episodes per model) and greedy.
D="models/generated/$c/rq1/seed-1"
ARGS=(--case "$c" --runs 5 --fes 1000000 --n-envs 20 --env-assignment cycle --eval-k 50 --partial_trajectories
  --val-path "$D/val" --test-path "$D/test" --eval-interval 10 --eval-save-solutions
  --select-on val --final-mode both)
HP="hidden_dim=64 n_steps=100 mini_batch_size=128 n_updates_per_iteration=4 lr=3e-4 clip=0.2 entropy_coef=0.01 \
gamma=0.99 gae_lambda=0.95 value_head=gaussian normalize_rewards=True time_feature=False target_kl=0 \
graph_encoder=sage node_decoder=rnn \
es_patience=10 es_min_timesteps=0"

for n in 20 40 60; do
  for enc in sage gatv2; do
    ./train_agent.sh "${ARGS[@]}" --train-path "$D/train-$n" --experiment-name "rq1a-$enc-n$n" \
      --hyperparams "$HP graph_encoder=$enc"
  done
done
