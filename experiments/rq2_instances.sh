#!/bin/bash
# RQ2: one policy per benchmark instance, 30 seeds, exactly 100k FEs.
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

# Chosen after the earlier subexperiments; edit by hand.
case $c in
  cra) SEL="" ;;
  slb) SEL="" ;;
  sp) SEL="" ;;
esac

ARGS=(--case "$c" --runs 30 --fes 100000 --n-envs 20 --env-assignment cycle --eval-k 100 --partial_trajectories)
HP="hidden_dim=64 n_steps=100 mini_batch_size=128 n_updates_per_iteration=4 lr=3e-4 clip=0.2 entropy_coef=0.01 \
gamma=0.99 gae_lambda=0.95 value_head=gaussian normalize_rewards=True time_feature=False target_kl=0 \
graph_encoder=sage node_decoder=rnn \
es_patience=0 $SEL"

for m in models/benchmark/"$c"/*.xmi; do
  ./train_agent.sh "${ARGS[@]}" --experiment-name "rq2-$(basename "$m" .xmi)" \
    --train-path "$m" --test-path "$m" --allow-train-as-test --hyperparams "$HP"
done
