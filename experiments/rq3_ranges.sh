#!/bin/bash
# RQ3: training ranges around the benchmarks, 30 seeds, a new dataset of 40 training models per seed,
# trained to convergence (at most 500k FEs, just above what RQ1 shows convergence takes).
# Ranges run from 0.9 x the lowest to 1.1 x the highest benchmark involved, rounded down.
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

# Chosen after the earlier subexperiments; edit by hand.
case $c in
  cra) SEL="" ;;
  slb) SEL="" ;;
  sp) SEL="" ;;
esac

case $c in
  cra) RANGES="D_B1:16-19 D_B1-B2:16-38 D_B1-B3:16-88 D_B1-B4:16-176" ;;
  slb) RANGES="D_B1:9-11 D_B1-B2:9-27 D_B1-B3:9-55 D_B1-B4:9-110" ;;
  sp) RANGES="D_B1:37-46 D_B1-B2:37-70 D_B1-B3:37-130 D_B1-B4:37-136" ;;
esac

ARGS=(--case "$c" --fes 500000 --n-envs 20 --env-assignment cycle --eval-k 100 --partial_trajectories
  --test-path "models/benchmark/$c")
HP="hidden_dim=64 n_steps=100 mini_batch_size=128 n_updates_per_iteration=4 lr=3e-4 clip=0.2 entropy_coef=0.01 \
gamma=0.99 gae_lambda=0.95 value_head=gaussian normalize_rewards=True time_feature=False target_kl=0 \
graph_encoder=sage node_decoder=rnn \
es_tick_episodes=40 es_patience=10 $SEL"

for range in $RANGES; do
  for i in $(seq 1 30); do
    d="models/generated/$c/rq3-${range%%:*}/seed-$i"
    ./generate_dataset.sh --case "$c" --sizes "${range#*:}" --train 40 --seed "$i" --out "$d" || exit 1
    ./train_agent.sh "${ARGS[@]}" --experiment-name "rq3-${range%%:*}" --run "$i" \
      --train-path "$d/train-40" --hyperparams "$HP" || exit 1
  done
done
