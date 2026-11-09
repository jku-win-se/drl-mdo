#!/bin/bash
# RQ1f: how HV grows with continued sampling. Trains nothing: rq1c-masking-on's stored policies
# sample 10 batches of 100 solutions on each held-out model.
c=${1:?usage: $0 <cra|slb|sp>}
cd "$(dirname "$0")/.." || exit 1

./run_policy.sh --case "$c" --mode batches --n-batches 10 --n-envs 100 --label heldout \
  --policy-dir rq1c-masking-on --models "models/generated/$c/rq1/seed-1/test"
