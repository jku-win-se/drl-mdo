#!/bin/bash
# Generate a dataset of case models: train-N/ per --train size (smaller sets are contained in larger
# ones), val/, test/ and manifest.csv. Models identical to a benchmark of the case, or to another
# generated model in any split, are rejected and redrawn. Skips if DIR already holds a dataset.
#
# generate_dataset.sh --case <cra|slb|sp> [--sizes MIN-MAX] --train 20,40 [--val N] [--test N] \
#   --seed S --out DIR [--exclude FOLDER]...
# Without --sizes, the case's size range in ModelGenerator.java applies; its other model
# characteristics (property ranges, SP stakeholders) are constants there too.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$SCRIPT_DIR/server-executables"

fail() {
  echo "Error: $1" >&2
  exit 1
}

[[ -f "$SCRIPT_DIR/.env" ]] || fail ".env not found in $SCRIPT_DIR"
set -a
source "$SCRIPT_DIR/.env"
set +a
source "$PROJECT_DIR/java_environment_server.sh"
trap - EXIT INT TERM  # Starts no server, so nothing to clean up.

VAL=0
TEST=0
EXCLUDE=()
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --case) CASE="$2"; shift ;;
    --sizes) SIZES="$2"; shift ;;
    --train) TRAIN="$2"; shift ;;
    --val) VAL="$2"; shift ;;
    --test) TEST="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --exclude) EXCLUDE+=("$2"); shift ;;
    *) fail "unknown parameter: $1" ;;
  esac
  shift
done

for flag in CASE TRAIN SEED OUT; do
  [[ -n "${!flag}" ]] || fail "--${flag,,} is required"
done

# Paths are relative to the repository root.
cd "$SCRIPT_DIR" || exit 1
if [[ -f "$OUT/manifest.csv" ]]; then
  echo "Dataset $OUT exists, skipping"
  exit 0
fi

exclude_args=(--exclude "models/benchmark/$CASE")
for folder in "${EXCLUDE[@]}"; do
  exclude_args+=(--exclude "$folder")
done

size_args=()
[[ -n "$SIZES" ]] && size_args=(--sizes "$SIZES")

echo "Generating dataset $OUT"
"$JAVA_BIN" -cp "$SERVER_CLASSPATH" learnapi.ModelGenerator --case "$(java_case "$CASE")" \
  "${size_args[@]}" --train "$TRAIN" --val "$VAL" --test "$TEST" --seed "$SEED" \
  "${exclude_args[@]}" --out "$OUT" || fail "generating $OUT failed"
