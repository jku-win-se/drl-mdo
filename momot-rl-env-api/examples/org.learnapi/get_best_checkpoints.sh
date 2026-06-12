#!/usr/bin/env bash
# --------------------------------------------------
# Function: get_best_from_subdir SUBDIR
# Outputs: seed best_overall_file best_pre100k_file
# --------------------------------------------------
get_best_from_subdir() {
    local subdir="$1"
    local parent_dir="$(dirname "$subdir")"   # parent of the subdirectory
    local dirname=$(basename "$subdir")
    local seed="${dirname%%-*}"
    local best_overall_file=""
    local best_overall_reward=""
    local best_pre100k_file=""
    local best_pre100k_reward=""
    # Loop only over files starting with ac_
    for ckpt in "$subdir"/ac_*.pth; do
        [[ -e "$ckpt" ]] || continue
        local fname=$(basename "$ckpt")
        # Use regex to extract iteration, evals, reward (with optional negative sign)
        if [[ "$fname" =~ ^ac_([0-9]+)-([0-9]+)-(-?[0-9.]+)\.pth$ ]]; then
            local iter="${BASH_REMATCH[1]}"
            local evals="${BASH_REMATCH[2]}"
            local reward="${BASH_REMATCH[3]}"
        else
            # Skip files not matching the pattern
            continue
        fi
        # Compare rewards using bc (float-safe)
        # For first file or if current reward is better
        if [[ -z "$best_overall_reward" ]] || [[ $(echo "$reward > $best_overall_reward" | bc -l) -eq 1 ]]; then
            best_overall_reward="$reward"
            best_overall_file="$ckpt"
        fi
        if (( evals <= 100000 )); then
            if [[ -z "$best_pre100k_reward" ]] || [[ $(echo "$reward > $best_pre100k_reward" | bc -l) -eq 1 ]]; then
                best_pre100k_reward="$reward"
                best_pre100k_file="$ckpt"
            fi
        fi
    done
    # Convert absolute paths to relative paths from parent_dir
    if [[ -n "$best_overall_file" ]]; then
        best_overall_file="${best_overall_file#$parent_dir/}"
    fi
    if [[ -n "$best_pre100k_file" ]]; then
        best_pre100k_file="${best_pre100k_file#$parent_dir/}"
    fi
    
    echo "$seed $best_overall_file $best_pre100k_file"
}
