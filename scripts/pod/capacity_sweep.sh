#!/usr/bin/env bash
# =============================================================================
# Sweep S1: capacity + efficiency of our engine vs grid size
# (docs/PROFILING_STRATEGY.md §5, change M7).
#
#     scripts/pod/capacity_sweep.sh [ENGINE ...]        default: cuda-v1 cuda-v3
#
# Needs the scaled datasets (scripts/pod/make_scaled_dataset.sh). For every
# dataset (12.5 m ... 1.25 m) and every engine:
#   1. policy A (one snapshot every 10 time steps, the dataset's STORE_INTERVAL):
#      timed run -> results/scaling.csv
#   2. if it fitted: an Nsight Systems run (memory-usage curve)
#   3. a short Nsight Compute run of the stencil kernel (DRAM %, L2, waves).
#      This works even where the full run is out of memory: 600 steps only.
#   4. at 3.75 m, 2.5 m and 1.25 m: policy B, sparser snapshots every
#      20 / 40 / 80 / 160 steps: which spacing makes each grid fit, and (at
#      3.75 m, where policy A still fits) what accuracy it costs.
# Memory with policy A grows as 1/dx^3 (grid points x snapshots), so the GPU's
# memory ceiling falls between two of the grids; the sparser policies show how
# far it moves.
# Then, per dataset: gate = the second engine's image vs the first engine's.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

engines=("$@")
[ ${#engines[@]} -gt 0 ] || engines=(cuda-v1 cuda-v3)
policy_b_decimations=" 3 2 1 "
policy_b_spacings_steps="20 40 80 160"

for decimation in $SWEEP_DECIMATIONS; do
    env_file="$(scaled_env_file "$decimation")"
    if [ ! -f "$env_file" ]; then
        echo "*** $env_file missing: run scripts/pod/make_scaled_dataset.sh $decimation first"
        continue
    fi
    load_dataset "$env_file"
    log_step "S1 capacity sweep: $DATASET_NAME (dx=$DX m, f0=$F0 Hz, nt=$NT)"
    image_dir="results/images/scaling/$DATASET_NAME"

    for engine in "${engines[@]}"; do
        run_timing_point "$engine" S1 "$STORE_INTERVAL" "$ORDER" "$image_dir/${engine}_every${STORE_INTERVAL}steps.bin"
        if [ "$LAST_STATUS" = ok ]; then
            scripts/pod/profile_nsys.sh "$engine" "$env_file" || true
        fi
        scripts/pod/profile_ncu.sh "$engine" "$env_file" --quick || true

        if [[ "$policy_b_decimations" == *" $decimation "* ]]; then
            for store in $policy_b_spacings_steps; do
                run_timing_point "$engine" S1B "$store" "$ORDER" "$image_dir/${engine}_every${store}steps.bin"
            done
        fi
    done

    # Gate: every engine after the first must reproduce the first one's image.
    first="${engines[0]}"
    for engine in "${engines[@]:1}"; do
        compare_images_row "$image_dir/${first}_every${STORE_INTERVAL}steps.bin" \
            "$image_dir/${engine}_every${STORE_INTERVAL}steps.bin" "${engine}_vs_${first}"
    done
    # Accuracy cost of sparser snapshots, where policy A still fits.
    if [[ "$policy_b_decimations" == *" $decimation "* ]]; then
        for store in $policy_b_spacings_steps; do
            compare_images_row "$image_dir/${first}_every${STORE_INTERVAL}steps.bin" \
                "$image_dir/${first}_every${store}steps.bin" "${first}_every${store}steps_vs_every${STORE_INTERVAL}steps"
        done
    fi
done

log_step "S1 done: results/scaling.csv, results/compare_scaling.csv, $PROFILES_DIR/marmousi_dx*/"
