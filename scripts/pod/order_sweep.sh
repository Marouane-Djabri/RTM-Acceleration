#!/usr/bin/env bash
# =============================================================================
# Sweep S2: stencil order at a fixed large grid (docs/PROFILING_STRATEGY.md §5.2).
#
#     scripts/pod/order_sweep.sh [ENGINE] [ORDERS]      default: cuda-v1 "4 8 12"
#
# dx = 2.5 m (fills the GPU, throughput on its plateau). For each order: a
# timed run -> results/scaling.csv (sweep S2) and a short Nsight Compute run
# of the stencil -> <engine>_order<N>_stencil.ncu-rep (the D11 roofline points).
# Our engine supports orders 2..12. The shots stay the order-8 files.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

engine="${1:-cuda-v1}"
orders="${2:-4 8 12}"
env_file="$(scaled_env_file 2)"
[ -f "$env_file" ] || { echo "$env_file missing: run scripts/pod/make_scaled_dataset.sh 2 first"; exit 1; }
load_dataset "$env_file"
log_step "S2 order sweep: $engine on $DATASET_NAME, orders $orders"

for order in $orders; do
    run_timing_point "$engine" S2 "$STORE_INTERVAL" 10 "$order" \
        "results/images/scaling/$DATASET_NAME/${engine}_order${order}.bin"
    scripts/pod/profile_ncu.sh "$engine" "$env_file" --quick --order "$order" || true
done
