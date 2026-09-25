#!/usr/bin/env bash
# =============================================================================
# Part A: profile every CUDA engine of the ladder on the fixed dataset
# (docs/PROFILING_STRATEGY.md §4).
#
#     scripts/pod/profile_ladder.sh nsys     timelines, ~2 min
#     scripts/pod/profile_ladder.sh ncu      kernel reports, ~5-10 min
#     scripts/pod/profile_ladder.sh all      both
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

mode="${1:-all}"
engines=$(./build/rtm --list-engines | grep '^cuda-v' || true)
[ -n "$engines" ] || { echo "no cuda-v* engines in this build"; exit 1; }

if [ "$mode" = nsys ] || [ "$mode" = all ]; then
    log_step "Part A: Nsight Systems, all shots, fixed dataset"
    for engine in $engines; do scripts/pod/profile_nsys.sh "$engine"; done
fi
if [ "$mode" = ncu ] || [ "$mode" = all ]; then
    log_step "Part A: Nsight Compute, 1 shot x $NCU_NT steps, fixed dataset"
    for engine in $engines; do scripts/pod/profile_ncu.sh "$engine"; done
fi
