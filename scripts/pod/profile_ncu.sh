#!/usr/bin/env bash
# =============================================================================
# Nsight Compute reports of ONE engine on ONE dataset (docs/PROFILING_STRATEGY.md §4.4).
#
#     scripts/pod/profile_ncu.sh ENGINE [DATASET_ENV] [--quick] [--order N]
#
# Full mode (the CUDA ladder on the fixed dataset), per engine:
#   <engine>_stencil.ncu-rep       main stencil kernel, --set full, cold caches
#   <engine>_stencil_warm.ncu-rep  same kernel, caches NOT flushed (SOL + memory only)
#   <engine>_imaging.ncu-rep       imaging kernel (fused or standalone), --set full
#   cuda-v0_sponge.ncu-rep         the separate sponge kernel, v0 only
# --quick (scaling sweeps): only <engine>[_orderN]_stencil.ncu-rep.
#
# Every report also gets <name>_raw.csv, <name>_details.txt and <name>.meta.
# Each capture: 1 shot, NCU_NT time steps, skip the first launches (warm-up),
# then profile 3 launches of the chosen kernel.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

engine="${1:?usage: $0 ENGINE [DATASET_ENV] [--quick] [--order N]}"; shift
env_file="$FIXED_DATASET_ENV"; quick=0; order_override=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) quick=1; shift ;;
        --order) order_override="$2"; shift 2 ;;
        *)       env_file="$1"; shift ;;
    esac
done

ncu_counters_available || exit 0
load_dataset "$env_file"
order="${order_override:-$ORDER}"
out_dir="$PROFILES_DIR/$DATASET_NAME/ncu"
mkdir -p "$out_dir" results/images/profile
name_prefix="$engine${order_override:+_order$order}"

# capture NAME KERNEL SKIP CACHE_MODE [ncu section flags...]
capture() {
    local capture_name="$1" kernel="$2" skip="$3" cache_mode="$4"; shift 4
    local report="$out_dir/${name_prefix}_${capture_name}.ncu-rep"
    echo "--- ncu: ${name_prefix} ${capture_name} ($kernel, skip $skip, cache $cache_mode)"
    ncu "$@" --import-source yes --cache-control "$cache_mode" \
        --kernel-name "$(kernel_regex "$kernel")" --launch-skip "$skip" --launch-count 3 \
        --export "${report%.ncu-rep}" --force-overwrite \
        ./build/rtm --engine "$engine" "${RTM_DATA_ARGS[@]}" --order "$order" \
        --output "results/images/profile/ncu_${engine}.bin" \
        --nt "$NCU_NT" --max-shots 1 --quiet \
        > "$out_dir/${name_prefix}_${capture_name}.log" 2>&1 || true
    if [ ! -s "$report" ]; then
        echo "    *** no report (kernel not matched or run failed), see $out_dir/${name_prefix}_${capture_name}.log"
        tail -3 "$out_dir/${name_prefix}_${capture_name}.log"
        return 0
    fi
    export_ncu_report "$report"
    write_ncu_meta "$report" "$engine" "$capture_name" "$cache_mode" "$order"
}

stencil="$(stencil_kernel_of "$engine")"
[ -n "$stencil" ] || { echo "no stencil kernel known for engine '$engine'"; exit 1; }

capture stencil "$stencil" 200 all --set full
if [ "$quick" = 0 ]; then
    capture stencil_warm "$stencil" 200 none \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats
    # Imaging runs only on stored backward steps (1 in STORE_INTERVAL).
    capture imaging "$(imaging_kernel_of "$engine")" 5 all --set full
    [ "$engine" = cuda-v0 ] && capture sponge k_sponge 200 all --set full
fi
exit 0
