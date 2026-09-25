#!/usr/bin/env bash
# =============================================================================
# Nsight Systems timeline of ONE engine on ONE dataset (docs/PROFILING_STRATEGY.md §4.2).
#
#     scripts/pod/profile_nsys.sh ENGINE [DATASET_ENV] [TAG] [-- extra rtm args]
#
#   DATASET_ENV  default scripts/dataset_marmousi.env (the fixed marmousi12 dataset)
#   TAG          appended to the report name, e.g. "snap40ms" (default: none)
#
# Writes results/profiles/<dataset>/nsys/<engine>[_TAG].nsys-rep plus the
# summary tables as CSV next to it (<name>_cuda_gpu_kern_sum.csv, ...).
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

engine="${1:?usage: $0 ENGINE [DATASET_ENV] [TAG] [-- extra rtm args]}"; shift
env_file="$FIXED_DATASET_ENV"; tag=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then env_file="$1"; shift; fi
if [ $# -gt 0 ] && [ "$1" != "--" ]; then tag="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
extra_args=("$@")

load_dataset "$env_file"
out_dir="$PROFILES_DIR/$DATASET_NAME/nsys"
mkdir -p "$out_dir" results/images/profile
name="$engine${tag:+_$tag}"
report="$out_dir/$name.nsys-rep"

rtm_command=(./build/rtm --engine "$engine" "${RTM_DATA_ARGS[@]}"
             --output "results/images/profile/${DATASET_NAME}_${name}.bin"
             --dataset "$DATASET_NAME" --benchmark-csv "$out_dir/runs_under_nsys.csv"
             --quiet "${extra_args[@]}")
nsys_flags=(--output "${report%.nsys-rep}" --force-overwrite true
            --trace cuda,nvtx,osrt --cuda-memory-usage true --stats false)

echo "--- nsys: $name on $DATASET_NAME"
gpu_metrics_flag="$(nsys_gpu_metrics_flag)"
if [ -n "$gpu_metrics_flag" ] && nsys profile "${nsys_flags[@]}" "$gpu_metrics_flag" "${rtm_command[@]}" \
        > "$out_dir/$name.log" 2>&1 && [ -s "$report" ]; then
    echo "    captured with GPU metrics"
else
    # GPU metrics need the same counter permission as ncu: retry without them.
    nsys profile "${nsys_flags[@]}" "${rtm_command[@]}" > "$out_dir/$name.log" 2>&1 || true
    [ -s "$report" ] || { echo "    *** nsys failed, see $out_dir/$name.log"; tail -5 "$out_dir/$name.log"; exit 1; }
    echo "    captured (no GPU metrics rows: counters not available)"
fi
export_nsys_stats "$report"
echo "    $report"
