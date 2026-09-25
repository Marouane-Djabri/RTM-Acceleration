#!/usr/bin/env bash
# =============================================================================
# EVERYTHING, IN ORDER, ON THE POD  (docs/PROFILING_STRATEGY.md §8)
#
#     ssh pod "cd /workspace/RTM && nohup scripts/pod/run_all.sh > run_all.out 2>&1 &"
#     ssh pod "tail -f /workspace/RTM/run_all.out"          # watch it
#
# Options
#   --skip-devito     phases 9-11 are skipped (no NVIDIA HPC SDK install)
#   --only PHASE      run just this phase (name or number), ignoring done-markers
#   --from PHASE      start at this phase, ignoring done-markers from there on
#   --list            print the phases and exit
#
# Each finished phase leaves results/profiles/.phases/<name>.done, so after a
# crash or a pod restart simply run the script again: finished phases are
# skipped. A failed phase is recorded and the script moves on; only a failed
# build stops everything. A status table is printed at the end.
#
# Rough time on an RTX 3090-class pod: ~2.5-3.5 h, of which ~30 min is the
# one-time CPU reference (skipped if results/ref/marmousi_ref.bin exists)
# and ~15 min the NVIDIA HPC SDK install for Devito.
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/common.sh"

PHASES=(
    "1:preflight:tool versions, rebuild, GPU counter permission test"
    "2:setup:build + dataset checksum + CPU reference (scripts/pod/setup.sh)"
    "3:bench:timing + gate of every engine on marmousi12 (bench_all.sh)"
    "4:nsys_ladder:Part A - Nsight Systems timelines, cuda-v0..v4"
    "5:ncu_ladder:Part A - Nsight Compute kernel reports, cuda-v0..v4"
    "6:datasets:scaled Marmousi datasets dx 12.5 .. 1.25 m (shots modelled on GPU)"
    "7:capacity:sweep S1 + snapshot policy B, cuda-v1 and cuda-v3"
    "8:orders:sweep S2, stencil orders 4 8 12, cuda-v1"
    "9:devito_setup:NVIDIA HPC SDK + Devito GPU venv"
    "10:devito_grid:Devito on sweep S1"
    "11:devito_order:Devito on sweep S2, orders 4 8 12 16"
    "12:report:tables + charts (profile_extract.py, project_survey.py)"
)

skip_devito=0; only=""; from=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-devito) skip_devito=1; shift ;;
        --only) only="$2"; shift 2 ;;
        --from) from="$2"; shift 2 ;;
        --list) printf '%s\n' "${PHASES[@]}" | tr ':' '\t'; exit 0 ;;
        *) echo "unknown option $1 (see the header of $0)"; exit 1 ;;
    esac
done

source "$FIXED_DATASET_ENV"     # REFERENCE_IMAGE
marker_dir="$PROFILES_DIR/.phases"
mkdir -p "$marker_dir"
declare -A STATUS

run_phase_commands() {
    case "$1" in
        preflight)
            {
                date
                nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
                echo "nsys: $(nsys --version 2>&1 | head -1)"
                echo "ncu:  $(ncu --version 2>&1 | tail -1)"
                echo "nvcc: $(nvcc --version | tail -1)"
                echo "cpu:  $(nproc) cores"
            } | tee "$PROFILES_DIR/TOOLS.txt"
            echo ">>> Your Nsight GUIs must be at least these nsys/ncu versions to open the reports."
            cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null &&
            cmake --build build -j"$(nproc)" || return 1
            rm -f "$PROFILES_DIR/ncu_permission.txt"
            if ncu_counters_available; then echo "GPU counters: available (Nsight Compute steps will run)"; fi
            return 0 ;;
        setup)        scripts/pod/setup.sh ;;
        bench)        scripts/pod/bench_all.sh ;;
        nsys_ladder)  scripts/pod/profile_ladder.sh nsys ;;
        ncu_ladder)   scripts/pod/profile_ladder.sh ncu ;;
        datasets)
            local failed=0
            for decimation in $SWEEP_DECIMATIONS; do
                scripts/pod/make_scaled_dataset.sh "$decimation" || failed=1
            done
            return $failed ;;
        capacity)     scripts/pod/capacity_sweep.sh cuda-v1 cuda-v3 ;;
        orders)       scripts/pod/order_sweep.sh cuda-v1 "4 8 12" ;;
        devito_setup) scripts/pod/setup_devito_gpu.sh ;;
        devito_grid)  scripts/pod/devito_sweep.sh grid ;;
        devito_order) scripts/pod/devito_sweep.sh order ;;
        report)
            python3 -c "import matplotlib" 2>/dev/null || python3 -m pip install -q matplotlib
            python3 scripts/profile_extract.py &&
            python3 scripts/project_survey.py --survey-shots 1000 --dx 6.25 &&
            python3 scripts/project_survey.py --survey-shots 1000 --dx 2.5 ;;
    esac
}

started=0
[ -z "$from" ] && started=1
for entry in "${PHASES[@]}"; do
    IFS=: read -r number name description <<<"$entry"
    if [ -n "$only" ] && [ "$only" != "$number" ] && [ "$only" != "$name" ]; then continue; fi
    if [ -n "$from" ] && { [ "$from" = "$number" ] || [ "$from" = "$name" ]; }; then started=1; fi
    if [ "$started" = 0 ] && [ -z "$only" ]; then STATUS[$name]="not run (before --from)"; continue; fi
    if [ "$skip_devito" = 1 ] && [[ "$name" == devito_* ]]; then STATUS[$name]="skipped (--skip-devito)"; continue; fi
    if [ -z "$only" ] && [ -z "$from" ] && [ -f "$marker_dir/$name.done" ]; then
        STATUS[$name]="done earlier"; continue
    fi
    if [[ "$name" == devito_grid || "$name" == devito_order ]] && [ ! -f scripts/pod/devito_gpu.env ]; then
        STATUS[$name]="skipped (devito_setup did not finish)"; continue
    fi
    # Only the bench phase needs the CPU reference image (every engine is gated against it).
    if [ "$name" = bench ] && [ ! -s "$REFERENCE_IMAGE" ]; then
        STATUS[$name]="skipped (no CPU reference image, see phase setup)"; continue
    fi

    echo
    echo "################################################################################"
    echo "# PHASE $number/12  $name: $description"
    echo "# started $(date '+%H:%M:%S')"
    echo "################################################################################"
    phase_start=$SECONDS
    if run_phase_commands "$name"; then
        touch "$marker_dir/$name.done"
        STATUS[$name]="OK ($(( (SECONDS - phase_start) / 60 )) min)"
    else
        STATUS[$name]="FAILED ($(( (SECONDS - phase_start) / 60 )) min)"
        # Without a build nothing can run. Anything else (e.g. the 15-30 min CPU
        # reference at the end of setup) only costs the phases that need it.
        if [ "$name" = preflight ] || [ ! -x build/rtm ]; then
            echo "*** $name failed and there is no working build (build/rtm). Stopping."
            break
        fi
        if [ "$name" = setup ] && [ ! -s "$REFERENCE_IMAGE" ]; then
            echo "*** setup failed before the CPU reference image was written ($REFERENCE_IMAGE)."
            echo "*** Continuing: only phase 3 (bench) needs it. Look above for the error, then run"
            echo "***     scripts/pod/run_all.sh --only setup && scripts/pod/run_all.sh --only bench"
        fi
    fi
done

echo
echo "================================ SUMMARY ======================================="
for entry in "${PHASES[@]}"; do
    IFS=: read -r number name description <<<"$entry"
    printf "  %2s  %-13s %s\n" "$number" "$name" "${STATUS[$name]:-not run}"
done
echo "================================================================================"
echo "Next, on your laptop:  scripts/local/sync_from_pod.sh pod"
echo "Terminate the pod only after it prints SAFE TO TERMINATE THE POD."
