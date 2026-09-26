#!/usr/bin/env bash
# =============================================================================
# Devito on the same scaling sweeps as our engine (docs/PROFILING_STRATEGY.md
# §6.2 "Scaling Devito up", change M8).
#
#     scripts/pod/devito_sweep.sh grid     S1: every scaled dataset, snapshot every 10 ms
#     scripts/pod/devito_sweep.sh order    S2: orders 4 8 12 16 at dx = 2.5 m
#
# Needs scripts/pod/setup_devito_gpu.sh. Per point:
#   1. timed run, all shots of the dataset, default Devito profiling (clean timing)
#   2. if it fitted: Nsight Systems run, 1 shot, NVTX ranges, DEVITO_PROFILING=advanced
#      (Devito's own GPts/s, GFlops/s, OI go into the scaling row)
#   3. Nsight Compute on the kernel that took the most GPU time in (2)
#   4. one row in results/scaling.csv (code=devito), and the image compared
#      with our cuda-v1 image of the same dataset (results/compare_scaling.csv)
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

mode="${1:?usage: $0 grid|order}"
[ -f scripts/pod/devito_gpu.env ] || { echo "run scripts/pod/setup_devito_gpu.sh first"; exit 1; }
# shellcheck disable=SC1091
source venv_devito/bin/activate
# shellcheck disable=SC1091
source scripts/pod/devito_gpu.env

devito_script=scripts/devito/marmousi_rtm_devito.py

# run_devito_point SWEEP ORDER
run_devito_point() {
    local sweep="$1" order="$2"
    local name="devito_${sweep}_order${order}"
    local run_dir="$PROFILES_DIR/$DATASET_NAME/runs"
    local image="results/images/scaling/$DATASET_NAME/devito_order${order}.bin"
    local nsys_dir="$PROFILES_DIR/$DATASET_NAME/nsys" ncu_dir="$PROFILES_DIR/$DATASET_NAME/ncu"
    mkdir -p "$run_dir" "$nsys_dir" "$ncu_dir" "$(dirname "$image")"
    local common_args=(--velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" --order "$order"
                       --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL"
                       --mute-direct "$MUTE_VELOCITY" --dataset "$DATASET_NAME")

    echo "--- devito: $DATASET_NAME, sweep $sweep, order $order"
    gpu_monitor_start "$run_dir/$name.gpu.csv"
    local exit_code=0
    python3 "$devito_script" "${common_args[@]}" --output "$image" \
        --run-json "$run_dir/$name.json" > "$run_dir/$name.log" 2>&1 || exit_code=$?
    gpu_monitor_stop
    local status=fail
    if [ -s "$run_dir/$name.json" ]; then status=ok
    elif [ "$exit_code" = 137 ]; then
        # SIGKILL from the kernel's OOM killer: Devito keeps its saved wavefield
        # in HOST memory, so its limit is the container's RAM, not the GPU's.
        status=oom; echo "    Devito killed for exceeding host memory (exit 137)"
    elif grep -qiE "out of memory|MemoryError|OUT_OF_MEMORY|MemoryAllocation|cuMemAlloc" "$run_dir/$name.log"; then status=oom
    else echo "    *** Devito run failed, see $run_dir/$name.log"; tail -3 "$run_dir/$name.log"
    fi

    local perf_json=""
    if [ "$status" = ok ]; then
        # Timeline of one shot, with Devito's own performance counters on.
        local report="$nsys_dir/devito_order${order}.nsys-rep"
        DEVITO_PROFILING=advanced nsys profile --output "${report%.nsys-rep}" --force-overwrite true \
            --trace cuda,nvtx,osrt,openacc --cuda-memory-usage true --stats false \
            python3 "$devito_script" "${common_args[@]}" --output /tmp/devito_nsys.bin \
            --max-shots 1 --nvtx --run-json "$run_dir/$name.perf.json" \
            > "$nsys_dir/devito_order${order}.log" 2>&1 || true
        [ -s "$run_dir/$name.perf.json" ] && perf_json="$run_dir/$name.perf.json"
        if [ -s "$report" ]; then
            export_nsys_stats "$report"
            # The stencil is the kernel with the largest total GPU time.
            local top_kernel
            top_kernel=$(python3 - "${report%.nsys-rep}_cuda_gpu_kern_sum.csv" <<'EOF'
import csv, re, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    time_key = next(k for k in rows[0] if "Total Time" in k)
    top = max(rows, key=lambda r: float(r[time_key]))
    print(re.escape(top["Name"].split("(")[0]))
except Exception:
    pass
EOF
)
            if [ -n "$top_kernel" ] && ncu_counters_available; then
                local ncu_report="$ncu_dir/devito_order${order}_stencil.ncu-rep"
                echo "--- ncu: devito kernel $top_kernel"
                ncu --set full --import-source yes --kernel-name "regex:(^|::)${top_kernel}(\\(|$)" \
                    --launch-skip 200 --launch-count 3 --export "${ncu_report%.ncu-rep}" --force-overwrite \
                    python3 "$devito_script" "${common_args[@]}" --output /tmp/devito_ncu.bin \
                    --max-shots 1 --nt "$NCU_NT" > "$ncu_dir/devito_order${order}_stencil.log" 2>&1 || true
                if [ -s "$ncu_report" ]; then
                    export_ncu_report "$ncu_report"
                    write_ncu_meta "$ncu_report" devito stencil all "$order"
                fi
            fi
        fi
    fi

    # Devito's performance numbers come from the profiled run; timings from the clean one.
    python3 - "$run_dir/$name.json" "$perf_json" <<'EOF'
import json, sys
timed, perf = sys.argv[1], sys.argv[2]
try:
    run = json.load(open(timed))
    if perf:
        for key, value in json.load(open(perf)).items():
            if key.startswith("devito_"):
                run[key] = value
    json.dump(run, open(timed, "w"), indent=1)
except (OSError, ValueError):
    pass
EOF
    python3 scripts/pod/scaling_row.py --code devito --engine devito-gpu --sweep "$sweep" \
        --env "$DATASET_ENV_FILE" --order "$order" --store-interval "$STORE_INTERVAL" \
        --snapshot-ms "$SNAPSHOT_MS" --status "$status" --devito-json "$run_dir/$name.json" \
        --monitor "$run_dir/$name.gpu.csv" --image "$image"

    local ours="results/images/scaling/$DATASET_NAME/cuda-v1_every${STORE_INTERVAL}steps.bin"
    [ "$sweep" = S2 ] && ours="results/images/scaling/$DATASET_NAME/cuda-v1_order${order}.bin"
    [ "$status" = ok ] && compare_images_row "$ours" "$image" "devito_vs_cuda-v1_order${order}"
    return 0
}

case "$mode" in
    grid)
        for decimation in $SWEEP_DECIMATIONS; do
            env_file="$(scaled_env_file "$decimation")"
            [ -f "$env_file" ] || { echo "*** $env_file missing, skipping"; continue; }
            load_dataset "$env_file"
            log_step "Devito S1: $DATASET_NAME (dx=$DX m, f0=$F0 Hz)"
            run_devito_point S1 "$ORDER"
        done ;;
    order)
        load_dataset "$(scaled_env_file 2)"
        STORE_INTERVAL=$(( STORE_INTERVAL * 4 ))    # same as order_sweep.sh: fits, throughput only
        SNAPSHOT_MS=$(awk -v s="$STORE_INTERVAL" -v dt="$DT" 'BEGIN { printf "%g", s * dt * 1000 }')
        for order in 4 8 12 16; do
            log_step "Devito S2: $DATASET_NAME order $order"
            run_devito_point S2 "$order"
        done ;;
    *) echo "usage: $0 grid|order"; exit 1 ;;
esac
