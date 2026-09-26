#!/usr/bin/env bash
# =============================================================================
# Helpers shared by the profiling / scaling scripts (docs/PROFILING_STRATEGY.md).
# Sourced, never run:   source "$(dirname "$0")/common.sh"
#
# Every script that sources this file runs from the repo root.
# =============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"

PROFILES_DIR=results/profiles
SCALING_CSV=results/scaling.csv
SCALING_COMPARE_CSV=results/compare_scaling.csv
FIXED_DATASET_ENV=scripts/dataset_marmousi.env

# Decimations of the 1.25 m Marmousi2 model used by the scaling sweeps, and
# the grid spacing each one gives (docs/PROFILING_STRATEGY.md §5.2, sweep S1).
SWEEP_DECIMATIONS="10 5 4 3 2 1"
declare -A DX_OF_DECIMATION=([10]=12.5 [5]=6.25 [4]=5 [3]=3.75 [2]=2.5 [1]=1.25)

# Short ncu runs: 1 shot, 600 time steps. Plenty of launches to skip the
# warm-up and capture a few, without replaying the whole migration.
NCU_NT=600

log_step() { echo; echo "=== $* ==="; }

# Cores this container may actually use. `nproc` reports every core of the
# host (256 on some RunPod machines) even when the pod is only allowed ~8
# cores of CPU time (cgroup quota). OpenMP then starts one thread per host
# core, they fight over the few allowed cores, and cpu-opt / Devito-CPU crawl.
pod_cpu_count() {
    local quota period cores
    if [ -r /sys/fs/cgroup/cpu.max ]; then                        # cgroup v2: "quota period" or "max period"
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then         # cgroup v1: quota -1 = unlimited
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    fi
    cores=$(nproc)
    if [[ "${quota:-max}" =~ ^[0-9]+$ ]] && [ "${period:-0}" -gt 0 ]; then
        local allowed=$(( (quota + period - 1) / period ))       # round up
        [ "$allowed" -ge 1 ] && [ "$allowed" -lt "$cores" ] && cores=$allowed
    fi
    echo "$cores"
}
# Every pod script (and everything run_all.sh starts) inherits this, unless
# the caller already chose OMP_NUM_THREADS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(pod_cpu_count)}"

# Install what a bare CUDA image may lack: cmake/python3 for the build and
# scripts, and Nsight Systems (CUDA *devel* images ship ncu but often not nsys;
# the CUDA apt repository they come with has it).
ensure_pod_tools() {
    local missing=()
    for tool in cmake g++ git python3 rsync; do command -v "$tool" >/dev/null || missing+=("$tool"); done
    python3 -c "import venv, ensurepip" 2>/dev/null || missing+=(python3-venv python3-pip)
    if [ ${#missing[@]} -gt 0 ]; then
        echo "installing: ${missing[*]}"
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            "${missing[@]}" >/dev/null
    fi
    # matplotlib from pip, matching the image's numpy. Ubuntu's python3-matplotlib
    # breaks against the pip-installed numpy 2 of CUDA/PyTorch images, and Ubuntu
    # 24.04 refuses plain "pip install" (PEP 668), hence the two extra flags.
    if ! python3 -c "import matplotlib" 2>/dev/null; then
        echo "installing matplotlib (pip)"
        python3 -m pip install -q matplotlib 2>/dev/null ||
            python3 -m pip install -q --break-system-packages --ignore-installed matplotlib >/dev/null 2>&1
    fi
    if ! command -v nsys >/dev/null; then
        # The package list may never have been downloaded on a fresh image.
        apt-get update -qq
        local package
        package=$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -1)
        if [ -n "$package" ]; then
            echo "installing $package"
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$package" >/dev/null
        fi
        command -v nsys >/dev/null || echo "*** nsys not available: the Nsight Systems phases will fail"
    fi
}

# Path of the env file describing one scaled dataset (written by make_scaled_dataset.sh).
scaled_env_file() {
    local decimation="$1"
    echo "data/scaled/dx${DX_OF_DECIMATION[$decimation]}/dataset.env"
}

# Load a dataset env file and fill RTM_DATA_ARGS with the input + migration
# flags every rtm run of that dataset uses.
load_dataset() {
    local env_file="$1"
    [ -f "$env_file" ] || { echo "dataset env missing: $env_file"; return 1; }
    # shellcheck disable=SC1090
    source "$env_file"
    DATASET_ENV_FILE="$env_file"
    RTM_DATA_ARGS=(--velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS"
                   --order "$ORDER" --nb "$NB" --f0 "$F0"
                   --store-interval "$STORE_INTERVAL" --mute-direct "$MUTE_VELOCITY")
}

# ---- Nsight Compute permission ----------------------------------------------
# Many container hosts block GPU performance counters (ERR_NVGPUCTRPERM). Test
# once on the synthetic model and cache the answer for the rest of the run.
ncu_counters_available() {
    local cache="$PROFILES_DIR/ncu_permission.txt"
    if [ -f "$cache" ]; then grep -qx yes "$cache"; return; fi
    mkdir -p "$PROFILES_DIR" data/synthetic
    if [ ! -s data/synthetic/shots.bin ]; then
        ./build/rtm_synth --outdir data/synthetic --nx 201 --nz 151 --dx 10 --dz 10 \
            --nt 1400 --dt 0.001 --f0 12 --nshots 4 >/dev/null
    fi
    local output
    output=$(ncu --metrics gpu__time_duration.sum --kernel-name regex:k_fd_time_step --launch-count 1 \
        ./build/rtm --engine cuda-v1 --velocity data/synthetic/velocity.bin \
        --shots data/synthetic/shots.bin --output /tmp/ncu_permission_test.bin \
        --order 8 --nb 60 --f0 12 --nt 50 --max-shots 1 --quiet 2>&1) || true
    if echo "$output" | grep -q "gpu__time_duration.sum"; then
        echo yes > "$cache"
        return 0
    fi
    echo no > "$cache"
    echo "$output" | grep -i -m3 "ERR_NVGPUCTRPERM\|error" >> "$cache" || true
    echo "*** Nsight Compute cannot read GPU counters on this machine; ncu steps are skipped."
    echo "*** See docs/PROFILING_STRATEGY.md §2.1. Reason saved in $cache"
    return 1
}

# ---- Nsight Systems GPU-metrics flag ----------------------------------------
# The flag was renamed between nsys versions; print whichever this nsys knows.
nsys_gpu_metrics_flag() {
    if nsys profile --help 2>&1 | grep -q -- "--gpu-metrics-devices"; then
        echo "--gpu-metrics-devices=all"
    elif nsys profile --help 2>&1 | grep -q -- "--gpu-metrics-device"; then
        echo "--gpu-metrics-device=all"
    fi
}

# ---- Kernel names per engine (docs/PROFILING_STRATEGY.md §4.1) --------------
# Anchored so k_fd_time_step does not also match k_fd_time_step_v1, whatever
# prefix (namespace) or suffix (parameter list) ncu shows around the name.
kernel_regex() {
    local name="$1"
    echo "regex:(^|::)${name}(\\(|$)"
}
stencil_kernel_of() {
    case "$1" in
        cuda-v0)         echo k_fd_time_step ;;
        cuda-v1|cuda-v2) echo k_fd_time_step_v1 ;;
        cuda-v3)         echo k_fd_time_step_v3 ;;
        cuda-v4)         echo k_fd_time_step_v4 ;;
    esac
}
imaging_kernel_of() {
    case "$1" in
        cuda-v0|cuda-v1) echo k_imaging ;;
        cuda-v2)         echo k_fd_time_step_v2_image ;;
        cuda-v3)         echo k_fd_time_step_v3_image ;;
        cuda-v4)         echo k_fd_time_step_v4_image ;;
    esac
}

# ---- GPU monitor: memory used + power every 200 ms --------------------------
gpu_monitor_start() {
    local log_file="$1"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits -lms 200 > "$log_file" 2>/dev/null &
    GPU_MONITOR_PID=$!
}
gpu_monitor_stop() {
    [ -n "${GPU_MONITOR_PID:-}" ] && kill "$GPU_MONITOR_PID" 2>/dev/null
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=
}

# ---- Report exports (the CSV/text versions Claude can read) ------------------
export_nsys_stats() {
    local report="$1"               # path/to/name.nsys-rep
    local base="${report%.nsys-rep}"
    nsys stats --force-export=true --format csv --output "$base" \
        --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
        "$report" >/dev/null 2>&1 \
    || nsys stats --force-export=true --format csv --output "$base" \
        --report gpukernsum,cudaapisum,gpumemtimesum,gpumemsizesum,nvtxppsum \
        "$report" >/dev/null 2>&1 \
    || echo "  (nsys stats export failed for $report; the .nsys-rep is still fine for the GUI)"
    # Old nsys versions use other report names: give them the new ones.
    local old new
    for pair in gpukernsum:cuda_gpu_kern_sum cudaapisum:cuda_api_sum gpumemtimesum:cuda_gpu_mem_time_sum \
                gpumemsizesum:cuda_gpu_mem_size_sum nvtxppsum:nvtx_sum; do
        old="${pair%%:*}"; new="${pair##*:}"
        [ -f "${base}_${old}.csv" ] && mv -f "${base}_${old}.csv" "${base}_${new}.csv"
    done
    return 0
}

export_ncu_report() {
    local report="$1"               # path/to/name.ncu-rep
    local base="${report%.ncu-rep}"
    ncu --import "$report" --page raw --csv --print-units base > "${base}_raw.csv" 2>/dev/null || true
    ncu --import "$report" --page details --print-units base > "${base}_details.txt" 2>/dev/null || true
}

# Sidecar that tells profile_extract.py what an ncu report measured.
write_ncu_meta() {
    local report="$1" engine="$2" capture="$3" cache_mode="$4" order="$5"
    cat > "${report%.ncu-rep}.meta" <<EOF
engine=$engine
capture=$capture
cache=$cache_mode
dataset=$DATASET_NAME
velocity=$MARMOUSI_VEL
nb=$NB
order=$order
EOF
}

# ---- One timed run of OUR engine at one sweep point -------------------------
# run_timing_point ENGINE SWEEP STORE_INTERVAL ORDER IMAGE_PATH
# STORE_INTERVAL is the snapshot interval in TIME STEPS (see make_scaled_dataset.sh).
# Needs load_dataset first. Runs rtm without a profiler (clean timing) while
# nvidia-smi logs memory + power, then appends one row to results/scaling.csv.
# Sets LAST_STATUS to ok | oom | fail. Never exits the calling script:
# running out of memory is a data point, not an error.
run_timing_point() {
    local engine="$1" sweep="$2" store="$3" order="$4" image="$5"
    local snapshot_ms
    snapshot_ms=$(awk -v s="$store" -v dt="$DT" 'BEGIN { printf "%g", s * dt * 1000 }')
    local run_dir="$PROFILES_DIR/$DATASET_NAME/runs"
    local name="${engine}_${sweep}_order${order}_every${store}steps"
    mkdir -p "$run_dir" "$(dirname "$image")"
    local bench_csv="$run_dir/$name.bench.csv" log="$run_dir/$name.log" monitor="$run_dir/$name.gpu.csv"
    rm -f "$bench_csv"

    echo "--- run: $engine on $DATASET_NAME, sweep $sweep, order $order, snapshot every $store steps ($snapshot_ms ms)"
    gpu_monitor_start "$monitor"
    ./build/rtm --engine "$engine" "${RTM_DATA_ARGS[@]}" --store-interval "$store" --order "$order" \
        --output "$image" --dataset "$DATASET_NAME" --benchmark-csv "$bench_csv" --quiet \
        > "$log" 2>&1 || true
    gpu_monitor_stop

    if grep -q "appended benchmark row" "$log"; then LAST_STATUS=ok
    elif grep -qi "out of memory" "$log"; then LAST_STATUS=oom
    else LAST_STATUS=fail; echo "    *** run failed, see $log"; tail -3 "$log"
    fi
    python3 scripts/pod/scaling_row.py --code ours --engine "$engine" --sweep "$sweep" \
        --env "$DATASET_ENV_FILE" --order "$order" --store-interval "$store" \
        --snapshot-ms "$snapshot_ms" --status "$LAST_STATUS" --bench-csv "$bench_csv" \
        --monitor "$monitor" --image "$image"
}

# Compare two images of the same grid, appending to results/compare_scaling.csv.
compare_images_row() {
    local reference="$1" test="$2" label="$3"
    [ -s "$reference" ] && [ -s "$test" ] || return 0
    ./build/rtm_compare "$reference" "$test" --csv "$SCALING_COMPARE_CSV" \
        --engine "$label" --dataset "$DATASET_NAME" | tail -1 || true
}
