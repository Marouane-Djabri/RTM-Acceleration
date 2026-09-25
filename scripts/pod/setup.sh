#!/usr/bin/env bash
# =============================================================================
# ONE-TIME POD SETUP  (docs/runbook/00_setup.md)
#
# Run once on a fresh RunPod pod after scripts/local/sync_to_pod.sh:
#     ssh pod "cd /workspace/RTM && scripts/pod/setup.sh"
#
# Every step is skipped when its output already exists, so re-running after
# a code sync is cheap (it just rebuilds).
#
#   1. tools        cmake, g++, git, python3 (apt) if missing
#   2. build        -DCMAKE_CUDA_ARCHITECTURES=native
#   3. dataset      verify the fixed dataset's SHA-256
#   4. smoke test   synthetic pipeline: rtm_synth -> rtm cpu -> rtm_compare, ~10 s
#   5. probe        CPU + GPU bandwidth -> results/ref/bandwidth.csv
#   6. reference    rtm --engine cpu on the fixed dataset -> results/ref/marmousi_ref.bin
#                   (15-30 min, ONCE; also appends the 1x row to results/benchmarks.csv)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/dataset_marmousi.env

step() { echo; echo "=== $* ==="; }

step "1/6 tools"
need_apt=0
for tool in cmake g++ git python3; do command -v "$tool" >/dev/null || need_apt=1; done
if [ "$need_apt" = 1 ]; then
    apt-get update -qq && apt-get install -y -qq cmake g++ git python3 python3-pip
fi
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "nvcc not found: use a CUDA *devel* image"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "cpu cores: $(nproc)"

step "2/6 build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
cmake --build build -j"$(nproc)"
./build/rtm --list-engines | tr '\n' ' '; echo

step "3/6 dataset checksum"
sha256sum -c "$MARMOUSI_CHECKSUMS"

step "4/6 smoke test on the synthetic model"
mkdir -p data/synthetic results/ref results/images results/plots
./build/rtm_synth --engine cpu --outdir data/synthetic \
    --nx 201 --nz 151 --dx 10 --dz 10 --nt 1400 --dt 0.001 --f0 12 --nshots 4 >/dev/null
./build/rtm --engine cpu --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output results/images/smoke_cpu.bin --order 8 --nb 60 --f0 12 --store-interval 2 \
    --mute-direct 1500 --quiet >/dev/null
./build/rtm --engine cpu --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output results/images/smoke_cpu_run2.bin --order 8 --nb 60 --f0 12 --store-interval 2 \
    --mute-direct 1500 --quiet >/dev/null
./build/rtm_compare results/images/smoke_cpu.bin results/images/smoke_cpu_run2.bin --nx 201 --nz 151 \
    | tail -1 || echo "*** reference build is NOT reproducible on this pod — tell me before going on ***"
# The GPU engine on the same tiny problem: seconds, and proves the driver works.
# A GPU failure here must not stop the CPU-only steps 5 and 6.
if ./build/rtm --list-engines | grep -q cuda-v0; then
    if ./build/rtm --engine cuda-v0 --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
        --output results/images/smoke_cuda_v0.bin --order 8 --nb 60 --f0 12 --store-interval 2 \
        --mute-direct 1500 --quiet >/dev/null; then
        ./build/rtm_compare results/images/smoke_cpu.bin results/images/smoke_cuda_v0.bin --nx 201 --nz 151 \
            | tail -1 || true
    else
        echo "*** cuda-v0 smoke test CRASHED (see error above); continuing with CPU steps ***"
    fi
fi

# Rows in results/ tagged with THIS machine's hostname (the CSVs' "host" column).
# Files synced from an earlier pod do not count: every pod measures its own.
this_host="$(hostname)"
has_row_for_this_host() {   # has_row_for_this_host FILE [ENGINE]
    [ -s "$1" ] && python3 - "$1" "$this_host" "${2:-}" <<'PYEOF'
import csv, sys
path, host, engine = sys.argv[1], sys.argv[2], sys.argv[3]
rows = csv.DictReader(open(path, newline=""))
sys.exit(0 if any(r.get("host") == host and (not engine or r.get("engine") == engine) for r in rows) else 1)
PYEOF
}

step "5/6 bandwidth probe"
if has_row_for_this_host results/ref/bandwidth.csv; then
    echo "bandwidth already measured on this machine ($this_host), skipping"
else
    ./build/rtm_bandwidth_probe --csv results/ref/bandwidth.csv
fi

step "6/6 CPU reference on the fixed dataset (once per machine)"
if [ -s "$REFERENCE_IMAGE" ] && has_row_for_this_host results/benchmarks.csv cpu; then
    echo "$REFERENCE_IMAGE exists and the cpu timing of this machine is recorded, skipping"
elif [ -s "$REFERENCE_IMAGE" ]; then
    # The image came from another machine (synced). Keep it as the gate reference,
    # but time the CPU reference HERE, so "speedup vs cpu" uses this machine's CPU,
    # and check that this machine reproduces the reference image.
    ./build/rtm --engine cpu --velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" \
        --output results/images/cpu_reference_this_machine.bin \
        --order "$ORDER" --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL" \
        --mute-direct "$MUTE_VELOCITY" \
        --dataset "$DATASET_NAME" --benchmark-csv results/benchmarks.csv \
        --benchmark results/benchmark_reports.txt
    ./build/rtm_compare "$REFERENCE_IMAGE" results/images/cpu_reference_this_machine.bin \
        --csv results/compare.csv --engine cpu --dataset "$DATASET_NAME" | tail -1 \
        || echo "*** this machine does NOT reproduce the reference image; tell Claude before trusting the gates ***"
else
    ./build/rtm --engine cpu --velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" \
        --output "$REFERENCE_IMAGE" \
        --order "$ORDER" --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL" \
        --mute-direct "$MUTE_VELOCITY" \
        --dataset "$DATASET_NAME" --benchmark-csv results/benchmarks.csv \
        --benchmark results/benchmark_reports.txt
    # The reference compared with itself: a 1x / L2rel=0 row so the plots have their baseline.
    ./build/rtm_compare "$REFERENCE_IMAGE" "$REFERENCE_IMAGE" \
        --csv results/compare.csv --engine cpu --dataset "$DATASET_NAME" | tail -1
fi

echo; echo "setup complete."
python3 scripts/pod/summary.py || true
