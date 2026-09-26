#!/usr/bin/env bash
# =============================================================================
# SHORT SESSION: the migrated image of every version, and Devito's, on the
# fixed 12.5 m Marmousi dataset. No sweeps, no profiling. About 15 minutes.
#
#     ssh pod "cd /workspace/RTM && nohup scripts/pod/images_run.sh > images_run.out 2>&1 < /dev/null &"
#     ssh pod "tail -f /workspace/RTM/images_run.out"
#
#   1. tools + build                                                  (~2-3 min)
#   2. cuda-v0 .. cuda-v4: one full migration each (12 shots, seconds)
#      -> results/images/<engine>.bin (+ _illum, _filtered variants)
#   3. Devito on the CPU (pip install, no NVIDIA HPC SDK needed: the image does
#      not depend on where Devito runs, only the speed does)         (~5-10 min)
#      -> results/images/devito.bin
#   4. every image compared with the CPU reference -> results/compare_images.csv
#
# Then on the laptop:  scripts/local/sync_from_pod.sh pod   (brings results/images/*.bin)
#                      myVenv/bin/python scripts/plot_version_images.py
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/common.sh"
source "$FIXED_DATASET_ENV"

log_step "1/4 tools + build"
missing=()
for tool in cmake g++ python3; do command -v "$tool" >/dev/null || missing+=("$tool"); done
python3 -c "import venv, ensurepip" 2>/dev/null || missing+=(python3-venv python3-pip)
if [ ${#missing[@]} -gt 0 ]; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}" >/dev/null
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null &&
    cmake --build build -j"$(nproc)" 2>&1 | grep -E "error|Built target rtm$" || { echo "*** build failed"; exit 1; }
[ -s "$REFERENCE_IMAGE" ] || { echo "*** $REFERENCE_IMAGE missing: run scripts/local/sync_to_pod.sh first"; exit 1; }
sha256sum -c "$MARMOUSI_CHECKSUMS" --quiet && echo "dataset checksum OK"

mkdir -p results/images
compare_csv=results/compare_images.csv
compare() {   # compare IMAGE LABEL
    ./build/rtm_compare "$REFERENCE_IMAGE" "$1" --nx 1361 --nz 281 \
        --csv "$compare_csv" --engine "$2" --dataset "$DATASET_NAME" | grep -E "correlation|relative L2|PASS|FAIL"
}

log_step "2/4 the five GPU versions"
for engine in cuda-v0 cuda-v1 cuda-v2 cuda-v3 cuda-v4; do
    echo "--- $engine"
    ./build/rtm --engine "$engine" --velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" \
        --output "results/images/$engine.bin" \
        --order "$ORDER" --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL" \
        --mute-direct "$MUTE_VELOCITY" --illumination --filter laplacian --quiet \
        | grep -E "Total:|wrote" | head -3
    compare "results/images/$engine.bin" "$engine"
done

log_step "3/4 Devito on the CPU (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
scripts/pod/setup_devito.sh >/dev/null 2>&1 || { echo "*** Devito install failed"; scripts/pod/setup_devito.sh 2>&1 | tail -5; }
# shellcheck disable=SC1091
source venv_devito/bin/activate
DEVITO_LANGUAGE=openmp python3 scripts/devito/marmousi_rtm_devito.py \
    --velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" --output results/images/devito.bin \
    --order "$ORDER" --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL" \
    --mute-direct "$MUTE_VELOCITY" --dataset "$DATASET_NAME" 2>&1 | grep -vE "^Operator|^\s*$" | tail -8
deactivate
[ -s results/images/devito.bin ] && compare results/images/devito.bin devito \
    || echo "*** no Devito image"

log_step "4/4 done"
ls -la results/images/*.bin | awk '{print $5, $9}'
echo
echo "Next, on the laptop:"
echo "    scripts/local/sync_from_pod.sh pod"
echo "    myVenv/bin/python scripts/plot_version_images.py"
