#!/usr/bin/env bash
# =============================================================================
# Benchmark ONE engine on the fixed dataset and gate it against the reference.
#
#     scripts/pod/bench.sh ENGINE [--gpus N] [--max-shots N]
#
# Writes:
#   results/images/<engine>[_gpusN].bin     the migrated image
#   results/benchmarks.csv                  one timing row     (rtm --benchmark-csv)
#   results/compare.csv                     one gate row       (rtm_compare --csv)
# and prints the summary table of every engine benchmarked so far.
#
# --max-shots is for debugging only: its rows are labelled <dataset>_shotsN so
# they never mix with the real bars.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/dataset_marmousi.env

[ $# -ge 1 ] || { echo "usage: $0 ENGINE [--gpus N] [--max-shots N]"; exit 1; }
engine="$1"; shift
gpus=1; max_shots=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)      gpus="$2"; shift 2 ;;
        --max-shots) max_shots="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

[ -s "$REFERENCE_IMAGE" ] || { echo "reference image missing: run scripts/pod/setup.sh first"; exit 1; }
./build/rtm --list-engines | grep -qx "$engine" || { echo "engine '$engine' not in this build:"; ./build/rtm --list-engines; exit 1; }

dataset="$DATASET_NAME"
image="results/images/${engine}.bin"
extra_args=()
if [ "$gpus" != 1 ]; then image="results/images/${engine}_gpus${gpus}.bin"; extra_args+=(--gpus "$gpus"); fi
if [ "$max_shots" != 0 ]; then
    dataset="${DATASET_NAME}_shots${max_shots}"
    image="results/images/${engine}_shots${max_shots}.bin"
    extra_args+=(--max-shots "$max_shots")
fi
mkdir -p results/images

echo "=== bench $engine on $dataset (gpus=$gpus) ==="
./build/rtm --engine "$engine" --velocity "$MARMOUSI_VEL" --shots "$MARMOUSI_SHOTS" \
    --output "$image" \
    --order "$ORDER" --nb "$NB" --f0 "$F0" --store-interval "$STORE_INTERVAL" \
    --mute-direct "$MUTE_VELOCITY" --quiet \
    --dataset "$dataset" --benchmark-csv results/benchmarks.csv \
    --benchmark results/benchmark_reports.txt "${extra_args[@]}"

echo "=== gate $engine vs reference ==="
compare_engine="$engine"
[ "$gpus" != 1 ] && compare_engine="${engine}_gpus${gpus}"
if [ "$max_shots" != 0 ]; then
    echo "(--max-shots run: gate skipped, the reference has all $NSHOTS shots)"
else
    set +e
    ./build/rtm_compare "$REFERENCE_IMAGE" "$image" \
        --csv results/compare.csv --engine "$compare_engine" --dataset "$dataset"
    gate_status=$?
    set -e
    [ $gate_status -eq 0 ] || echo "*** GATE FAILED for $engine (exit $gate_status) ***"
fi

echo
python3 scripts/pod/summary.py
