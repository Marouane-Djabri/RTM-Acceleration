#!/usr/bin/env bash
# Benchmark every engine in this build on the fixed dataset, then plot.
#     scripts/pod/bench_all.sh            all engines except cpu (its row comes from setup.sh)
#     scripts/pod/bench_all.sh --with-cpu re-run the reference too (15-30 min)
set -euo pipefail
cd "$(dirname "$0")/../.."
with_cpu=0; [ "${1:-}" = "--with-cpu" ] && with_cpu=1
for engine in $(./build/rtm --list-engines); do
    if [ "$engine" = cpu ] && [ $with_cpu = 0 ]; then continue; fi
    scripts/pod/bench.sh "$engine"
done
scripts/pod/plot.sh
