#!/usr/bin/env bash
# Render results/plots/*.png and docs/RESULTS.md from the CSVs.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c "import matplotlib" 2>/dev/null || python3 -m pip install -q matplotlib
python3 scripts/plot_benchmark.py \
    --benchmarks results/benchmarks.csv --compare results/compare.csv \
    --bandwidth results/ref/bandwidth.csv --out results/plots --results-md docs/RESULTS.md
