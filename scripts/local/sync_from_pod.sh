#!/usr/bin/env bash
# Bring results (CSVs, reference image, plots, RESULTS.md) back from the pod.
#     scripts/local/sync_from_pod.sh <ssh-alias> [remote-dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
alias_name="${1:?usage: $0 <ssh-alias> [remote-dir]}"
remote_dir="${2:-/workspace/RTM}"
rsync -az --info=progress2 \
    --include 'ref/***' --include 'plots/***' --include '*.csv' --include '*.txt' \
    --exclude '*' \
    "$alias_name:$remote_dir/results/" results/
rsync -az "$alias_name:$remote_dir/docs/RESULTS.md" docs/RESULTS.md 2>/dev/null || true
echo "results synced from $alias_name"
ls -la results/*.csv results/ref/ 2>/dev/null || true
