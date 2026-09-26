#!/usr/bin/env bash
# Bring results back from the pod: CSVs, reference image, plots, RESULTS.md,
# and every profiler report (results/profiles/, docs/PROFILING_STRATEGY.md M5).
#     scripts/local/sync_from_pod.sh <ssh-alias> [remote-dir]
#
# Ends with a checksum comparison of results/profiles/ and results/*.csv on both
# sides, and prints "SAFE TO TERMINATE THE POD" only when everything arrived.
# Also the migrated images of the fixed 12.5 m dataset (results/images/*.bin,
# ~1.5 MB each: the per-version result images for the report).
# Not synced on purpose: results/images/scaling/ and data/scaled/ (GBs, regenerable).
set -euo pipefail
cd "$(dirname "$0")/../.."
alias_name="${1:?usage: $0 <ssh-alias> [remote-dir]}"
remote_dir="${2:-/workspace/RTM}"
rsync -az --info=progress2 \
    --include 'ref/***' --include 'plots/***' --include 'profiles/***' \
    --include '*.csv' --include '*.txt' --include '*.md' \
    --include 'images/' --include 'images/*.bin' \
    --exclude '*' \
    "$alias_name:$remote_dir/results/" results/
rsync -az "$alias_name:$remote_dir/docs/RESULTS.md" docs/RESULTS.md 2>/dev/null || true
echo "results synced from $alias_name"

# ---- verify: same files, same bytes, on both sides ---------------------------
checksum_list='cd results && { find profiles -type f 2>/dev/null; ls *.csv *.md images/*.bin 2>/dev/null; } | LC_ALL=C sort | xargs -r md5sum'
remote_list=$(ssh "$alias_name" "cd $remote_dir && $checksum_list")
local_list=$(bash -c "$checksum_list")
remote_count=$(echo "$remote_list" | grep -c . || true)
# Every pod file must exist here with the same checksum (extra local files are fine).
missing=$(comm -23 <(echo "$remote_list" | LC_ALL=C sort) <(echo "$local_list" | LC_ALL=C sort))
if [ -z "$missing" ] && [ "$remote_count" -gt 0 ]; then
    echo
    echo "verified: all $remote_count pod files are here, byte-identical."
    echo "SAFE TO TERMINATE THE POD (reports are in results/profiles/)."
else
    echo
    echo "*** Missing or different here (checksum  file). Do NOT terminate the pod yet:"
    echo "${missing:-  (the pod has no results at all)}" | head -20
    echo "*** Re-run this script; if it persists, look at the files listed above."
    exit 1
fi
