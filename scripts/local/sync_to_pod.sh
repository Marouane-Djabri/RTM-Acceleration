#!/usr/bin/env bash
# Copy the repo (code + fixed dataset) to a RunPod pod over SSH.
#     scripts/local/sync_to_pod.sh <ssh-alias> [remote-dir]
# Excludes the local build, the python venv and generated results.
set -euo pipefail
cd "$(dirname "$0")/../.."
alias_name="${1:?usage: $0 <ssh-alias> [remote-dir]}"
remote_dir="${2:-/workspace/RTM}"
ssh "$alias_name" "mkdir -p $remote_dir"
rsync -az --info=progress2 \
    --exclude build/ --exclude '*/venv/' --exclude results/images/ \
    --exclude '.git/' \
    ./ "$alias_name:$remote_dir/"
echo "synced to $alias_name:$remote_dir"
