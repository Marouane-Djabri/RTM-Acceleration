#!/usr/bin/env bash
# Copy the repo (code + fixed dataset) to a RunPod pod over SSH.
#     scripts/local/sync_to_pod.sh <ssh-alias> [remote-dir]
# Excludes the local build, the python venvs, data the pod regenerates
# (synthetic + scaled datasets) and generated results.
set -euo pipefail
cd "$(dirname "$0")/../.."
alias_name="${1:?usage: $0 <ssh-alias> [remote-dir]}"
remote_dir="${2:-/workspace/RTM}"
ssh "$alias_name" "mkdir -p $remote_dir"
# --no-owner --no-group: RunPod network volumes refuse chown ("Operation not
# permitted" on every file); the files belong to the pod's root user anyway.
# results/: the POD's results are the truth. Only the CPU reference image goes
# up (the gates need it); never the CSVs, reports or plots, or the laptop's old
# copies would overwrite what the pod measured.
rsync -az --no-owner --no-group --info=progress2 \
    --include /results/ --include /results/ref/ --include /results/ref/marmousi_ref.bin \
    --exclude '/results/**' \
    --exclude build/ --exclude '*/venv/' \
    --exclude '.git/' --exclude myVenv/ --exclude venv_devito/ \
    --exclude data/synthetic/ --exclude data/scaled/ --exclude results/profiles/ \
    --exclude data/real/MODEL_P-WAVE_VELOCITY_1.25m.segy \
    ./ "$alias_name:$remote_dir/"
echo "synced to $alias_name:$remote_dir"
