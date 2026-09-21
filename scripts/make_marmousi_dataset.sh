#!/usr/bin/env bash
# =============================================================================
# Rebuild the FIXED benchmark dataset from the AGL Marmousi2 SEG-Y model.
#
# You normally never run this: the dataset already exists and its checksums
# are in data/real/marmousi_dataset.sha256. It is kept so the files can be
# regenerated if lost. Parameters come ONLY from scripts/dataset_marmousi.env.
#
# The AGL distribution ships models only (no prestack gathers), so the shots
# are MODELLED with the CPU reference propagator: an "inverse crime" that
# validates the pipeline at realistic scale, not the physics.
#
#   MARMOUSI=/path/to/elastic-marmousi-model scripts/make_marmousi_dataset.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dataset_marmousi.env

MARMOUSI="${MARMOUSI:-$HOME/RTM/marmousi/elastic-marmousi-model}"
SEGY="$MARMOUSI/model/MODEL_P-WAVE_VELOCITY_1.25m.segy"
[ -f "$SEGY" ] || { echo "SEG-Y model not found: $SEGY (set MARMOUSI=...)"; exit 1; }

cmake --build build -j"$(nproc)"
mkdir -p data/real

echo "=== [1/3] SEG-Y -> raw velocity ($SEGY_DECIMATION x decimation) ==="
python3 scripts/segy_to_raw.py "$SEGY" "$MARMOUSI_VEL" \
    --dx 1.25 --dz 1.25 --decimate-x "$SEGY_DECIMATION" --decimate-z "$SEGY_DECIMATION" \
    --vmin-clip "$VMIN_CLIP"

echo "=== [2/3] modelling $NSHOTS shots with the CPU reference engine ==="
./build/rtm_synth --engine cpu --velocity "$MARMOUSI_VEL" --shots-out "$MARMOUSI_SHOTS" \
    --nt "$NT" --dt "$DT" --f0 "$F0" --nshots "$NSHOTS" \
    --order "$ORDER" --nb "$NB" --rec-dx "$REC_DX" --src-z "$SRC_Z" --rec-z "$REC_Z"

echo "=== [3/3] checksums ==="
if sha256sum -c "$MARMOUSI_CHECKSUMS"; then
    echo "dataset is byte-identical to the recorded one"
else
    echo "WARNING: checksums differ from $MARMOUSI_CHECKSUMS."
    echo "         Either the SEG-Y source differs or this compiler rounds differently."
    echo "         Do NOT mix results made with different dataset bytes."
fi
