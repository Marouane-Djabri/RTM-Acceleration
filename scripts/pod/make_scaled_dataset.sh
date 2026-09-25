#!/usr/bin/env bash
# =============================================================================
# One scaled Marmousi2 dataset for the scaling sweeps (docs/PROFILING_STRATEGY.md
# §5.2, sweep S1, change M6).
#
#     scripts/pod/make_scaled_dataset.sh DECIMATION      10 5 4 3 2 or 1
#
# From the 1.25 m Marmousi2 P-wave SEG-Y, decimated by DECIMATION:
#   dx = dz = 1.25 m x DECIMATION
#   DT      = 0.1 ms x DECIMATION            (CFL: dt shrinks with dx)
#   NT      = 2.8 s / DT                     (same record length everywhere)
#   F0      = 10 Hz x 12.5 m / dx            (same points per wavelength everywhere)
#   STORE_INTERVAL = 10 ms / DT              (snapshot policy A: one per 10 ms)
#   NB      = 50 cells at every dx: the wavelength shrinks with dx, so a fixed
#             number of cells is a fixed number of wavelengths of sponge.
#
# Writes data/scaled/dx<dx>/{velocity.bin,velocity.bin.hdr,shots.bin,dataset.env}.
# Shots are modelled on the GPU (cuda-v1); the CPU would take days at 1.25 m.
# Skips every step whose output already exists.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

decimation="${1:?usage: $0 DECIMATION (10 5 4 3 2 1)}"
dx="${DX_OF_DECIMATION[$decimation]:-}"
[ -n "$dx" ] || { echo "unsupported decimation $decimation (use one of: $SWEEP_DECIMATIONS)"; exit 1; }

dir="data/scaled/dx$dx"
env_file="$dir/dataset.env"
segy=data/real/MODEL_P-WAVE_VELOCITY_1.25m.segy
mkdir -p "$dir"

# Every derived number, computed once in Python so the rounding is explicit.
read -r DT NT F0 STORE_INTERVAL <<<"$(python3 - "$decimation" <<'EOF'
import math, sys
d = int(sys.argv[1])
dt = 1e-4 * d
nt = math.ceil(2.8 / dt - 1e-9)
f0 = 10.0 * 10 / d
store = max(1, round(0.010 / dt))
print(f"{dt:.6g} {nt} {f0:.6g} {store}")
EOF
)"
NSHOTS=2

log_step "dataset dx=$dx m (decimation $decimation): DT=$DT NT=$NT F0=$F0 STORE_INTERVAL=$STORE_INTERVAL"

if [ ! -s "$dir/velocity.bin" ]; then
    if [ ! -s "$segy" ]; then
        tar xzf data/real/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz -C data/real
    fi
    python3 scripts/segy_to_raw.py "$segy" "$dir/velocity.bin" \
        --dx 1.25 --dz 1.25 --decimate-x "$decimation" --decimate-z "$decimation" --vmin-clip 1000
fi

cat > "$env_file" <<EOF
# Scaled dataset for the scaling sweeps (scripts/pod/make_scaled_dataset.sh).
DATASET_NAME=marmousi_dx$dx
DECIMATION=$decimation
DX=$dx
MARMOUSI_VEL=$dir/velocity.bin
MARMOUSI_SHOTS=$dir/shots.bin
NSHOTS=$NSHOTS
NT=$NT
DT=$DT
F0=$F0
REC_DX=50
SRC_Z=25
REC_Z=25
ORDER=8
NB=50
STORE_INTERVAL=$STORE_INTERVAL
SNAPSHOT_MS=10
MUTE_VELOCITY=1500
EOF

if [ ! -s "$dir/shots.bin" ]; then
    ./build/rtm_synth --engine cuda-v1 --velocity "$dir/velocity.bin" --shots-out "$dir/shots.bin" \
        --nt "$NT" --dt "$DT" --f0 "$F0" --nshots "$NSHOTS" \
        --order 8 --nb 50 --rec-dx 50 --src-z 25 --rec-z 25 | grep -v "modelling shot"
else
    echo "$dir/shots.bin exists, skipping modelling"
fi
ls -la "$dir"
