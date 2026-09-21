#!/usr/bin/env bash
# End-to-end synthetic validation: build -> model -> migrate -> compare.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

mkdir -p data/synthetic results

echo "=== [1/3] generating SYNTHETIC VALIDATION DATA ==="
./build/rtm_synth --outdir data/synthetic \
    --nx 201 --nz 151 --dx 10 --dz 10 \
    --nt 1400 --dt 0.001 --f0 12 --nshots 4

echo "=== [2/3] running the CPU reference RTM ==="
./build/rtm \
    --velocity data/synthetic/velocity.bin \
    --shots    data/synthetic/shots.bin \
    --output   results/reference_rtm_image.bin \
    --order 8 --nb 60 --f0 12 --store-interval 2 \
    --mute-direct 1500 \
    --illumination --filter laplacian \
    --benchmark results/benchmark.txt

# Comparing a file with itself can never fail, so it tests nothing. What the
# reference build actually promises is REPRODUCIBILITY (-O2, no -ffast-math, no
# -march=native), so migrate a second time into a different file and require the
# two runs to be bit-identical. That does catch uninitialised reads, stale
# buffers between shots, and any accidental non-determinism.
echo "=== [3/3] reproducibility check (two independent runs must be identical) ==="
./build/rtm \
    --velocity data/synthetic/velocity.bin \
    --shots    data/synthetic/shots.bin \
    --output   results/reference_rtm_image_run2.bin \
    --order 8 --nb 60 --f0 12 --store-interval 2 \
    --mute-direct 1500 --quiet > /dev/null

./build/rtm_compare results/reference_rtm_image.bin \
                    results/reference_rtm_image_run2.bin --nx 201 --nz 151

echo
echo "Reference image : results/reference_rtm_image.bin"
echo "Display it with : python3 scripts/plot_image.py \\"
echo "                    results/reference_rtm_image_filtered.bin --nx 201 --nz 151"