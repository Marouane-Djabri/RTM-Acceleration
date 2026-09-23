#!/usr/bin/env bash
# =============================================================================
# ONE-TIME DEVITO SETUP  (docs/runbook/15_devito_reference.md)
#
# Installs Devito (github.com/devitocodes/devito) into its own venv on the
# pod and runs a tiny smoke test -- completely independent of the project's
# C++ build. Devito is an INDUSTRY REFERENCE cross-check, not a rung in the
# CUDA ladder (docs/OPTIMIZATION_PLAN.md): it never touches
# results/benchmarks.csv or results/compare.csv.
#
#     ssh pod "cd /workspace/RTM && scripts/pod/setup_devito.sh"
#
# Runs on CPU with OpenMP (DEVITO_LANGUAGE=openmp) -- reliable on any pod
# image, no NVIDIA HPC SDK / OpenACC compiler required. Devito's own GPU
# offload path is optional and covered separately in the runbook, since it
# needs an nvc/nvc++ toolchain that most RunPod images don't ship.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { echo; echo "=== $* ==="; }

step "1/3 python + venv"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
python3 -c "import ensurepip" 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv python3-pip; }
if [ ! -d venv_devito ]; then
    python3 -m venv venv_devito
fi
source venv_devito/bin/activate
python3 -m pip install -q --upgrade pip

step "2/3 install devito"
pip install -q devito
python3 -c "import devito; print('devito', devito.__version__)"
export DEVITO_LANGUAGE=openmp
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
echo "DEVITO_LANGUAGE=$DEVITO_LANGUAGE  OMP_NUM_THREADS=$OMP_NUM_THREADS  cpu cores: $(nproc)"

step "3/3 smoke test on the small synthetic model"
mkdir -p data/synthetic results/ref
if [ ! -s data/synthetic/velocity.bin ] || [ ! -s data/synthetic/shots.bin ]; then
    echo "data/synthetic/{velocity,shots}.bin missing -- build the C++ tools and run"
    echo "  ./build/rtm_synth --outdir data/synthetic --nx 201 --nz 151 --dx 10 --dz 10 \\"
    echo "      --nt 1400 --dt 0.001 --f0 12 --nshots 4"
    echo "first (see scripts/run_synthetic.sh), or skip straight to the Marmousi run below."
else
    python3 scripts/devito/marmousi_rtm_devito.py \
        --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
        --output results/ref/smoke_devito.bin \
        --order 8 --nb 60 --f0 12 --store-interval 2 --mute-direct 1500
    echo "smoke test wrote results/ref/smoke_devito.bin"
fi

echo
echo "setup complete. Next: docs/runbook/15_devito_reference.md section 3 (the Marmousi run)."
