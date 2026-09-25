#!/usr/bin/env bash
# =============================================================================
# Devito on the GPU (docs/PROFILING_STRATEGY.md §6.2, change M8).
#
#     scripts/pod/setup_devito_gpu.sh
#
#   1. the CPU Devito venv (scripts/pod/setup_devito.sh) + the nvtx package
#   2. the NVIDIA HPC SDK (nvc compiler, OpenACC) from NVIDIA's apt repository,
#      unless an nvc is already installed. ~10 GB on the CONTAINER disk (/opt):
#      give the pod a container disk of >= 30 GB.
#   3. scripts/pod/devito_gpu.env: PATH + DEVITO_* variables for GPU runs
#   4. a GPU smoke test: 1 shot, 200 steps of the synthetic model
#
# Safe to re-run: every step is skipped when already done.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/common.sh"

log_step "1/4 Devito venv"
scripts/pod/setup_devito.sh
# shellcheck disable=SC1091
source venv_devito/bin/activate
pip install -q nvtx
python3 -c "import devito, nvtx; print('devito', devito.__version__, '+ nvtx')"

log_step "2/4 NVIDIA HPC SDK (nvc)"
find_nvc_bin() { ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin 2>/dev/null | sort -V | tail -1; }
nvc_bin="$(find_nvc_bin)"
if [ -z "$nvc_bin" ] && ! command -v nvc >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq curl gnupg ca-certificates >/dev/null
    curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
        | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
    echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' \
        > /etc/apt/sources.list.d/nvhpc.list
    apt-get update -qq
    package=$(apt-cache search --names-only '^nvhpc-[0-9]+-[0-9]+$' | awk '{print $1}' | sort -V | tail -1)
    [ -n "$package" ] || { echo "no nvhpc package found in NVIDIA's repository"; exit 1; }
    echo "installing $package (large download, several minutes)"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$package" >/dev/null
    nvc_bin="$(find_nvc_bin)"
fi
[ -n "$nvc_bin" ] || nvc_bin="$(dirname "$(command -v nvc)")"
"$nvc_bin/nvc" --version | head -2

log_step "3/4 scripts/pod/devito_gpu.env"
cat > scripts/pod/devito_gpu.env <<EOF
# Sourced by scripts/pod/devito_sweep.sh (written by setup_devito_gpu.sh).
export PATH="$nvc_bin:\$PATH"
export DEVITO_PLATFORM=nvidiaX
export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_LOGGING=PERF
EOF
cat scripts/pod/devito_gpu.env

log_step "4/4 GPU smoke test (synthetic model, 1 shot, 200 steps)"
# shellcheck disable=SC1091
source scripts/pod/devito_gpu.env
python3 scripts/devito/marmousi_rtm_devito.py \
    --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output /tmp/devito_gpu_smoke.bin --order 8 --nb 60 --f0 12 --store-interval 2 \
    --max-shots 1 --nt 200 --dataset smoke
echo; echo "Devito GPU setup complete."
