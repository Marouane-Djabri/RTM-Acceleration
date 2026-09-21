// =============================================================================
// KERNEL 7 — zero-lag cross-correlation imaging condition, mirrors
// CPUReferenceRTM::imaging (src/rtm/imaging.cpp). Fused into the tail of the
// backward stencil driver in propagation_cuda.cu; no atomics needed, one
// thread owns one interior point (docs/CUDA_PLAN.md §3.4 #7).
// =============================================================================
#include "rtm_cuda.hpp"
#include "cuda_check.hpp"
#include "kernels_cuda.hpp"
#include <stdexcept>

namespace rtm {

__global__ void k_imaging(const float* fwd_snapshot, const float* bwd_extended,
                          float* image, float* illum,
                          int nx, int nz, int nb, int nze) {
    const int iz = blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= nz || ix >= nx) return;

    const std::size_t i = (std::size_t)ix * nz + iz;
    const float f = fwd_snapshot[i];
    const float b = bwd_extended[(std::size_t)(ix + nb) * nze + (iz + nb)];
    image[i] += f * b;
    illum[i] += f * f;
}

// The base RTMEngine interface requires an override, but CUDARTM keeps its
// wavefields device-resident and only ever runs the imaging condition fused
// into backward_propagation via migrate(). A standalone call here would mean
// someone bypassed migrate() and is holding host buffers we never populate —
// honest to refuse rather than silently do the wrong thing (§3.5).
void CUDARTM::imaging(const float*, const float*, std::vector<float>&, std::vector<float>&) {
    throw std::runtime_error(
        "CUDARTM::imaging() is not implemented standalone; "
        "use migrate() (device-resident imaging fused into backward_propagation)");
}

} // namespace rtm
