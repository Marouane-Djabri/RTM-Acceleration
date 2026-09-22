// =============================================================================
// cuda-v1 — G1 rung of the CUDA ladder (docs/CUDA_PLAN.md §6,
// docs/OPTIMIZATION_PLAN.md §4 "Phase 2"). One change from cuda-v0: the
// Cerjan sponge multiply is folded into the stencil kernel instead of being
// its own launch, removing two full read+write passes over the wavefield
// per time step (the separate k_sponge dispatch in propagation_cuda.cu).
//
// BIT-IDENTITY ARGUMENT (must hold for every step, forward and backward):
//
// V0 order per step is  stencil -> inject -> sponge -> rotate:
//   pn[i] = 2 pc[i] - pp[i] + vdt2[i]*lap(pc)      (k_fd_time_step)
//   pn[src] += vdt2[src]*amplitude                 (k_inject_source/receivers)
//   pc[i] *= w[i] ;  pn[i] *= w[i]  for every i     (k_sponge)
//
// This kernel instead computes, per thread, in this order:
//   pn[i] = 2 pc[i] - pp[i] + vdt2[i]*lap(pc)      (identical formula)
//   pc[i] *= w[i] ;  pn[i] *= w[i]                  (damp immediately)
// and k_inject_source/k_inject_receivers still run afterwards, as their own
// kernel, unchanged.
//
// The only behavioural difference from V0 is that the sponge multiply now
// happens BEFORE injection instead of after. That is safe because every
// source and receiver index sits strictly inside the physical model
// (mapped through eidx(ix+nb, iz+nb) with ix in [0,nx-1], iz in [0,nz-1]),
// i.e. at extended-grid distance exactly nb from the outer edge — precisely
// the boundary of the taper, where CUDARTM::setup's taper() returns 1.0f
// exactly (d = nb >= nb). So at every point injection ever touches:
//   V0:  (stencil_value + inject_amount) * 1.0f == stencil_value + inject_amount
//   V1:  (stencil_value * 1.0f) + inject_amount  == stencil_value + inject_amount
// IEEE-754 multiplication by 1.0f is an exact no-op (no rounding), so both
// orders produce the identical bit pattern. At every other point injection
// never touches, the single multiply by w[i] is unchanged in value — moving
// which kernel performs it does not change the float operation itself.
//
// pc[i] *= w[i] has no interaction with injection at all (injection only
// ever writes to pn), so reordering it earlier is trivially safe.
//
// The outermost `half`-cell ring of the extended grid (never touched by the
// stencil, always 0) is no longer explicitly damped by any kernel once
// k_sponge is gone for this variant — that's fine, since 0.0f * w == 0.0f
// for every finite w, matching the "stays 0 forever" invariant already
// relied on by V0 (docs/CUDA_PLAN.md §3.7 checklist).
// =============================================================================
#include "rtm_cuda.hpp"
#include "kernels_cuda.hpp"

namespace rtm {

// Same __constant__ symbols CUDARTM::setup uploads once (propagation_cuda.cu)
// — extern here because CUDA_SEPARABLE_COMPILATION is on, so device symbols
// resolve across .cu files at device-link time (CMakeLists.txt).
extern __constant__ float c0_dev;
extern __constant__ float cx_dev[MAX_HALF + 1];
extern __constant__ float cz_dev[MAX_HALF + 1];
extern __constant__ int   half_dev;
extern __constant__ int   nxe_dev;
extern __constant__ int   nze_dev;

__global__ void k_fd_time_step_v1(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge) {
    const int iz = half_dev + blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = half_dev + blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;

    const std::size_t i = (std::size_t)ix * nze_dev + iz;
    const float pc_i = p_cur[i];
    float lap = c0_dev * pc_i;
    for (int k = 1; k <= half_dev; ++k) {
        const std::size_t kx = (std::size_t)k * nze_dev;
        lap += cx_dev[k] * (p_cur[i + kx] + p_cur[i - kx])   // d2/dx2
             + cz_dev[k] * (p_cur[i + k ] + p_cur[i - k ]);  // d2/dz2
    }
    const float w    = sponge[i];
    const float pn_i = 2.0f * pc_i - p_prev[i] + vdt2[i] * lap;

    p_cur[i]  = pc_i * w;   // same single multiply as k_sponge's pc[i] *= w
    p_next[i] = pn_i * w;   // same single multiply as k_sponge's pn[i] *= w
}

} // namespace rtm
