// =============================================================================
// cuda-v2 — G2 rung of the CUDA ladder (docs/CUDA_PLAN.md §6,
// docs/OPTIMIZATION_PLAN.md §4 "Phase 2"). Builds on cuda-v1 (sponge fused
// into the stencil kernel). This rung additionally fuses the imaging
// cross-correlation into that same kernel, on stored backward steps, so the
// standalone k_imaging launch (a full extra read of the backward wavefield
// over every interior point) is no longer needed in the common case.
//
// WHY THIS CANNOT BE A NAIVE "SAME THREAD ALSO DOES image += f*b" FUSION
// -----------------------------------------------------------------------
// The backward driver's per-step order is, and must stay:
//     stencil+sponge (writes p_next, damped)
//     inject_receivers (adds to p_next at nrec sparse points, atomicAdd)
//     rotate
//     [stored step] imaging (reads the now-current p_cur = old p_next)
//
// Imaging must read the wavefield *after* receiver injection — the residual
// at receiver locations is part of the physical backward wavefield being
// correlated. If the thread that computes p_next[i] inside the stencil
// kernel also did image[i] += f*p_next[i] right there, it would do so
// *before* k_inject_receivers has run, so at the (up to a few hundred)
// points that are actually receiver locations this step, the image would be
// missing that step's injected contribution — wrong, and not a rounding
// difference: a structurally different (smaller) number, so not bit
// identical to cuda-v1 for those pixels. src/cpu_opt/cpu_opt.cpp hit the
// same fact and documents it (see its backward_propagation comment,
// "would image the receiver cells BEFORE their injection"); it responds by
// never fusing imaging into the stencil sweep at all. This rung does fuse
// it, but only where it's provably safe, and closes the gap for the rest:
//
//   1. k_fd_time_step_v2_image (this file) does the stencil+sponge exactly
//      like k_fd_time_step_v1, and ALSO does image/illum accumulation using
//      its own just-computed p_next value — but ONLY for interior points
//      that are NOT a receiver location this shot (checked against
//      `is_receiver`, built per shot by k_mark_receivers below). For every
//      such point, receiver injection will never touch it this step or any
//      other, so the value it just computed IS the final post-injection
//      value — reading it from a register is bit-identical to how the old
//      standalone k_imaging read it from global memory one kernel later
//      (same float, same bits, whichever kernel happens to read it).
//   2. k_inject_receivers still runs immediately after, unchanged.
//   3. k_image_unique_receivers (this file) then finishes the job for
//      exactly the points k_fd_time_step_v2_image skipped: one thread per
//      UNIQUE extended-grid receiver index this shot (deduplicated on the
//      host in CUDARTM::map_geometry — see rtm_cuda.cu), reading p_cur
//      *after* injection and rotation, computing image += f*b exactly as
//      k_imaging always did. Deduplication matters: k_imaging visits each
//      grid point once regardless of how many receivers clamp onto it, so a
//      kernel indexed by receiver instead of by unique point would double
//      an image contribution wherever two receivers coincide — silently
//      wrong, not caught by a shape/size check. Working from the unique set
//      keeps this bit-identical to k_imaging's original one-thread-per-point
//      semantics.
//
// A tempting shortcut — let step 1 image every point including receivers
// with the pre-injection value, then have step 3 ADD a correction
// f*injection_amount — is NOT bit-identical: float multiplication does not
// distribute exactly over addition (f*(a+b) is not bit-identical to
// f*a + f*b in general), so that would silently produce a different
// rounding than k_imaging's single f*b_final multiply. Step 3 therefore
// always recomputes f*b from scratch with the true final b, never adds a
// delta to an already-written value.
//
// illum[i] += f*f does not depend on b at all, so it could in principle be
// computed anywhere — it is left inside the same skip/finish split as image
// purely so both arrays are always written together, which is easier to
// reason about than splitting them.
// =============================================================================
#include "rtm_cuda.hpp"
#include "kernels_cuda.hpp"

namespace rtm {

// Same __constant__ symbols CUDARTM::setup uploads once (propagation_cuda.cu).
extern __constant__ float c0_dev;
extern __constant__ float cx_dev[MAX_HALF + 1];
extern __constant__ float cz_dev[MAX_HALF + 1];
extern __constant__ int   half_dev;
extern __constant__ int   nxe_dev;
extern __constant__ int   nze_dev;
extern __constant__ int   nb_dev;

__global__ void k_mark_receivers(unsigned char* is_receiver, const int* rec_index, int nrec) {
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrec) return;
    is_receiver[rec_index[ir]] = 1;
}

__global__ void k_fd_time_step_v2_image(const float* p_prev, float* p_cur, float* p_next,
                                        const float* vdt2, const float* sponge,
                                        const float* fwd_snapshot,
                                        const unsigned char* is_receiver,
                                        float* image, float* illum,
                                        int nx, int nz) {
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
    const float w      = sponge[i];
    const float pn_i   = 2.0f * pc_i - p_prev[i] + vdt2[i] * lap;
    const float pn_dmp = pn_i * w;

    p_cur[i]  = pc_i * w;
    p_next[i] = pn_dmp;

    // Only interior points participate in imaging at all (matches k_imaging's
    // nx*nz thread range, never the sponge/taper padding).
    if (ix < nb_dev || ix >= nb_dev + nx || iz < nb_dev || iz >= nb_dev + nz) return;
    if (is_receiver[i]) return;   // finished later by k_image_unique_receivers

    const int interior_idx = (ix - nb_dev) * nz + (iz - nb_dev);
    const float f = fwd_snapshot[interior_idx];
    image[interior_idx] += f * pn_dmp;
    illum[interior_idx] += f * f;
}

__global__ void k_image_unique_receivers(const float* fwd_snapshot, const float* p_cur,
                                         const int* unique_ext_index, int nuniq,
                                         float* image, float* illum,
                                         int nz, int nb, int nze) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nuniq) return;

    const int ext    = unique_ext_index[k];
    const int ix_ext = ext / nze;
    const int iz_ext = ext - ix_ext * nze;
    const int interior_idx = (ix_ext - nb) * nz + (iz_ext - nb);

    const float f = fwd_snapshot[interior_idx];
    const float b = p_cur[ext];
    image[interior_idx] += f * b;
    illum[interior_idx] += f * f;
}

} // namespace rtm
