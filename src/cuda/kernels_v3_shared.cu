// =============================================================================
// cuda-v3 — G3 rung of the CUDA ladder (docs/CUDA_PLAN.md §6,
// docs/OPTIMIZATION_PLAN.md §4 "Phase 2"). Builds on cuda-v2. Pure memory
// optimization, no change in arithmetic: each block loads its
// (32+2*half) x (8+2*half) tile of p_cur into __shared__ once, and every
// thread's Laplacian reads its `1 + 2*half` neighbours per axis from shared
// memory instead of global memory. Bit-identical to cuda-v2 because the
// VALUES read are byte-for-byte copies of the same global-memory floats
// (a plain load, no computation) and the arithmetic — same k-loop, same
// order of adds/multiplies — is untouched; only where each float physically
// comes from changes.
//
// This file provides shared-memory versions of both cuda-v2 kernels:
//   k_fd_time_step_v3        — replaces k_fd_time_step_v1 everywhere it was
//                               used (forward driver always; backward driver
//                               on non-stored steps).
//   k_fd_time_step_v3_image  — replaces k_fd_time_step_v2_image (backward
//                               driver, stored steps): same shared-memory
//                               stencil, plus the same receiver-skipping
//                               imaging fusion as v2 (see kernels_v2_imaging.cu
//                               for why imaging must skip receiver points and
//                               be finished afterwards by
//                               k_image_unique_receivers, unchanged from v2).
//
// BOUNDARY SAFETY — the part that actually needs care here
// ----------------------------------------------------------
// v0/v1/v2's kernels compute i = ix*nze+iz and immediately `if (out of the
// valid [half, n-half) compute range) return;` before ever touching global
// memory — safe, because with no __syncthreads() in those kernels, threads
// are free to take different paths.
//
// This kernel calls __syncthreads() after loading the tile, which every
// thread in the block MUST reach — an early `return` before it is undefined
// behaviour (a classic shared-memory stencil bug: a block whose grid
// dimension doesn't divide evenly by the block size launches threads whose
// (ix,iz) can be arbitrarily far past the end of the array, not just past
// the compute-valid range; see the derivation kept in the coding log for
// why the excess isn't bounded by `half`). So every thread — valid or not —
// participates in the load, but each global read is clamped to
// [0,nxe_dev) x [0,nze_dev) first, so no thread ever reads outside the
// allocated p_cur/p_prev buffers even when its "true" (ix,iz) is nonsense.
// Clamped loads for threads whose own point is invalid are never read by a
// valid thread: a valid thread's own true neighbours (iz-half..iz+half,
// ix-half..ix+half) are always themselves within [0,nxe_dev)x[0,nze_dev) —
// proven the same way the v1 kernel's lack of clamping was already safe for
// *its* reads — so clamping never substitutes a wrong value for a real one,
// it only fills in harmless padding for slots nobody valid will read.
// Only after __syncthreads() does each thread check its own validity and
// bail out (now safe, since the barrier has already been satisfied by the
// whole block).
//
// Block dimensions are fixed at kStencilBlockZ=32 x kStencilBlockX=8 to
// match propagation_cuda.cu's launch configuration for these kernels — the
// tile layout below is only correct for that exact block shape.
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

namespace {
constexpr int kStencilBlockZ = 32;   // MUST match propagation_cuda.cu's block.x
constexpr int kStencilBlockX = 8;    // MUST match propagation_cuda.cu's block.y
constexpr int kTileZ = kStencilBlockZ + 2 * MAX_HALF;
constexpr int kTileX = kStencilBlockX + 2 * MAX_HALF;

// Clamped global-memory fetch: never reads outside [0,nxe_dev) x [0,nze_dev),
// regardless of how far (izz,ixx) has strayed from the true grid.
__device__ __forceinline__ float safe_load(const float* p, int izz, int ixx) {
    const int zc = izz < 0 ? 0 : (izz >= nze_dev ? nze_dev - 1 : izz);
    const int xc = ixx < 0 ? 0 : (ixx >= nxe_dev ? nxe_dev - 1 : ixx);
    return p[(std::size_t)xc * nze_dev + zc];
}

// Loads this block's (kStencilBlockX+2*half) x (kStencilBlockZ+2*half) tile
// of `p` into `tile`. Every thread loads its own centre point; the threads
// nearest each edge (threadIdx.{x,y} < half_dev) additionally load that
// edge's halo. Must be called by every thread in the block, unconditionally
// (see the file header) — synchronizes internally, callers must NOT have
// returned early before calling this.
__device__ __forceinline__ void load_tile(float tile[kTileX][kTileZ], const float* p,
                                          int iz, int ix, int tz, int tx) {
    tile[tx][tz] = safe_load(p, iz, ix);
    if (threadIdx.x < half_dev) {
        tile[tx][tz - half_dev]            = safe_load(p, iz - half_dev, ix);
        tile[tx][tz + kStencilBlockZ]      = safe_load(p, iz + kStencilBlockZ, ix);
    }
    if (threadIdx.y < half_dev) {
        tile[tx - half_dev][tz]            = safe_load(p, iz, ix - half_dev);
        tile[tx + kStencilBlockX][tz]      = safe_load(p, iz, ix + kStencilBlockX);
    }
    __syncthreads();
}
} // namespace

__global__ void k_fd_time_step_v3(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge) {
    const int iz = half_dev + blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = half_dev + blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = threadIdx.x + half_dev;
    const int tx = threadIdx.y + half_dev;

    __shared__ float tile[kTileX][kTileZ];
    load_tile(tile, p_cur, iz, ix, tz, tx);

    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;

    const std::size_t i = (std::size_t)ix * nze_dev + iz;
    const float pc_i = tile[tx][tz];
    float lap = c0_dev * pc_i;
    for (int k = 1; k <= half_dev; ++k) {
        lap += cx_dev[k] * (tile[tx + k][tz] + tile[tx - k][tz])   // d2/dx2
             + cz_dev[k] * (tile[tx][tz + k] + tile[tx][tz - k]);  // d2/dz2
    }
    const float w    = sponge[i];
    const float pn_i = 2.0f * pc_i - p_prev[i] + vdt2[i] * lap;

    p_cur[i]  = pc_i * w;
    p_next[i] = pn_i * w;
}

__global__ void k_fd_time_step_v3_image(const float* p_prev, float* p_cur, float* p_next,
                                        const float* vdt2, const float* sponge,
                                        const float* fwd_snapshot,
                                        const unsigned char* is_receiver,
                                        float* image, float* illum,
                                        int nx, int nz) {
    const int iz = half_dev + blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = half_dev + blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = threadIdx.x + half_dev;
    const int tx = threadIdx.y + half_dev;

    __shared__ float tile[kTileX][kTileZ];
    load_tile(tile, p_cur, iz, ix, tz, tx);

    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;

    const std::size_t i = (std::size_t)ix * nze_dev + iz;
    const float pc_i = tile[tx][tz];
    float lap = c0_dev * pc_i;
    for (int k = 1; k <= half_dev; ++k) {
        lap += cx_dev[k] * (tile[tx + k][tz] + tile[tx - k][tz])   // d2/dx2
             + cz_dev[k] * (tile[tx][tz + k] + tile[tx][tz - k]);  // d2/dz2
    }
    const float w      = sponge[i];
    const float pn_i   = 2.0f * pc_i - p_prev[i] + vdt2[i] * lap;
    const float pn_dmp = pn_i * w;

    p_cur[i]  = pc_i * w;
    p_next[i] = pn_dmp;

    if (ix < nb_dev || ix >= nb_dev + nx || iz < nb_dev || iz >= nb_dev + nz) return;
    if (is_receiver[i]) return;   // finished later by k_image_unique_receivers

    const int interior_idx = (ix - nb_dev) * nz + (iz - nb_dev);
    const float f = fwd_snapshot[interior_idx];
    image[interior_idx] += f * pn_dmp;
    illum[interior_idx] += f * f;
}

} // namespace rtm
