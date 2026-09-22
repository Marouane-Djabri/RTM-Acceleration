// =============================================================================
// cuda-v4 — G4 rung of the CUDA ladder (docs/CUDA_PLAN.md §6,
// docs/OPTIMIZATION_PLAN.md §4 "Phase 2"). Builds on cuda-v3. docs/CUDA_PLAN.md
// describes this rung tersely as "register queue along z, x-neighbours from
// shared memory" and flags it as something to "try after V3, measure" rather
// than a guaranteed win — a 3D-FD-style register-queue sweep doesn't map
// cleanly onto a 2D (x,z) problem the way it does onto a real third
// dimension, so this file takes a specific, verifiable reading of that
// instruction rather than guessing at a bigger restructuring:
//
//   Block layout is unchanged from v3 (32 threads along z, 8 along x). With
//   blockDim.x == 32 == warpSize, each fixed-x row of 32 threads IS exactly
//   one warp (CUDA groups threads into warps by linear index
//   threadIdx.x + threadIdx.y*blockDim.x, and blockDim.x==32 means each
//   threadIdx.y selects a warp-aligned range). So the z-neighbours a thread
//   needs (p_cur at iz-k..iz+k) are literally the register values held by
//   OTHER LANES of its own warp — k lanes away, since threadIdx.x is the
//   lane id. That is a register-to-register warp shuffle
//   (__shfl_up/down_sync), not a shared-memory read: no shared-memory
//   traffic for the z-halo at all for interior lanes. The x-halo (a
//   different warp's data) still comes from the shared tile, exactly as
//   v3 — "x-neighbours from shared memory", matching the plan's wording.
//
//   Lanes within `half` of the warp's own edge (lane < half or
//   lane >= 32-half) don't have a same-warp neighbour that far away; those
//   fall back to the shared tile's z-halo, same as v3 computed it.
//
// BIT-IDENTITY: the value a shuffle returns for the interior case is the
// exact same float the shared tile would have held for that lane — both are
// exactly that thread's own `pc_i` (the same safe_load() result), just
// reaching the consuming thread by a different path (register vs shared
// memory). No arithmetic changes, no reordering of the lap accumulation.
//
// TWO SEPARATE SAFETY HAZARDS, BOTH HANDLED THE SAME WAY AS v3 — CLAMP,
// NEVER BRANCH AWAY FROM A COLLECTIVE OP:
//
// 1. __shfl_up_sync/__shfl_down_sync require every lane named in the mask
//    to actually execute that same shuffle call, or behaviour is undefined
//    (the same class of hazard as a divergent __syncthreads()). So every
//    lane calls both shuffles UNCONDITIONALLY with the full 0xFFFFFFFFu
//    mask — CUDA's shuffle semantics already define the out-of-warp case
//    (the calling thread's own value comes back, not a crash), and the
//    divergent CHOICE of which result to use happens afterward, on already
//    -computed register values, which is ordinary (safe) divergence.
//
// 2. Because of (1), the usual "check validity, return before touching
//    memory" guard (safe in v0-v3, which have no per-lane collective op
//    after the tile load) can't run before the shuffles here — so vdt2,
//    sponge and p_prev must be read through the same clamped safe_load()
//    the tile uses, for every thread, not through the raw (possibly wildly
//    out-of-array, see kernels_v3_shared.cu's header for the derivation)
//    flat index `i`. Only the FINAL p_cur/p_next writes, at the very end,
//    after the validity guard, use the true unclamped `i` — by then it's
//    guaranteed in range.
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

// Identical to kernels_v3_shared.cu's helpers of the same name (duplicated,
// not shared, because __device__ __forceinline__ functions in an anonymous
// namespace have internal linkage per translation unit — see that file for
// the full boundary-safety argument for why every load is clamped and why
// load_tile must be called unconditionally by every thread in the block).
__device__ __forceinline__ float safe_load(const float* p, int izz, int ixx) {
    const int zc = izz < 0 ? 0 : (izz >= nze_dev ? nze_dev - 1 : izz);
    const int xc = ixx < 0 ? 0 : (ixx >= nxe_dev ? nxe_dev - 1 : ixx);
    return p[(std::size_t)xc * nze_dev + zc];
}

__device__ __forceinline__ void load_tile(float tile[kTileX][kTileZ], const float* p,
                                          int iz, int ix, int tz, int tx) {
    tile[tx][tz] = safe_load(p, iz, ix);
    if (threadIdx.x < half_dev) {
        tile[tx][tz - half_dev]       = safe_load(p, iz - half_dev, ix);
        tile[tx][tz + kStencilBlockZ] = safe_load(p, iz + kStencilBlockZ, ix);
    }
    if (threadIdx.y < half_dev) {
        tile[tx - half_dev][tz]       = safe_load(p, iz, ix - half_dev);
        tile[tx + kStencilBlockX][tz] = safe_load(p, iz, ix + kStencilBlockX);
    }
    __syncthreads();
}

// Shared by both kernels below: computes the damped p_next value for this
// thread's point using shuffle for interior z-neighbours and the tile for
// x-neighbours + edge-lane z-neighbours. Reads vdt2/sponge/p_prev through
// safe_load (see file header, hazard 2) since it must be called
// unconditionally, including by threads whose true (ix,iz) is out of range.
// Does NOT write p_cur/p_next — the caller does that after its own validity
// guard, using the true (by-then-guaranteed-valid) flat index.
__device__ __forceinline__ float fused_step_v4(const float tile[kTileX][kTileZ],
                                               const float* p_prev, const float* vdt2,
                                               const float* sponge,
                                               int iz, int ix, int tz, int tx) {
    const int lane   = threadIdx.x;
    const float pc_i = tile[tx][tz];

    float lap = c0_dev * pc_i;
    for (int k = 1; k <= half_dev; ++k) {
        const float xterm = cx_dev[k] * (tile[tx + k][tz] + tile[tx - k][tz]);

        const float shfl_minus = __shfl_up_sync(0xFFFFFFFFu, pc_i, k);
        const float shfl_plus  = __shfl_down_sync(0xFFFFFFFFu, pc_i, k);
        const float zminus = (lane >= k)      ? shfl_minus : tile[tx][tz - k];
        const float zplus  = (lane < 32 - k)  ? shfl_plus  : tile[tx][tz + k];
        const float zterm = cz_dev[k] * (zplus + zminus);

        // Grouped exactly like v0/v1/v2/v3's `lap += cx*(...) + cz*(...);`
        // (single addition of the two terms, then added to lap) — float
        // addition isn't associative, so splitting this into two separate
        // `lap +=` statements would silently NOT be bit-identical.
        lap += xterm + zterm;
    }
    const float vdt2_i   = safe_load(vdt2, iz, ix);
    const float sponge_i = safe_load(sponge, iz, ix);
    const float pprev_i  = safe_load(p_prev, iz, ix);
    return (2.0f * pc_i - pprev_i + vdt2_i * lap) * sponge_i;
}
} // namespace

__global__ void k_fd_time_step_v4(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge) {
    const int iz = half_dev + blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = half_dev + blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = threadIdx.x + half_dev;
    const int tx = threadIdx.y + half_dev;

    __shared__ float tile[kTileX][kTileZ];
    load_tile(tile, p_cur, iz, ix, tz, tx);

    const float pn = fused_step_v4(tile, p_prev, vdt2, sponge, iz, ix, tz, tx);

    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;
    const std::size_t i = (std::size_t)ix * nze_dev + iz;   // now guaranteed in range
    p_cur[i]  = tile[tx][tz] * safe_load(sponge, iz, ix);
    p_next[i] = pn;
}

__global__ void k_fd_time_step_v4_image(const float* p_prev, float* p_cur, float* p_next,
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

    const float pn = fused_step_v4(tile, p_prev, vdt2, sponge, iz, ix, tz, tx);

    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;
    const std::size_t i = (std::size_t)ix * nze_dev + iz;   // now guaranteed in range
    p_cur[i]  = tile[tx][tz] * safe_load(sponge, iz, ix);
    p_next[i] = pn;

    if (ix < nb_dev || ix >= nb_dev + nx || iz < nb_dev || iz >= nb_dev + nz) return;
    if (is_receiver[i]) return;   // finished later by k_image_unique_receivers

    const int interior_idx = (ix - nb_dev) * nz + (iz - nb_dev);
    const float f = fwd_snapshot[interior_idx];
    image[interior_idx] += f * pn;
    illum[interior_idx] += f * f;
}

} // namespace rtm
