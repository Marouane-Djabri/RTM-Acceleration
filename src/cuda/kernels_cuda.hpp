#pragma once
// Internal CUDA-only header: kernel declarations shared between
// propagation_cuda.cu (defines kernels 1-5, launches all of them plus
// kernel 7) and imaging_cuda.cu (defines kernel 7). Requires
// CUDA_SEPARABLE_COMPILATION so a __global__ defined in one .cu can be
// launched from another (docs/CUDA_PLAN.md §2.1, §2.3).
#include <cstddef>

namespace rtm {

// Uploads the stencil coefficients / grid constants used by k_fd_time_step
// into __constant__ memory (defined in propagation_cuda.cu). Called once
// from CUDARTM::setup().
void upload_stencil_constants(float c0, const float* cx, const float* cz,
                              int half, int nxe, int nze, int nb);

// KERNEL 1 — mirrors CPUReferenceRTM::fd_time_step (src/rtm/forward.cpp).
__global__ void k_fd_time_step(const float* p_prev, const float* p_cur,
                               float* p_next, const float* vdt2);

// KERNEL 2 — mirrors CPUReferenceRTM::apply_sponge.
__global__ void k_sponge(float* p_cur, float* p_next, const float* sponge,
                         std::size_t n);

// KERNEL 1+2 FUSED (cuda-v1, docs/CUDA_PLAN.md §6 rung V1). Same thread that
// computes p_next[i] also damps p_cur[i] and p_next[i] by sponge[i], right
// there in registers — no separate k_sponge launch, no extra global
// read/write pass. Bit-identical to k_fd_time_step + k_inject_* + k_sponge
// (V0) because source/receiver injection only ever touches points where
// sponge[i] == 1.0 exactly (they sit in the interior, never in the taper),
// so multiplying by w there commutes exactly with the later/earlier add —
// see src/cuda/kernels_v1_fused.cu for the full argument.
__global__ void k_fd_time_step_v1(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge);

// KERNEL 3 — mirrors CPUReferenceRTM::inject_source. Reads wavelet[it]
// straight from device memory (the "index d_wavelet_[it]" option in
// docs/CUDA_PLAN.md §3.4, kernel 3 note).
__global__ void k_inject_source(float* p, const float* vdt2,
                                std::size_t src_index, const float* wavelet, int it);

// KERNEL 4 — mirrors CPUReferenceRTM::inject_receivers. Uses atomicAdd:
// two receivers clamped onto the same cell must both add.
__global__ void k_inject_receivers(float* p, const float* vdt2,
                                   const int* rec_index, const float* traces,
                                   int nt_stride, int it, int nrec);

// KERNEL 5 — mirrors CPUReferenceRTM::record_traces.
__global__ void k_record_traces(const float* p, const int* rec_index,
                                float* out, int nt_stride, int it, int nrec);

// KERNEL 7 — mirrors CPUReferenceRTM::imaging (src/rtm/imaging.cpp).
// fwd_snapshot: interior grid (nx*nz). bwd_extended: extended grid (nxe*nze).
__global__ void k_imaging(const float* fwd_snapshot, const float* bwd_extended,
                          float* image, float* illum,
                          int nx, int nz, int nb, int nze);

// -----------------------------------------------------------------------------
// cuda-v2 (docs/CUDA_PLAN.md §6 rung V2) — imaging fused into the backward
// stencil+sponge kernel for every interior point EXCEPT this step's receiver
// locations, which are finished afterwards by k_image_unique_receivers once
// injection has actually landed. See src/cuda/kernels_v2_imaging.cu for why
// the split is required for bit-identity (imaging needs the post-injection
// value; the stencil kernel runs before injection).
// -----------------------------------------------------------------------------

// Scatters is_receiver[rec_index[ir]] = 1 for ir in [0,nrec). Called once per
// shot from map_geometry (extended-grid indices, one byte per point).
__global__ void k_mark_receivers(unsigned char* is_receiver, const int* rec_index, int nrec);

// KERNEL 1+2+7 FUSED, minus receiver points. Same as k_fd_time_step_v1, plus:
// for interior points (ix,iz within [nb,nb+nx) x [nb,nb+nz)) that are NOT
// marked in is_receiver, accumulate image/illum using the just-computed
// damped p_next value (no extra global read of the wavefield).
__global__ void k_fd_time_step_v2_image(const float* p_prev, float* p_cur, float* p_next,
                                        const float* vdt2, const float* sponge,
                                        const float* fwd_snapshot,
                                        const unsigned char* is_receiver,
                                        float* image, float* illum,
                                        int nx, int nz);

// Finishes imaging at the (deduplicated) set of interior points this shot's
// receivers touch this step, reading p_cur AFTER injection + rotation —
// exactly what k_imaging would have read for those specific points. One
// thread per UNIQUE extended-grid receiver index (deduplicated on the host
// in map_geometry) so a clamped-coincident pair of receivers still images
// that point exactly once, matching k_imaging's per-grid-point semantics.
__global__ void k_image_unique_receivers(const float* fwd_snapshot, const float* p_cur,
                                         const int* unique_ext_index, int nuniq,
                                         float* image, float* illum,
                                         int nz, int nb, int nze);

// -----------------------------------------------------------------------------
// cuda-v3 (docs/CUDA_PLAN.md §6 rung V3) — same kernels as v2, but the
// stencil reads p_cur through a shared-memory tile instead of global memory.
// See src/cuda/kernels_v3_shared.cu for the boundary-safety argument (every
// thread in the block, valid or not, must reach __syncthreads(), which
// changes how out-of-range threads have to be handled compared to v0-v2).
// -----------------------------------------------------------------------------
__global__ void k_fd_time_step_v3(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge);
__global__ void k_fd_time_step_v3_image(const float* p_prev, float* p_cur, float* p_next,
                                        const float* vdt2, const float* sponge,
                                        const float* fwd_snapshot,
                                        const unsigned char* is_receiver,
                                        float* image, float* illum,
                                        int nx, int nz);

// -----------------------------------------------------------------------------
// cuda-v4 (docs/CUDA_PLAN.md §6 rung V4) — z-neighbours via warp shuffle
// instead of the v3 shared tile (x-neighbours still come from the tile).
// See src/cuda/kernels_v4_regqueue.cu for why every lane must call the
// shuffle unconditionally, and for the same clamped-read pattern v3 needs.
// -----------------------------------------------------------------------------
__global__ void k_fd_time_step_v4(const float* p_prev, float* p_cur, float* p_next,
                                  const float* vdt2, const float* sponge);
__global__ void k_fd_time_step_v4_image(const float* p_prev, float* p_cur, float* p_next,
                                        const float* vdt2, const float* sponge,
                                        const float* fwd_snapshot,
                                        const unsigned char* is_receiver,
                                        float* image, float* illum,
                                        int nx, int nz);

} // namespace rtm
