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

} // namespace rtm
