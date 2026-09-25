// =============================================================================
// V0 — naive mirror of CPUReferenceRTM's propagation kernels + drivers.
// Same loop structure, same index convention (index = ix*nze + iz, depth
// fastest), same per-point operation order as src/rtm/forward.cpp and
// src/rtm/backward.cpp. One thread per grid point, plain global-memory
// loads, no shared memory, no fusion — see docs/CUDA_PLAN.md §3.
// =============================================================================
#include "rtm_cuda.hpp"
#include "cuda_check.hpp"
#include "kernels_cuda.hpp"
#include "benchmark.hpp"
#include <algorithm>
#include <cstdio>

namespace rtm {

// -----------------------------------------------------------------------------
// Stencil coefficients / grid constants, uploaded once per setup() (§2.2, §3.2).
// -----------------------------------------------------------------------------
__constant__ float c0_dev;
__constant__ float cx_dev[MAX_HALF + 1];
__constant__ float cz_dev[MAX_HALF + 1];
__constant__ int   half_dev;
__constant__ int   nxe_dev;
__constant__ int   nze_dev;
__constant__ int   nb_dev;

void upload_stencil_constants(float c0, const float* cx, const float* cz,
                              int half, int nxe, int nze, int nb) {
    CUDA_CHECK(cudaMemcpyToSymbol(c0_dev, &c0, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cx_dev, cx, sizeof(float) * (MAX_HALF + 1)));
    CUDA_CHECK(cudaMemcpyToSymbol(cz_dev, cz, sizeof(float) * (MAX_HALF + 1)));
    CUDA_CHECK(cudaMemcpyToSymbol(half_dev, &half, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(nxe_dev, &nxe, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(nze_dev, &nze, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(nb_dev, &nb, sizeof(int)));
}

// -----------------------------------------------------------------------------
// KERNEL 1 — the acoustic time step (docs/CUDA_PLAN.md §3.3, §3.4 #1).
// threadIdx.x maps to iz (depth, the fastest-varying axis): consecutive
// threads in a warp then read consecutive floats, i.e. a coalesced load.
// -----------------------------------------------------------------------------
__global__ void k_fd_time_step(const float* p_prev, const float* p_cur,
                               float* p_next, const float* vdt2) {
    const int iz = half_dev + blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = half_dev + blockIdx.y * blockDim.y + threadIdx.y;
    if (iz >= nze_dev - half_dev || ix >= nxe_dev - half_dev) return;

    const std::size_t i = (std::size_t)ix * nze_dev + iz;
    float lap = c0_dev * p_cur[i];
    for (int k = 1; k <= half_dev; ++k) {
        const std::size_t kx = (std::size_t)k * nze_dev;
        lap += cx_dev[k] * (p_cur[i + kx] + p_cur[i - kx])   // d2/dx2
             + cz_dev[k] * (p_cur[i + k ] + p_cur[i - k ]);  // d2/dz2
    }
    p_next[i] = 2.0f * p_cur[i] - p_prev[i] + vdt2[i] * lap;
}

// KERNEL 2 — Cerjan sponge, elementwise over the whole extended grid.
__global__ void k_sponge(float* p_cur, float* p_next, const float* sponge,
                         std::size_t n) {
    const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = sponge[i];
    p_cur[i]  *= w;
    p_next[i] *= w;
}

// KERNEL 3 — single-point source injection.
__global__ void k_inject_source(float* p, const float* vdt2,
                                std::size_t src_index, const float* wavelet, int it) {
    p[src_index] += vdt2[src_index] * wavelet[it];
}

// KERNEL 4 — receiver injection. atomicAdd because two receivers clamped
// onto the same extended-grid cell must both contribute (§3.7 checklist).
__global__ void k_inject_receivers(float* p, const float* vdt2,
                                   const int* rec_index, const float* traces,
                                   int nt_stride, int it, int nrec) {
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrec) return;
    const int idx = rec_index[ir];
    atomicAdd(&p[idx], vdt2[idx] * traces[(std::size_t)ir * nt_stride + it]);
}

// KERNEL 5 — trace recording (gather).
__global__ void k_record_traces(const float* p, const int* rec_index,
                                float* out, int nt_stride, int it, int nrec) {
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrec) return;
    out[(std::size_t)ir * nt_stride + it] = p[rec_index[ir]];
}

namespace {
constexpr int kStencilBlockZ = 32;   // one warp along z (contiguous)
constexpr int kStencilBlockX = 8;
constexpr int kSpongeThreads = 256;
constexpr int kRecThreads    = 128;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Picks the plain (non-imaging) fused stencil+sponge kernel for this rung —
// used by the forward driver always, and by the backward driver on every
// non-stored step (or for V0, every step). V2 reuses V1's kernel here since
// V2 only changes what happens on stored backward steps.
void launch_plain_stencil(CudaVariant variant, dim3 grid, dim3 block,
                          float* p_prev, float* p_cur, float* p_next,
                          const float* vdt2, const float* sponge) {
    switch (variant) {
        case CudaVariant::V4_RegQueue:
            k_fd_time_step_v4<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge);
            break;
        case CudaVariant::V3_Shared:
            k_fd_time_step_v3<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge);
            break;
        case CudaVariant::V0_Naive:
            k_fd_time_step<<<grid, block>>>(p_prev, p_cur, p_next, vdt2);
            break;
        case CudaVariant::V1_Fused:
        case CudaVariant::V2_Imaging:
            k_fd_time_step_v1<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge);
            break;
    }
}

// Picks the imaging-fused stencil kernel for a stored backward step, for
// whichever of V2/V3/V4 this engine is (see kernels_v2_imaging.cu for why
// imaging can't simply be "the same thread also does image += f*b" and has
// to skip receiver points here, finished by k_image_unique_receivers).
void launch_imaging_stencil(CudaVariant variant, dim3 grid, dim3 block,
                            float* p_prev, float* p_cur, float* p_next,
                            const float* vdt2, const float* sponge,
                            const float* fwd_snapshot, const unsigned char* is_receiver,
                            float* image, float* illum, int nx, int nz) {
    switch (variant) {
        case CudaVariant::V4_RegQueue:
            k_fd_time_step_v4_image<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge,
                                                      fwd_snapshot, is_receiver, image, illum, nx, nz);
            break;
        case CudaVariant::V3_Shared:
            k_fd_time_step_v3_image<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge,
                                                      fwd_snapshot, is_receiver, image, illum, nx, nz);
            break;
        case CudaVariant::V0_Naive:
        case CudaVariant::V1_Fused:
        case CudaVariant::V2_Imaging:
            k_fd_time_step_v2_image<<<grid, block>>>(p_prev, p_cur, p_next, vdt2, sponge,
                                                      fwd_snapshot, is_receiver, image, illum, nx, nz);
            break;
    }
}
} // namespace

// =============================================================================
// FORWARD DRIVER — device-resident. The host `snapshots` pointer is ignored
// (§3.5): the source wavefield always lands in d_snap_, the engine's own
// persistent device buffer, never copied back through a host pointer here.
// =============================================================================
void CUDARTM::forward_propagation(const ShotRecord& shot,
                                  std::vector<float>* /*snapshots*/,
                                  std::vector<float>* recorded) {
    ScopedAccumulator acc(times.forward);

    map_geometry(shot);

    const std::size_t next = g_.n_extended() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_pp_, 0, next));
    CUDA_CHECK(cudaMemset(d_pc_, 0, next));
    CUDA_CHECK(cudaMemset(d_pn_, 0, next));

    const int nt    = ta_.nt;
    const int store = par_.store_interval;
    const int nxe = g_.nxe(), nze = g_.nze(), half = g_.half;
    const int nrec  = shot.nrec();
    const std::size_t n_ext = g_.n_extended();

    const dim3 block(kStencilBlockZ, kStencilBlockX);
    const dim3 grid_stencil(ceil_div(nze - 2 * half, block.x),
                            ceil_div(nxe - 2 * half, block.y));
    const int sponge_blocks = (int)((n_ext + kSpongeThreads - 1) / kSpongeThreads);
    const int rec_blocks    = ceil_div(nrec, kRecThreads);

    float* d_rec_out = nullptr;
    if (recorded) {
        const std::size_t nbytes = (std::size_t)nrec * nt * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_rec_out, nbytes));
        CUDA_CHECK(cudaMemset(d_rec_out, 0, nbytes));
        record_device_memory();
    }

    for (int it = 0; it < nt; ++it) {
        // Forward never images, so the plain fused kernel is always right
        // here, whichever rung this engine is.
        launch_plain_stencil(variant_, grid_stencil, block, d_pp_, d_pc_, d_pn_, d_vdt2_, d_sponge_);
        CUDA_CHECK_KERNEL();
        k_inject_source<<<1, 1>>>(d_pn_, d_vdt2_, src_index_, d_wavelet_, it);
        CUDA_CHECK_KERNEL();
        if (variant_ == CudaVariant::V0_Naive) {
            k_sponge<<<sponge_blocks, kSpongeThreads>>>(d_pc_, d_pn_, d_sponge_, n_ext);
            CUDA_CHECK_KERNEL();
        }

        // rotate: pp <- pc, pc <- pn, pn <- (old pp, reused as scratch)
        std::swap(d_pp_, d_pc_);
        std::swap(d_pc_, d_pn_);

        if (recorded && nrec > 0) {
            k_record_traces<<<rec_blocks, kRecThreads>>>(d_pc_, d_rec_index_, d_rec_out, nt, it, nrec);
            CUDA_CHECK_KERNEL();
        }
        if ((it % store) == 0) {
            // KERNEL 6 — extended -> interior copy, straight device-to-device,
            // no kernel needed (§3.4 #6).
            const std::size_t slot = (std::size_t)(it / store) * g_.n_interior();
            CUDA_CHECK(cudaMemcpy2DAsync(
                d_snap_ + slot, (std::size_t)g_.nz * sizeof(float),
                d_pc_ + g_.eidx(g_.nb, g_.nb), (std::size_t)nze * sizeof(float),
                (std::size_t)g_.nz * sizeof(float), (std::size_t)g_.nx,
                cudaMemcpyDeviceToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (recorded) {
        recorded->assign((std::size_t)nrec * nt, 0.0f);
        CUDA_CHECK(cudaMemcpy(recorded->data(), d_rec_out,
                              (std::size_t)nrec * nt * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_rec_out));
    }
}

// =============================================================================
// BACKWARD DRIVER — device-resident. Host `snapshots`/`image`/`illumination`
// arguments are ignored (§3.5): d_snap_, d_image_, d_illum_ are the live
// copies. it = nt-1 .. 0, same rotation convention as forward so imaging
// correlates the two wavefields at the same physical time (§3.7).
// =============================================================================
void CUDARTM::backward_propagation(const ShotRecord& shot,
                                   const std::vector<float>& /*snapshots*/,
                                   std::vector<float>& /*image*/,
                                   std::vector<float>& /*illumination*/) {
    ScopedAccumulator acc(times.backward);

    map_geometry(shot);

    const std::size_t next = g_.n_extended() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_pp_, 0, next));
    CUDA_CHECK(cudaMemset(d_pc_, 0, next));
    CUDA_CHECK(cudaMemset(d_pn_, 0, next));

    const int nt    = ta_.nt;
    const int store = par_.store_interval;
    const int nxe = g_.nxe(), nze = g_.nze(), half = g_.half;
    const int nrec  = shot.nrec();
    const std::size_t n_ext = g_.n_extended();

    {
        Timer h2d;
        CUDA_CHECK(cudaMemcpy(d_traces_, shot.traces.data(),
                              (std::size_t)nrec * shot.nt * sizeof(float), cudaMemcpyHostToDevice));
        times.h2d += h2d.elapsed();
    }

    const dim3 block(kStencilBlockZ, kStencilBlockX);
    const dim3 grid_stencil(ceil_div(nze - 2 * half, block.x),
                            ceil_div(nxe - 2 * half, block.y));
    const dim3 grid_image(ceil_div(g_.nz, block.x), ceil_div(g_.nx, block.y));
    const int sponge_blocks = (int)((n_ext + kSpongeThreads - 1) / kSpongeThreads);
    const int rec_blocks    = ceil_div(nrec, kRecThreads);

    for (int it = nt - 1; it >= 0; --it) {
        const bool stored = (it % store) == 0;
        const std::size_t slot = (std::size_t)(it / store) * g_.n_interior();
        const bool fuses_imaging = uses_receiver_marker();   // V2, V3, V4

        if (fuses_imaging && stored) {
            // Stencil+sponge+imaging fused, skipping this step's receiver
            // points (finished below, after injection) — kernels_v2_imaging.cu
            // has the full bit-identity argument for why the split is needed.
            launch_imaging_stencil(variant_, grid_stencil, block, d_pp_, d_pc_, d_pn_, d_vdt2_, d_sponge_,
                                   d_snap_ + slot, d_is_receiver_, d_image_, d_illum_, g_.nx, g_.nz);
            CUDA_CHECK_KERNEL();
        } else {
            launch_plain_stencil(variant_, grid_stencil, block, d_pp_, d_pc_, d_pn_, d_vdt2_, d_sponge_);
            CUDA_CHECK_KERNEL();
        }
        if (nrec > 0) {
            k_inject_receivers<<<rec_blocks, kRecThreads>>>(d_pn_, d_vdt2_, d_rec_index_, d_traces_,
                                                             shot.nt, it, nrec);
            CUDA_CHECK_KERNEL();
        }
        if (variant_ == CudaVariant::V0_Naive) {
            k_sponge<<<sponge_blocks, kSpongeThreads>>>(d_pc_, d_pn_, d_sponge_, n_ext);
            CUDA_CHECK_KERNEL();
        }

        std::swap(d_pp_, d_pc_);
        std::swap(d_pc_, d_pn_);

        if (stored) {
            if (fuses_imaging) {
                // Finish exactly the points the fused kernel skipped, now
                // that injection has landed — one thread per unique
                // receiver-touched interior point this shot.
                if (nuniq_rec_ > 0) {
                    const int blocks = (nuniq_rec_ + kRecThreads - 1) / kRecThreads;
                    k_image_unique_receivers<<<blocks, kRecThreads>>>(
                        d_snap_ + slot, d_pc_, d_unique_rec_ext_, nuniq_rec_,
                        d_image_, d_illum_, g_.nz, g_.nb, nze);
                    CUDA_CHECK_KERNEL();
                }
            } else {
                k_imaging<<<grid_image, block>>>(d_snap_ + slot, d_pc_, d_image_, d_illum_,
                                                 g_.nx, g_.nz, g_.nb, nze);
                CUDA_CHECK_KERNEL();
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace rtm
