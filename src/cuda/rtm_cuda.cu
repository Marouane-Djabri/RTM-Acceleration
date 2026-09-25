// setup(), ~CUDARTM(), map_geometry(), migrate() — docs/CUDA_PLAN.md §3.2, §3.5.
#include "rtm_cuda.hpp"
#include "cuda_check.hpp"
#include "kernels_cuda.hpp"
#include "benchmark.hpp"
#include "nvtx_range.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace rtm {

const char* CUDARTM::name() const {
    switch (variant_) {
        case CudaVariant::V0_Naive:    return "cuda-v0 (naive mirror)";
        case CudaVariant::V1_Fused:    return "cuda-v1 (sponge fused into stencil)";
        case CudaVariant::V2_Imaging:  return "cuda-v2 (imaging fused into backward stencil)";
        case CudaVariant::V3_Shared:   return "cuda-v3 (shared-memory stencil tile)";
        case CudaVariant::V4_RegQueue: return "cuda-v4 (warp-shuffle z-neighbours)";
    }
    return "cuda";
}

// -----------------------------------------------------------------------------
// setup() reproduces CPUReferenceRTM::setup's four host tables VERBATIM
// (same formulas, same float arithmetic — do not "improve" them; see
// docs/CUDA_PLAN.md §3.2), then uploads each to its device buffer.
// -----------------------------------------------------------------------------
void CUDARTM::setup(const VelocityModel& model, const RTMParams& par,
                    const TimeAxis& ta) {
    // Fail here, cleanly, if there is no GPU (docs/CUDA_PLAN.md §2.5).
    CUDA_CHECK(cudaSetDevice(0));

    par_ = par;
    ta_  = ta;
    g_       = model.grid;
    g_.nb    = par.nb;
    g_.half  = stencil_half(par.order);

    if (g_.nb < g_.half + 1)
        throw std::runtime_error("--nb must be at least order/2 + 1");
    if (model.v.size() != g_.n_interior())
        throw std::runtime_error("velocity model size does not match nx*nz");
    if (par.store_interval < 1)
        throw std::runtime_error("--store-interval must be >= 1");

    nsnap_ = (ta.nt - 1) / par.store_interval + 1;

    const int nxe = g_.nxe(), nze = g_.nze(), nb = g_.nb;

    // 1. Extended (v*dt)^2 table.
    std::vector<float> vdt2(g_.n_extended(), 0.0f);
    const float dt2 = ta.dt * ta.dt;
    for (int ixe = 0; ixe < nxe; ++ixe) {
        const int ix = std::min(std::max(ixe - nb, 0), g_.nx - 1);
        for (int ize = 0; ize < nze; ++ize) {
            const int iz = std::min(std::max(ize - nb, 0), g_.nz - 1);
            const float v = model.v[g_.iidx(ix, iz)];
            vdt2[g_.eidx(ixe, ize)] = v * v * dt2;
        }
    }

    // 2. Cerjan sponge.
    auto taper = [&](int i, int n_extended) {
        const int d = std::min(i, n_extended - 1 - i);
        if (d >= nb) return 1.0f;
        const float a = par.sponge_alpha * (float)(nb - d);
        return std::exp(-a * a);
    };
    std::vector<float> wx(nxe), wz(nze);
    for (int i = 0; i < nxe; ++i) wx[i] = taper(i, nxe);
    for (int i = 0; i < nze; ++i) wz[i] = taper(i, nze);
    std::vector<float> sponge(g_.n_extended(), 1.0f);
    for (int ixe = 0; ixe < nxe; ++ixe)
        for (int ize = 0; ize < nze; ++ize)
            sponge[g_.eidx(ixe, ize)] = wx[ixe] * wz[ize];

    // 3. Ricker source wavelet.
    std::vector<float> wavelet(ta.nt);
    const float t0 = 1.2f / par.f0;
    for (int it = 0; it < ta.nt; ++it)
        wavelet[it] = ricker((float)it * ta.dt - t0, par.f0);

    // 4. Stencil coefficients, pre-divided by the grid spacing.
    const float* c = stencil_coefficients(par.order);
    const float idx2 = 1.0f / (g_.dx * g_.dx);
    const float idz2 = 1.0f / (g_.dz * g_.dz);
    const float c0 = c[0] * (idx2 + idz2);
    float cx[MAX_HALF + 1] = {};
    float cz[MAX_HALF + 1] = {};
    for (int k = 1; k <= g_.half; ++k) { cx[k] = c[k] * idx2; cz[k] = c[k] * idz2; }

    // 5. Device buffers: allocate, then upload.
    CUDA_CHECK(cudaMalloc(&d_vdt2_,    g_.n_extended() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sponge_,  g_.n_extended() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wavelet_, (std::size_t)ta.nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pp_,      g_.n_extended() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pc_,      g_.n_extended() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pn_,      g_.n_extended() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_snap_,    (std::size_t)nsnap_ * g_.n_interior() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_vdt2_, vdt2.data(), g_.n_extended() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sponge_, sponge.data(), g_.n_extended() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wavelet_, wavelet.data(), (std::size_t)ta.nt * sizeof(float),
                          cudaMemcpyHostToDevice));

    upload_stencil_constants(c0, cx, cz, g_.half, nxe, nze, nb);

    // cuda-v2+ only: per-point receiver marker, fixed size for the run's
    // lifetime (unlike d_rec_index_/d_traces_, which grow with nrec).
    if (uses_receiver_marker())
        CUDA_CHECK(cudaMalloc(&d_is_receiver_, g_.n_extended() * sizeof(unsigned char)));

    record_device_memory();
}

void CUDARTM::record_device_memory() {
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t used_bytes = total_bytes - free_bytes;
    if (used_bytes > peak_device_bytes_) peak_device_bytes_ = used_bytes;
}

std::string CUDARTM::device_name() const {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return "unknown-gpu";
    return std::string(properties.name);
}

CUDARTM::~CUDARTM() {
    cudaFree(d_vdt2_);
    cudaFree(d_sponge_);
    cudaFree(d_wavelet_);
    cudaFree(d_pp_);
    cudaFree(d_pc_);
    cudaFree(d_pn_);
    cudaFree(d_snap_);
    cudaFree(d_traces_);
    cudaFree(d_image_);
    cudaFree(d_illum_);
    cudaFree(d_rec_index_);
    cudaFree(d_is_receiver_);
    cudaFree(d_unique_rec_ext_);
}

// Same clamp logic as CPUReferenceRTM::map_geometry, then upload rec_index_
// to the device. d_rec_index_ / d_traces_ are (re)allocated only when nrec
// grows past the current capacity (the receiver set can change per shot in
// SEG-Y data — called at the start of both drivers).
void CUDARTM::map_geometry(const ShotRecord& shot) {
    int nclamped = 0;
    auto to_index = [&](float x, float z) -> int {
        const int ix0 = (int)std::lround((x - g_.ox) / g_.dx);
        const int iz0 = (int)std::lround((z - g_.oz) / g_.dz);
        const int ix  = std::min(std::max(ix0, 0), g_.nx - 1);
        const int iz  = std::min(std::max(iz0, 0), g_.nz - 1);
        if (ix != ix0 || iz != iz0) ++nclamped;
        return (int)g_.eidx(ix + g_.nb, iz + g_.nb);
    };
    src_index_ = to_index(shot.sx, shot.sz);

    const int nrec = shot.nrec();
    std::vector<int> rec_index((std::size_t)nrec);
    for (int ir = 0; ir < nrec; ++ir)
        rec_index[(std::size_t)ir] = to_index(shot.rx[ir], shot.rz[ir]);

    if (nrec > nrec_cap_) {
        CUDA_CHECK(cudaFree(d_rec_index_));
        CUDA_CHECK(cudaFree(d_traces_));
        d_rec_index_ = nullptr;
        d_traces_    = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rec_index_, (std::size_t)nrec * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_traces_,    (std::size_t)nrec * shot.nt * sizeof(float)));
        if (uses_receiver_marker()) {
            CUDA_CHECK(cudaFree(d_unique_rec_ext_));
            d_unique_rec_ext_ = nullptr;
            CUDA_CHECK(cudaMalloc(&d_unique_rec_ext_, (std::size_t)nrec * sizeof(int)));
        }
        nrec_cap_ = nrec;
        record_device_memory();
    }
    if (nrec > 0)
        CUDA_CHECK(cudaMemcpy(d_rec_index_, rec_index.data(), (std::size_t)nrec * sizeof(int),
                              cudaMemcpyHostToDevice));

    // cuda-v2+ only: rebuild the per-point receiver marker and the
    // deduplicated receiver list for this shot's geometry (docs/CUDA_PLAN.md
    // §6 rung V2 — see kernels_v2_imaging.cu for why both are needed).
    if (uses_receiver_marker()) {
        CUDA_CHECK(cudaMemsetAsync(d_is_receiver_, 0, g_.n_extended() * sizeof(unsigned char)));

        std::vector<int> unique_ext = rec_index;
        std::sort(unique_ext.begin(), unique_ext.end());
        unique_ext.erase(std::unique(unique_ext.begin(), unique_ext.end()), unique_ext.end());
        nuniq_rec_ = (int)unique_ext.size();
        if (nuniq_rec_ > 0)
            CUDA_CHECK(cudaMemcpy(d_unique_rec_ext_, unique_ext.data(),
                                  (std::size_t)nuniq_rec_ * sizeof(int), cudaMemcpyHostToDevice));

        if (nrec > 0) {
            const int blocks = (nrec + 127) / 128;
            k_mark_receivers<<<blocks, 128>>>(d_is_receiver_, d_rec_index_, nrec);
            CUDA_CHECK_KERNEL();
        }
    }

    if (nclamped > 0 && par_.verbose) {
        std::fprintf(stderr,
            "WARNING: %d of %d source/receiver positions fell outside the "
            "%.1f x %.1f m model and were clamped to the edge.\n"
            "         Check the coordinate units (SEG-Y scalco, or coordinates "
            "stored pre-multiplied) before trusting this image.\n",
            nclamped, nrec + 1,
            (g_.nx - 1) * g_.dx, (g_.nz - 1) * g_.dz);
    }
}

// Overrides the base RTMEngine::migrate (src/rtm/rtm.cpp) because the base
// version passes host buffers; this keeps image/illumination device-resident
// across every shot and only downloads once at the end (§3.5).
void CUDARTM::migrate(const std::vector<ShotRecord>& shots,
                      std::vector<float>& image, std::vector<float>& illumination) {
    Timer total;
    const std::size_t nxz = g_.n_interior();

    CUDA_CHECK(cudaMalloc(&d_image_, nxz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_illum_, nxz * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_image_, 0, nxz * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_illum_, 0, nxz * sizeof(float)));
    record_device_memory();

    image.assign(nxz, 0.0f);
    illumination.assign(nxz, 0.0f);

    for (std::size_t is = 0; is < shots.size(); ++is) {
        if (par_.verbose) {
            std::printf("  shot %2zu / %2zu   sx=%8.1f m  sz=%6.1f m  nrec=%d\n",
                        is + 1, shots.size(), shots[is].sx, shots[is].sz,
                        shots[is].nrec());
            std::fflush(stdout);
        }
        // snapshots/image/illumination arguments are ignored by these
        // overrides (device-resident); the real destinations are d_snap_,
        // d_image_, d_illum_.
        NvtxRange shot_range("shot " + std::to_string(is + 1));
        {
            NvtxRange range("forward");
            forward_propagation(shots[is], nullptr, nullptr);
        }
        {
            NvtxRange range("backward");
            backward_propagation(shots[is], {}, image, illumination);
        }
    }

    NvtxRange download_range("download image");
    Timer d2h;
    CUDA_CHECK(cudaMemcpy(image.data(),        d_image_, nxz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(illumination.data(), d_illum_, nxz * sizeof(float), cudaMemcpyDeviceToHost));
    times.d2h += d2h.elapsed();

    CUDA_CHECK(cudaFree(d_image_)); d_image_ = nullptr;
    CUDA_CHECK(cudaFree(d_illum_)); d_illum_ = nullptr;

    times.total += total.elapsed();
}

} // namespace rtm
