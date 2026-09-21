// =============================================================================
// OPTIMIZED CPU ENGINE — see cpu_opt.hpp for the summary.
//
// Compiled with -O3 -march=native -ffp-contract=off (CMakeLists.txt), unlike
// the reference which stays at -O2 for reproducibility.
// =============================================================================
#include "cpu_opt.hpp"
#include "benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtm {

int CPUOptimizedRTM::num_threads() const {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// -----------------------------------------------------------------------------
// setup: the four tables are computed with EXACTLY the reference's formulas
// (src/rtm/rtm_cpu.cpp, steps 1-4). Copied, not shared, so the reference file
// stays untouched. Any difference here would show up as a non-identical image.
// -----------------------------------------------------------------------------
void CPUOptimizedRTM::setup(const VelocityModel& model, const RTMParams& par,
                            const TimeAxis& ta) {
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

    // 1. (v*dt)^2 on the extended grid, edge-replicated into the pad.
    vdt2_.assign(g_.n_extended(), 0.0f);
    const float dt2 = ta.dt * ta.dt;
    for (int ixe = 0; ixe < nxe; ++ixe) {
        const int ix = std::min(std::max(ixe - nb, 0), g_.nx - 1);
        for (int ize = 0; ize < nze; ++ize) {
            const int iz = std::min(std::max(ize - nb, 0), g_.nz - 1);
            const float v = model.v[g_.iidx(ix, iz)];
            vdt2_[g_.eidx(ixe, ize)] = v * v * dt2;
        }
    }

    // 2. Cerjan sponge, separable product of two 1D tapers; 1.0 in the interior.
    auto taper = [&](int i, int n_extended) {
        const int d = std::min(i, n_extended - 1 - i);
        if (d >= nb) return 1.0f;
        const float a = par.sponge_alpha * (float)(nb - d);
        return std::exp(-a * a);
    };
    std::vector<float> wx(nxe), wz(nze);
    for (int i = 0; i < nxe; ++i) wx[i] = taper(i, nxe);
    for (int i = 0; i < nze; ++i) wz[i] = taper(i, nze);
    sponge_.assign(g_.n_extended(), 1.0f);
    for (int ixe = 0; ixe < nxe; ++ixe)
        for (int ize = 0; ize < nze; ++ize)
            sponge_[g_.eidx(ixe, ize)] = wx[ixe] * wz[ize];

    // 3. Ricker wavelet delayed by 1.2/f0.
    wavelet_.resize(ta.nt);
    const float t0 = 1.2f / par.f0;
    for (int it = 0; it < ta.nt; ++it)
        wavelet_[it] = ricker((float)it * ta.dt - t0, par.f0);

    // 4. Stencil coefficients pre-divided by the grid spacing.
    const float* c = stencil_coefficients(par.order);
    const float idx2 = 1.0f / (g_.dx * g_.dx);
    const float idz2 = 1.0f / (g_.dz * g_.dz);
    c0_ = c[0] * (idx2 + idz2);
    for (int k = 1; k <= g_.half; ++k) { cx_[k] = c[k] * idx2; cz_[k] = c[k] * idz2; }

    // 5. Wavefield buffers.
    p_prev_.assign(g_.n_extended(), 0.0f);
    p_cur_.assign(g_.n_extended(), 0.0f);
    p_next_.assign(g_.n_extended(), 0.0f);
}

// Same nearest-grid-point mapping and clamp warning as the reference.
void CPUOptimizedRTM::map_geometry(const ShotRecord& shot) {
    int nclamped = 0;
    auto to_index = [&](float x, float z) -> std::size_t {
        const int ix0 = (int)std::lround((x - g_.ox) / g_.dx);
        const int iz0 = (int)std::lround((z - g_.oz) / g_.dz);
        const int ix  = std::min(std::max(ix0, 0), g_.nx - 1);
        const int iz  = std::min(std::max(iz0, 0), g_.nz - 1);
        if (ix != ix0 || iz != iz0) ++nclamped;
        return g_.eidx(ix + g_.nb, iz + g_.nb);
    };
    src_index_ = to_index(shot.sx, shot.sz);
    rec_index_.resize((std::size_t)shot.nrec());
    for (int ir = 0; ir < shot.nrec(); ++ir)
        rec_index_[ir] = to_index(shot.rx[ir], shot.rz[ir]);

    if (nclamped > 0 && par_.verbose) {
        std::fprintf(stderr,
            "WARNING: %d of %d source/receiver positions fell outside the "
            "%.1f x %.1f m model and were clamped to the edge.\n",
            nclamped, shot.nrec() + 1,
            (g_.nx - 1) * g_.dx, (g_.nz - 1) * g_.dz);
    }
}

// =============================================================================
// KERNEL 1+2 FUSED — stencil and sponge in one sweep.
//
// The reference does, per step:   p_next = 2 p_cur - p_prev + vdt2 * lap(p_cur)
//                                 [inject]
//                                 p_cur  *= w ;  p_next *= w        (two extra sweeps)
//
// Here the two sponge multiplies are moved without changing a single float
// operation:
//   * p_next[i] is damped right after it is computed:  p_next[i] = (...) * w[i]
//     — nobody reads p_next during the sweep, so this is safe. Injection
//     happens after the sweep at points where w == 1.0 exactly (they are in
//     the interior), so (stencil + inject) * 1 == stencil * 1 + inject.
//   * p_cur's in-place damping is DEFERRED: the reference multiplies p_cur by
//     w at step n and reads it as p_prev at step n+1. We leave the buffer
//     alone and multiply on read instead: p_prev[i] * w[i]. Same operands,
//     same single multiply, same result.
// Everything else that reads the rotated buffers (traces, snapshots, imaging)
// reads the damped p_next, exactly as in the reference.
//
// HALF is a template parameter so the k-loop unrolls completely and the
// compiler vectorizes the iz loop (depth is contiguous in memory).
// =============================================================================
template <int HALF>
static void fused_time_step_impl(int nxe, int nze,
                                 float c0, const float* cx, const float* cz,
                                 const float* __restrict__ p_prev,
                                 const float* __restrict__ p_cur,
                                 float* __restrict__ p_next,
                                 const float* __restrict__ vdt2,
                                 const float* __restrict__ sponge) {
#pragma omp parallel for schedule(static)
    for (int ix = HALF; ix < nxe - HALF; ++ix) {
        const std::size_t row = (std::size_t)ix * nze;
        for (int iz = HALF; iz < nze - HALF; ++iz) {
            const std::size_t i = row + iz;

            float lap = c0 * p_cur[i];
            for (int k = 1; k <= HALF; ++k) {
                const std::size_t kx = (std::size_t)k * nze;
                lap += cx[k] * (p_cur[i + kx] + p_cur[i - kx])   // d2/dx2
                     + cz[k] * (p_cur[i + k ] + p_cur[i - k ]);  // d2/dz2
            }
            const float damped_prev = p_prev[i] * sponge[i];             // deferred p_cur *= w
            const float stencil     = 2.0f * p_cur[i] - damped_prev + vdt2[i] * lap;
            p_next[i] = stencil * sponge[i];                              // p_next *= w
        }
    }
}

void CPUOptimizedRTM::fused_time_step(const float* p_prev, const float* p_cur,
                                      float* p_next) const {
    const int nxe = g_.nxe(), nze = g_.nze();
    switch (g_.half) {
        case 1: fused_time_step_impl<1>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        case 2: fused_time_step_impl<2>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        case 3: fused_time_step_impl<3>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        case 4: fused_time_step_impl<4>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        case 5: fused_time_step_impl<5>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        case 6: fused_time_step_impl<6>(nxe, nze, c0_, cx_, cz_, p_prev, p_cur, p_next, vdt2_.data(), sponge_.data()); break;
        default: throw std::runtime_error("cpu-opt: unsupported stencil order");
    }
}

// The injection points sit in the interior where the sponge weight is exactly
// 1.0, so adding after the damped write is identical to the reference order.
void CPUOptimizedRTM::inject_source(float* p, float amplitude) const {
    p[src_index_] += vdt2_[src_index_] * amplitude;
}

void CPUOptimizedRTM::inject_receivers(float* p, const ShotRecord& shot, int it) const {
    const std::size_t stride = (std::size_t)shot.nt;
    for (std::size_t ir = 0; ir < rec_index_.size(); ++ir) {
        const std::size_t idx = rec_index_[ir];
        p[idx] += vdt2_[idx] * shot.traces[ir * stride + (std::size_t)it];
    }
}

void CPUOptimizedRTM::record_traces(const float* p, int it, int nt_stride,
                                    float* out) const {
    for (std::size_t ir = 0; ir < rec_index_.size(); ++ir)
        out[ir * (std::size_t)nt_stride + it] = p[rec_index_[ir]];
}

// Extended -> interior copy: one contiguous memcpy per column, columns in parallel.
void CPUOptimizedRTM::save_snapshot(const float* p_extended, float* dst_interior) const {
    const int nb = g_.nb, nz = g_.nz, nze = g_.nze();
#pragma omp parallel for schedule(static)
    for (int ix = 0; ix < g_.nx; ++ix)
        std::memcpy(dst_interior + (std::size_t)ix * nz,
                    p_extended + (std::size_t)(ix + nb) * nze + nb,
                    (std::size_t)nz * sizeof(float));
}

void CPUOptimizedRTM::zero_wavefields() {
    std::fill(p_prev_.begin(), p_prev_.end(), 0.0f);
    std::fill(p_cur_.begin(),  p_cur_.end(),  0.0f);
    std::fill(p_next_.begin(), p_next_.end(), 0.0f);
}

// =============================================================================
// FORWARD DRIVER — same convention as the reference: step -> inject -> rotate.
// After the rotation p_cur_ is the (damped) field at time index `it`.
// =============================================================================
void CPUOptimizedRTM::forward_propagation(const ShotRecord& shot,
                                          std::vector<float>* snapshots,
                                          std::vector<float>* recorded) {
    ScopedAccumulator acc(times.forward);

    map_geometry(shot);
    zero_wavefields();

    const int nt          = ta_.nt;
    const int store       = par_.store_interval;
    const std::size_t nxz = g_.n_interior();

    for (int it = 0; it < nt; ++it) {
        fused_time_step(p_prev_.data(), p_cur_.data(), p_next_.data());
        inject_source(p_next_.data(), wavelet_[it]);

        p_prev_.swap(p_cur_);
        p_cur_.swap(p_next_);

        if (recorded)  record_traces(p_cur_.data(), it, nt, recorded->data());
        if (snapshots && (it % store) == 0)
            save_snapshot(p_cur_.data(),
                          snapshots->data() + (std::size_t)(it / store) * nxz);
    }
}

// =============================================================================
// BACKWARD DRIVER — it = nt-1 .. 0; imaging on stored steps as a separate
// parallel pass. (Fusing imaging into the stencil sweep would image the
// receiver cells BEFORE their injection and break bit-identity; at
// store_interval = 10 the separate pass is < 10 % of the traffic.)
// =============================================================================
void CPUOptimizedRTM::backward_propagation(const ShotRecord& shot,
                                           const std::vector<float>& snapshots,
                                           std::vector<float>& image,
                                           std::vector<float>& illumination) {
    Timer  wall;
    double imaging_time = 0.0;

    map_geometry(shot);
    zero_wavefields();

    const int store       = par_.store_interval;
    const std::size_t nxz = g_.n_interior();

    for (int it = ta_.nt - 1; it >= 0; --it) {
        fused_time_step(p_prev_.data(), p_cur_.data(), p_next_.data());
        inject_receivers(p_next_.data(), shot, it);

        p_prev_.swap(p_cur_);
        p_cur_.swap(p_next_);

        if ((it % store) == 0) {
            Timer t;
            imaging(snapshots.data() + (std::size_t)(it / store) * nxz,
                    p_cur_.data(), image, illumination);
            imaging_time += t.elapsed();
        }
    }

    times.imaging  += imaging_time;
    times.backward += wall.elapsed() - imaging_time;
}

// KERNEL 7 — zero-lag cross-correlation, parallel over columns.
void CPUOptimizedRTM::imaging(const float* fwd_snapshot, const float* bwd_extended,
                              std::vector<float>& image,
                              std::vector<float>& illumination) {
    const int nx = g_.nx, nz = g_.nz, nb = g_.nb, nze = g_.nze();
    float* __restrict__ image_data = image.data();
    float* __restrict__ illum_data = illumination.data();

#pragma omp parallel for schedule(static)
    for (int ix = 0; ix < nx; ++ix) {
        const float* __restrict__ fwd_col = fwd_snapshot + (std::size_t)ix * nz;
        const float* __restrict__ bwd_col = bwd_extended + (std::size_t)(ix + nb) * nze + nb;
        float* __restrict__ image_col = image_data + (std::size_t)ix * nz;
        float* __restrict__ illum_col = illum_data + (std::size_t)ix * nz;
        for (int iz = 0; iz < nz; ++iz) {
            const float f = fwd_col[iz];
            image_col[iz] += f * bwd_col[iz];
            illum_col[iz] += f * f;
        }
    }
}

} // namespace rtm
