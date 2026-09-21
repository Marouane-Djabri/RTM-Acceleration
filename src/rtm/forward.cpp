// =============================================================================
// FORWARD PROPAGATION
//        v
// computational kernel  (fd_time_step)
//        v
// CUDA CANDIDATE #1  <-- port this first, it is >90% of the runtime
// =============================================================================
#include "rtm_cpu.hpp"
#include "benchmark.hpp"
#include <algorithm>

namespace rtm {

// -----------------------------------------------------------------------------
// KERNEL 1 — the acoustic time step.
//
//   p^{n+1}_i = 2 p^n_i - p^{n-1}_i + (c_i dt)^2 * Laplacian(p^n)_i
//
// This is the single hottest loop in the whole program.
// Flops per point per step: 6*half + 5   (29 for order 8)
// Bytes per point per step: ~16 (p_prev, p_cur, vdt2 read + p_next write)
//  -> arithmetic intensity ~1.8 flop/byte => MEMORY BOUND on any modern CPU.
//
// CUDA mapping: one thread per grid point, 2D block (e.g. 32x16),
// halo of `half` cells loaded into shared memory, or a z-register-queue
// sweep. See README "CUDA roadmap".
// Deliberately left plain here: no OpenMP, no blocking, no intrinsics.
// -----------------------------------------------------------------------------
void CPUReferenceRTM::fd_time_step(const float* p_prev, const float* p_cur,
                                   float* p_next) const {
    const int nxe = g_.nxe();
    const int nze = g_.nze();
    const int h   = g_.half;

    for (int ix = h; ix < nxe - h; ++ix) {
        for (int iz = h; iz < nze - h; ++iz) {
            const std::size_t i = (std::size_t)ix * nze + iz;

            float lap = c0_ * p_cur[i];
            for (int k = 1; k <= h; ++k) {
                const std::size_t kx = (std::size_t)k * nze;
                lap += cx_[k] * (p_cur[i + kx] + p_cur[i - kx])   // d2/dx2
                     + cz_[k] * (p_cur[i + k ] + p_cur[i - k ]);  // d2/dz2
            }
            p_next[i] = 2.0f * p_cur[i] - p_prev[i] + vdt2_[i] * lap;
        }
    }
    // The outermost `half` cells are never updated and stay 0. They sit deep
    // inside the sponge, so the rigid edge they represent is unreachable.
}

// -----------------------------------------------------------------------------
// KERNEL 2 — Cerjan absorbing boundary. Elementwise, trivially parallel.
// CUDA: fold into KERNEL 1 to avoid two extra passes over the wavefield.
// -----------------------------------------------------------------------------
void CPUReferenceRTM::apply_sponge(float* p_cur, float* p_next) const {
    const std::size_t n = g_.n_extended();
    for (std::size_t i = 0; i < n; ++i) {
        const float w = sponge_[i];
        p_cur[i]  *= w;
        p_next[i] *= w;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 3 — source injection.
// The dt^2*c^2 factor is the discrete form of the delta source term.
// (A constant scale factor; it does not change the image geometry.)
// -----------------------------------------------------------------------------
void CPUReferenceRTM::inject_source(float* p, float amplitude) const {
    p[src_index_] += vdt2_[src_index_] * amplitude;
}

// KERNEL 5 — trace recording (gather at receiver positions).
void CPUReferenceRTM::record_traces(const float* p, int it, int nt_stride,
                                    float* out) const {
    for (std::size_t ir = 0; ir < rec_index_.size(); ++ir)
        out[ir * (std::size_t)nt_stride + it] = p[rec_index_[ir]];
}

// KERNEL 6 — extended grid -> interior snapshot copy.
// On GPU this is the wavefield-storage bottleneck (D2H traffic).
void CPUReferenceRTM::save_snapshot(const float* p_ext, float* dst) const {
    const int nb = g_.nb;
    for (int ix = 0; ix < g_.nx; ++ix)
        for (int iz = 0; iz < g_.nz; ++iz)
            dst[g_.iidx(ix, iz)] = p_ext[g_.eidx(ix + nb, iz + nb)];
}

// =============================================================================
// FORWARD DRIVER
//
// Time-stepping convention (must match backward_propagation exactly):
//   step -> inject -> damp -> rotate.  After the rotation p_cur is the
//   wavefield "at time index it", i.e. the field that contains source sample it.
// =============================================================================
void CPUReferenceRTM::forward_propagation(const ShotRecord& shot,
                                          std::vector<float>* snapshots,
                                          std::vector<float>* recorded) {
    ScopedAccumulator acc(times.forward);

    map_geometry(shot);
    std::fill(pp_.begin(), pp_.end(), 0.0f);
    std::fill(pc_.begin(), pc_.end(), 0.0f);
    std::fill(pn_.begin(), pn_.end(), 0.0f);

    const int nt          = ta_.nt;
    const int store       = par_.store_interval;
    const std::size_t nxz = g_.n_interior();

    for (int it = 0; it < nt; ++it) {
        fd_time_step(pp_.data(), pc_.data(), pn_.data());
        inject_source(pn_.data(), wavelet_[it]);
        apply_sponge(pc_.data(), pn_.data());

        // rotate: pp <- pc, pc <- pn, pn <- (old pp, reused as scratch)
        pp_.swap(pc_);
        pc_.swap(pn_);

        if (recorded)  record_traces(pc_.data(), it, nt, recorded->data());
        if (snapshots && (it % store) == 0)
            save_snapshot(pc_.data(),
                          snapshots->data() + (std::size_t)(it / store) * nxz);
    }
}

} // namespace rtm