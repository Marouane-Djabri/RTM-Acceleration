// =============================================================================
// BACKWARD PROPAGATION
//        v
// computational kernel  (fd_time_step + inject_receivers)
//        v
// CUDA CANDIDATE #2  <-- same stencil kernel, different injection
// =============================================================================
#include "rtm_cpu.hpp"
#include "benchmark.hpp"
#include <algorithm>

namespace rtm {

// -----------------------------------------------------------------------------
// KERNEL 4 — receiver injection.
// The acoustic operator is self-adjoint, so back-propagation is the SAME
// stencil run with the recorded traces injected in reverse time order.
// Scatter into nrec points; on GPU this is a tiny kernel (or a fused tail).
// -----------------------------------------------------------------------------
void CPUReferenceRTM::inject_receivers(float* p, const ShotRecord& shot,
                                       int it) const {
    const std::size_t stride = (std::size_t)shot.nt;
    for (std::size_t ir = 0; ir < rec_index_.size(); ++ir) {
        const std::size_t idx = rec_index_[ir];
        p[idx] += vdt2_[idx] * shot.traces[ir * stride + (std::size_t)it];
    }
}

// =============================================================================
// BACKWARD DRIVER
//
// Runs it = nt-1 .. 0. After the rotation q_cur is the receiver wavefield "at
// time index it" — the same label the forward pass used — so the zero-lag
// cross-correlation is applied to two wavefields at the SAME physical time.
//
// Timing note: the imaging condition is nested inside this loop, so it is
// measured separately and SUBTRACTED from the backward total. `times.backward`
// therefore means "backward propagation excluding imaging".
// =============================================================================
void CPUReferenceRTM::backward_propagation(const ShotRecord& shot,
                                           const std::vector<float>& snapshots,
                                           std::vector<float>& image,
                                           std::vector<float>& illumination) {
    Timer  wall;
    double imaging_time = 0.0;

    map_geometry(shot);
    // The source wavefield is already stored, so the same three buffers are
    // reused for the receiver wavefield.
    std::fill(pp_.begin(), pp_.end(), 0.0f);
    std::fill(pc_.begin(), pc_.end(), 0.0f);
    std::fill(pn_.begin(), pn_.end(), 0.0f);

    const int store       = par_.store_interval;
    const std::size_t nxz = g_.n_interior();

    for (int it = ta_.nt - 1; it >= 0; --it) {
        fd_time_step(pp_.data(), pc_.data(), pn_.data());
        inject_receivers(pn_.data(), shot, it);
        apply_sponge(pc_.data(), pn_.data());

        pp_.swap(pc_);
        pc_.swap(pn_);

        if ((it % store) == 0) {
            Timer t;
            imaging(snapshots.data() + (std::size_t)(it / store) * nxz,
                    pc_.data(), image, illumination);
            imaging_time += t.elapsed();
        }
    }

    times.imaging  += imaging_time;
    times.backward += wall.elapsed() - imaging_time;
}

} // namespace rtm