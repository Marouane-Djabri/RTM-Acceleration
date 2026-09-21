#pragma once
#include "rtm_engine.hpp"

namespace rtm {

// =============================================================================
// CPU REFERENCE IMPLEMENTATION (Phase 5)
//
// Priorities: correctness > readability > reproducibility.
// Deliberately NOT optimized: no OpenMP, no blocking, no intrinsics,
// no restrict qualifiers, no fast-math. This is the ground truth.
//
// Every method in the "computational kernels" block below is a 1:1 candidate
// for a CUDA kernel (Phase 8).
// =============================================================================
class CPUReferenceRTM : public RTMEngine {
public:
    const char* name() const override { return "CPU reference"; }

    void setup(const VelocityModel& model, const RTMParams& par,
               const TimeAxis& ta) override;

    void forward_propagation(const ShotRecord& shot,
                             std::vector<float>* snapshots,
                             std::vector<float>* recorded) override;

    void backward_propagation(const ShotRecord& shot,
                              const std::vector<float>& snapshots,
                              std::vector<float>& image,
                              std::vector<float>& illumination) override;

    void imaging(const float* fwd_snapshot, const float* bwd_extended,
                 std::vector<float>& image,
                 std::vector<float>& illumination) override;

    // ---------------- computational kernels (CUDA candidates) ---------------
    // KERNEL 1 — >90% of the runtime. Port this first.
    void fd_time_step(const float* p_prev, const float* p_cur, float* p_next) const;
    // KERNEL 2 — trivially parallel, elementwise.
    void apply_sponge(float* p_cur, float* p_next) const;
    // KERNEL 3 — single point scatter.
    void inject_source(float* p, float amplitude) const;
    // KERNEL 4 — nrec-point scatter.
    void inject_receivers(float* p, const ShotRecord& shot, int it) const;
    // KERNEL 5 — nrec-point gather.
    void record_traces(const float* p, int it, int nt_stride, float* out) const;
    // KERNEL 6 — extended -> interior copy (a D2H candidate on GPU).
    void save_snapshot(const float* p_extended, float* dst_interior) const;
    // KERNEL 7 — imaging(), see above.
    // ------------------------------------------------------------------------

    const std::vector<float>& wavelet() const { return wavelet_; }

private:
    void map_geometry(const ShotRecord& shot);

    std::vector<float> vdt2_;    // (v*dt)^2 on the extended grid
    std::vector<float> sponge_;  // Cerjan damping on the extended grid
    std::vector<float> wavelet_; // Ricker source, nt samples
    std::vector<float> pp_, pc_, pn_;   // p^{n-1}, p^{n}, p^{n+1} (extended)

    std::size_t              src_index_ = 0;  // extended-grid index
    std::vector<std::size_t> rec_index_;      // extended-grid indices

    float c0_ = 0.0f;              // c[0] * (1/dx^2 + 1/dz^2)
    float cx_[MAX_HALF + 1] = {};  // c[k] / dx^2
    float cz_[MAX_HALF + 1] = {};  // c[k] / dz^2
};

} // namespace rtm