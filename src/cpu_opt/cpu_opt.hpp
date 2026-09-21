#pragma once
#include "rtm_engine.hpp"
#include <string>
#include <vector>

namespace rtm {

// =============================================================================
// OPTIMIZED CPU ENGINE  (`--engine cpu-opt`, docs/OPTIMIZATION_PLAN.md §3)
//
// The "best CPU" bar every GPU rung is judged against. Same algorithm and
// same tables as CPUReferenceRTM, with everything that helps on a multi-core
// CPU applied at once:
//
//   * OpenMP over ix in every grid-sized loop (stencil, imaging, snapshot copy)
//   * the Cerjan sponge fused into the stencil pass, which removes two full
//     read+write sweeps of the wavefield per time step (see fused_time_step)
//   * compile-time stencil order (template on HALF) so the k-loop unrolls and
//     the inner iz loop auto-vectorizes; this file is compiled with
//     -O3 -march=native -ffp-contract=off (CMakeLists.txt)
//
// Result: bit-identical to the reference image by construction (same float
// operations in the same order, no FMA contraction). If a run is not
// bit-identical, something is wrong — that is the point of keeping it exact.
// =============================================================================
class CPUOptimizedRTM : public RTMEngine {
public:
    const char* name() const override { return "cpu-opt (OpenMP + fused sponge + vectorized)"; }
    int         num_threads() const override;

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

private:
    void map_geometry(const ShotRecord& shot);

    // One time step = stencil + source/receiver injection hook + sponge, all
    // in one sweep. Dispatches on the stencil half-width to a template.
    void fused_time_step(const float* p_prev, const float* p_cur, float* p_next) const;

    void inject_source(float* p, float amplitude) const;
    void inject_receivers(float* p, const ShotRecord& shot, int it) const;
    void record_traces(const float* p, int it, int nt_stride, float* out) const;
    void save_snapshot(const float* p_extended, float* dst_interior) const;
    void zero_wavefields();

    // Tables, identical to CPUReferenceRTM's (rtm_cpu.cpp setup, steps 1-4).
    std::vector<float> vdt2_;      // (v*dt)^2 on the extended grid
    std::vector<float> sponge_;    // Cerjan weights on the extended grid, 1.0 in the interior
    std::vector<float> wavelet_;   // Ricker source, nt samples
    float c0_ = 0.0f;
    float cx_[MAX_HALF + 1] = {};
    float cz_[MAX_HALF + 1] = {};

    // Wavefield buffers p^{n-1}, p^n, p^{n+1} on the extended grid. NOTE: the
    // p^{n-1} buffer holds values damped ONCE (at creation); the second sponge
    // multiply the reference applies in place is applied on read instead.
    std::vector<float> p_prev_, p_cur_, p_next_;

    std::size_t              src_index_ = 0;
    std::vector<std::size_t> rec_index_;
};

} // namespace rtm
