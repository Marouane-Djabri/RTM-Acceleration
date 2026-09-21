#pragma once
#include <vector>
#include "rtm_types.hpp"

namespace rtm {

// Wall-clock accounting for every RTM stage (Phase 6).
struct StageTimes {
    double io = 0, forward = 0, backward = 0, imaging = 0, total = 0;
    // Reserved for the CUDA engines (Phase 8) — stay 0 on CPU.
    double h2d = 0, kernel = 0, d2h = 0;
    void reset() { *this = StageTimes(); }
};

// -----------------------------------------------------------------------------
// The single abstraction of the project (Phase 4).
// CPUReferenceRTM / CPUOptimizedRTM / CUDARTM / CUDAOptimizedRTM all implement
// this and must produce numerically comparable images.
// -----------------------------------------------------------------------------
class RTMEngine {
public:
    virtual ~RTMEngine() = default;
    virtual const char* name() const = 0;

    virtual void setup(const VelocityModel& model,
                       const RTMParams& par,
                       const TimeAxis& ta) = 0;

    // Propagate the source wavefield.
    //   snapshots : if non-null, receives the interior wavefield every
    //               store_interval steps, size num_snapshots()*nx*nz
    //   recorded  : if non-null, receives modelled traces (nrec*nt, trace-major)
    virtual void forward_propagation(const ShotRecord& shot,
                                     std::vector<float>* snapshots,
                                     std::vector<float>* recorded) = 0;

    // Back-propagate the recorded data and accumulate the image.
    virtual void backward_propagation(const ShotRecord& shot,
                                      const std::vector<float>& snapshots,
                                      std::vector<float>& image,
                                      std::vector<float>& illumination) = 0;

    // Zero-lag cross-correlation for ONE time step.
    //   fwd_snapshot : interior grid  (nx*nz)
    //   bwd_extended : extended grid  (nxe*nze)
    virtual void imaging(const float* fwd_snapshot,
                         const float* bwd_extended,
                         std::vector<float>& image,
                         std::vector<float>& illumination) = 0;

    // Full migration over all shots. Default implementation in src/rtm/rtm.cpp;
    // a CUDA engine may override it to keep data resident on the device.
    virtual void migrate(const std::vector<ShotRecord>& shots,
                         std::vector<float>& image,
                         std::vector<float>& illumination);

    // Forward modelling only (used to build the synthetic dataset).
    virtual void model_shot(ShotRecord& shot);

    const Grid&      grid()          const { return g_; }
    const RTMParams& params()        const { return par_; }
    const TimeAxis&  time_axis()     const { return ta_; }
    int              num_snapshots() const { return nsnap_; }
    std::size_t      snapshot_bytes() const {
        return (std::size_t)nsnap_ * g_.n_interior() * sizeof(float);
    }

    // Identity of the run for the benchmark CSV. The reference is single
    // threaded on the CPU; threaded and GPU engines override these.
    virtual int         num_threads() const { return 1; }
    virtual int         num_devices() const { return 0; }       // GPUs used
    virtual std::string device_name() const { return "cpu"; }   // e.g. "NVIDIA A4000"
    virtual bool        is_gpu()      const { return false; }   // print H2D/D2H rows

    StageTimes times;

protected:
    Grid       g_;
    RTMParams  par_;
    TimeAxis   ta_;
    int        nsnap_ = 0;
    std::vector<float> snap_;   // source wavefield history
};

// ---- image post-processing (src/rtm/imaging.cpp) ----------------------------
void illumination_compensation(std::vector<float>& image,
                               const std::vector<float>& illumination,
                               float eps_relative);
void laplacian_filter(std::vector<float>& image, const Grid& g);

} // namespace rtm