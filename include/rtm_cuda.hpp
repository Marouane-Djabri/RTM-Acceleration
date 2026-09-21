#pragma once
#include "rtm_engine.hpp"

namespace rtm {

class CUDARTM : public RTMEngine {
public:
    ~CUDARTM() override;                     // cudaFree everything
    const char* name() const override { return "cuda-v0 (naive mirror)"; }

    // Benchmark CSV identity (rtm_engine.hpp).
    bool        is_gpu()      const override { return true; }
    int         num_devices() const override { return 1; }
    std::string device_name() const override;   // cudaDeviceProp::name of device 0

    void setup(const VelocityModel&, const RTMParams&, const TimeAxis&) override;
    void forward_propagation(const ShotRecord&, std::vector<float>* snapshots,
                             std::vector<float>* recorded) override;
    void backward_propagation(const ShotRecord&, const std::vector<float>& snapshots,
                              std::vector<float>& image,
                              std::vector<float>& illumination) override;
    void imaging(const float*, const float*, std::vector<float>&,
                 std::vector<float>&) override;      // host-side fallback, see §3.5
    void migrate(const std::vector<ShotRecord>&, std::vector<float>& image,
                 std::vector<float>& illumination) override;   // device-resident loop

private:
    void map_geometry(const ShotRecord&);    // host: same as CPU, then upload rec_index

    // device tables (extended grid unless noted)
    float* d_vdt2_   = nullptr;
    float* d_sponge_ = nullptr;
    float* d_wavelet_= nullptr;              // nt
    float* d_pp_ = nullptr, *d_pc_ = nullptr, *d_pn_ = nullptr;
    float* d_snap_   = nullptr;              // nsnap * nx*nz   (interior)
    float* d_traces_ = nullptr;              // nrec * nt, trace-major
    float* d_image_  = nullptr, *d_illum_ = nullptr;   // nx*nz (interior)
    int*   d_rec_index_ = nullptr;           // nrec, extended-grid indices
    int    src_index_ = 0;
    int    nrec_cap_  = 0;                   // current allocation size of d_traces_/d_rec_index_
};

} // namespace rtm