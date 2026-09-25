#pragma once
#include "rtm_engine.hpp"

namespace rtm {

// Selects which kernel set CUDARTM's drivers dispatch to (docs/CUDA_PLAN.md
// §6 / OPTIMIZATION_PLAN.md §4 "Phase 2 — CUDA ladder"). Every variant must
// stay bit-identical to the one before it; only the kernel set differs.
enum class CudaVariant {
    V0_Naive,    // cuda-v0: separate k_sponge launch (docs/CUDA_PLAN.md §3)
    V1_Fused,    // cuda-v1: sponge multiply folded into the stencil kernel
    V2_Imaging,  // cuda-v2: imaging fused into the backward stencil kernel
                 //          (except at this step's receiver points, finished
                 //          right after injection — see kernels_v2_imaging.cu)
    V3_Shared,   // cuda-v3: stencil reads p_cur from a shared-memory tile
                 //          instead of global memory (kernels_v3_shared.cu)
    V4_RegQueue, // cuda-v4: z-neighbours via warp shuffle instead of the
                 //          shared tile; x-neighbours still from the tile
                 //          (kernels_v4_regqueue.cu)
};

class CUDARTM : public RTMEngine {
public:
    explicit CUDARTM(CudaVariant variant = CudaVariant::V0_Naive) : variant_(variant) {}
    ~CUDARTM() override;                     // cudaFree everything
    const char* name() const override;

    // Benchmark CSV identity (rtm_engine.hpp).
    bool        is_gpu()      const override { return true; }
    int         num_devices() const override { return 1; }
    std::string device_name() const override;   // cudaDeviceProp::name of device 0
    std::size_t peak_device_bytes() const override { return peak_device_bytes_; }

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

    // Reads cudaMemGetInfo and keeps the highest "used" value seen. Called
    // right after every group of cudaMalloc calls (docs/PROFILING_STRATEGY.md M2).
    void record_device_memory();
    std::size_t peak_device_bytes_ = 0;

    // V2 and every rung built on it (V3, V4) fuse imaging into the backward
    // stencil kernel and so need the per-shot receiver marker / unique list
    // from kernels_v2_imaging.cu; V0/V1 never do.
    bool uses_receiver_marker() const {
        return variant_ == CudaVariant::V2_Imaging ||
               variant_ == CudaVariant::V3_Shared  ||
               variant_ == CudaVariant::V4_RegQueue;
    }

    CudaVariant variant_;

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

    // cuda-v2 only (docs/CUDA_PLAN.md §6 rung V2, kernels_v2_imaging.cu).
    unsigned char* d_is_receiver_    = nullptr;   // n_extended, 1 at this shot's receiver indices
    int*           d_unique_rec_ext_ = nullptr;   // <= nrec_cap_, deduplicated extended indices
    int            nuniq_rec_        = 0;         // valid entries in d_unique_rec_ext_ this shot
};

} // namespace rtm