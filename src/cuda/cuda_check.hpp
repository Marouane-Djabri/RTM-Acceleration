#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Wrap every CUDA API call in this — no exceptions (see docs/CUDA_PLAN.md §2.3).
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess)                                                  \
            throw std::runtime_error(std::string(#call) + ": " +                \
                                     cudaGetErrorString(e_) + " (" +             \
                                     __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
    } while (0)

// After every kernel launch: always check the launch itself; only pay for a
// full synchronize (needed to catch async faults like illegal memory access)
// under RTM_CUDA_SYNC_DEBUG, since it kills performance in a release build.
#ifdef RTM_CUDA_SYNC_DEBUG
#define CUDA_CHECK_KERNEL() do { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
#else
#define CUDA_CHECK_KERNEL() do { CUDA_CHECK(cudaGetLastError()); } while (0)
#endif
