#pragma once
// Named ranges on the Nsight Systems timeline (docs/PROFILING_STRATEGY.md M1).
//
//     {
//         NvtxRange range("forward");
//         ... everything in this scope shows up under "forward" ...
//     }
//
// Built with RTM_WITH_NVTX (CMake option, on by default when CUDA is found),
// this pushes/pops an NVTX range. Without it, the class does nothing, so the
// CPU-only build needs no CUDA headers. NVTX costs well under a microsecond
// per range when no profiler is attached; we only open a handful per shot.
#include <string>

#ifdef RTM_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace rtm {

class NvtxRange {
public:
    explicit NvtxRange(const std::string& name) {
#ifdef RTM_WITH_NVTX
        nvtxRangePushA(name.c_str());
#else
        (void)name;
#endif
    }
    ~NvtxRange() {
#ifdef RTM_WITH_NVTX
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace rtm
