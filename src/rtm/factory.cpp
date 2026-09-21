#include "rtm_factory.hpp"
#include "rtm_cpu.hpp"
#ifdef RTM_WITH_CPU_OPT
#include "cpu_opt.hpp"
#endif
#ifdef RTM_WITH_CUDA
#include "rtm_cuda.hpp"
#endif
#include <stdexcept>

// Engines added by later phases are registered here as they appear:
//   cuda-v1..v4  Phase 2   CUDARTM variants
//   cuda-multi   Phase 3   include "rtm_cuda_multi.hpp"

namespace rtm {

std::unique_ptr<RTMEngine> make_engine(const std::string& engine_name,
                                       int requested_gpus) {
    (void)requested_gpus;   // used by cuda-multi only
    if (engine_name == "cpu") return std::make_unique<CPUReferenceRTM>();
#ifdef RTM_WITH_CPU_OPT
    if (engine_name == "cpu-opt") return std::make_unique<CPUOptimizedRTM>();
#endif
#ifdef RTM_WITH_CUDA
    if (engine_name == "cuda-v0" || engine_name == "cuda")
        return std::make_unique<CUDARTM>();
#endif
    throw std::runtime_error("unknown or unavailable engine: '" + engine_name +
                             "' (see --list-engines)");
}

std::vector<std::string> list_engines() {
    std::vector<std::string> names;
    names.push_back("cpu");
#ifdef RTM_WITH_CPU_OPT
    names.push_back("cpu-opt");
#endif
#ifdef RTM_WITH_CUDA
    names.push_back("cuda-v0");
#endif
    return names;
}

} // namespace rtm
