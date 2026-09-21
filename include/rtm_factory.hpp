#pragma once
#include <memory>
#include <string>
#include <vector>
#include "rtm_engine.hpp"

namespace rtm {

// Every optimization in the ladder is an engine with a name. `rtm --engine`
// and `rtm_synth --engine` pick one here, so all engines run through exactly
// the same driver, data and benchmark code.
//
//   cpu        CPUReferenceRTM   the correctness truth, never modified
//   cpu-opt    CPUOptimizedRTM   OpenMP + fused passes + vectorization   (Phase 1)
//   cuda-v0..  CUDARTM variants  the CUDA ladder                          (Phase 2)
//   cuda-multi MultiGPURTM       shot-parallel across GPUs                (Phase 3)
//
// Throws std::runtime_error for an unknown name or an engine compiled out.
std::unique_ptr<RTMEngine> make_engine(const std::string& engine_name,
                                       int requested_gpus = 1);

// Names accepted by make_engine in this build, in ladder order.
std::vector<std::string> list_engines();

} // namespace rtm
