#pragma once
#include <chrono>
#include <ostream>
#include <string>
#include <vector>
#include "rtm_engine.hpp"

namespace rtm {

// High-resolution monotonic timing (steady_clock is guaranteed monotonic;
// system_clock is not and must never be used for benchmarking).
class Timer {
public:
    Timer() { reset(); }
    void   reset()         { t0_ = clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(clock::now() - t0_).count();
    }
private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_;
};

class ScopedAccumulator {
public:
    explicit ScopedAccumulator(double& acc) : acc_(acc) {}
    ~ScopedAccumulator() { acc_ += t_.elapsed(); }
private:
    double& acc_;
    Timer   t_;
};

struct EngineResult {
    std::string name;
    StageTimes  t;
    bool        gpu = false;   // print H2D/Kernel/D2H rows
    std::size_t peak_device_bytes = 0;   // RTMEngine::peak_device_bytes(), 0 on CPU
};

struct BenchmarkContext {
    int   nx = 0, nz = 0, nb = 0, nt = 0, nshots = 0, order = 0;
    int   store_interval = 1;
    float dx = 0, dz = 0, dt = 0;
    std::size_t snapshot_bytes = 0;
    std::vector<EngineResult> results;   // results[0] is the reference
};

void print_benchmark_report(std::ostream& os, const BenchmarkContext& ctx);

// ---- machine-readable output (Phase 0) --------------------------------------
// One CSV row per run. The plotting script joins benchmark rows and
// comparison rows on (host, engine, dataset).
struct RunInfo {
    std::string host;        // gethostname()
    std::string cpu_name;    // /proc/cpuinfo "model name"
    std::string gpu_name;    // engine->device_name(), "cpu" for CPU engines
    std::string engine;      // --engine name
    std::string dataset;     // --dataset label (default: shots file stem)
    int threads = 1;         // engine->num_threads()
    int gpus    = 0;         // engine->num_devices()
};

std::string host_name();
std::string cpu_model_name();
std::string date_now_iso();

void append_benchmark_csv(const std::string& path, const BenchmarkContext& ctx,
                          const EngineResult& result, const RunInfo& info);

} // namespace rtm