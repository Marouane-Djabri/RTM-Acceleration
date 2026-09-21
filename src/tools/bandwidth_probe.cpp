// =============================================================================
// MEMORY BANDWIDTH PROBE
//
// The stencil is memory-bound, so "how close to the roofline is this engine"
// needs the bandwidth the machine can actually deliver, not a datasheet
// number. This measures it the same way on every host:
//
//   CPU : STREAM-style triad  a[i] = b[i] + scale * c[i]   over 3 x 256 MB,
//         all OpenMP threads, best of `repeats`.
//   GPU : copy kernel  dst[i] = src[i]  over 2 x 256 MB, best of `repeats`
//         (only when built with CUDA; see bandwidth_probe_cuda.cu).
//
// Output: one CSV row per device appended to --csv PATH:
//     date,host,device_kind,device_name,threads,gbps
// =============================================================================
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef RTM_WITH_CUDA
namespace rtm {
// Defined in bandwidth_probe_cuda.cu. Returns false (with a message) when no
// usable GPU is present, so the CPU number is still written.
bool probe_gpu_copy_bandwidth(std::size_t bytes_per_buffer, int repeats,
                              std::string& gpu_name, double& gbps, std::string& error);
}
#endif

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Triad bandwidth in GB/s: 3 arrays touched (2 reads + 1 write) per element.
double probe_cpu_triad_bandwidth(std::size_t elements, int repeats, int& threads_used) {
    std::vector<float> a(elements), b(elements, 1.0f), c(elements, 2.0f);
    const float scale = 3.0f;
    threads_used = 1;
#ifdef _OPENMP
    threads_used = omp_get_max_threads();
#endif

    double best_seconds = 1e30;
    for (int repeat = 0; repeat < repeats + 1; ++repeat) {   // +1: first pass warms pages
        const auto start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < (long long)elements; ++i)
            a[(std::size_t)i] = b[(std::size_t)i] + scale * c[(std::size_t)i];
        const double elapsed = seconds_since(start);
        if (repeat > 0 && elapsed < best_seconds) best_seconds = elapsed;
    }
    // Keep the compiler from dropping the loop.
    volatile float sink = a[elements / 2];
    (void)sink;

    const double bytes_moved = 3.0 * (double)elements * sizeof(float);
    return bytes_moved / best_seconds / 1e9;
}

void append_csv_row(const std::string& path, const std::string& device_kind,
                    const std::string& device_name, int threads, double gbps) {
    bool need_header = true;
    {
        std::ifstream existing(path);
        if (existing && existing.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot append bandwidth csv: " + path);
    if (need_header) out << "date,host,device_kind,device_name,threads,gbps\n";
    std::string safe_name = device_name;
    for (char& ch : safe_name) if (ch == ',') ch = ' ';
    out << rtm::date_now_iso() << ',' << rtm::host_name() << ',' << device_kind << ','
        << safe_name << ',' << threads << ',' << gbps << '\n';
}

} // namespace

int main(int argc, char** argv) try {
    std::string csv_path;
    std::size_t megabytes = 256;
    int repeats = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--csv"     && i + 1 < argc) csv_path  = argv[++i];
        else if (a == "--mb"      && i + 1 < argc) megabytes = (std::size_t)std::stoul(argv[++i]);
        else if (a == "--repeats" && i + 1 < argc) repeats   = std::stoi(argv[++i]);
        else {
            std::printf("usage: rtm_bandwidth_probe [--csv PATH] [--mb 256] [--repeats 5]\n");
            return a == "--help" || a == "-h" ? 0 : 1;
        }
    }
    const std::size_t bytes_per_buffer = megabytes * 1024 * 1024;
    const std::size_t elements = bytes_per_buffer / sizeof(float);

    int cpu_threads = 1;
    const double cpu_gbps = probe_cpu_triad_bandwidth(elements, repeats, cpu_threads);
    const std::string cpu_name = rtm::cpu_model_name();
    std::printf("CPU  %-40s  threads=%-3d  triad  %.1f GB/s\n",
                cpu_name.c_str(), cpu_threads, cpu_gbps);
    if (!csv_path.empty()) append_csv_row(csv_path, "cpu", cpu_name, cpu_threads, cpu_gbps);

#ifdef RTM_WITH_CUDA
    std::string gpu_name, error;
    double gpu_gbps = 0.0;
    if (rtm::probe_gpu_copy_bandwidth(bytes_per_buffer, repeats, gpu_name, gpu_gbps, error)) {
        std::printf("GPU  %-40s  copy   %.1f GB/s\n", gpu_name.c_str(), gpu_gbps);
        if (!csv_path.empty()) append_csv_row(csv_path, "gpu", gpu_name, 0, gpu_gbps);
    } else {
        std::printf("GPU  not probed: %s\n", error.c_str());
    }
#endif
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
