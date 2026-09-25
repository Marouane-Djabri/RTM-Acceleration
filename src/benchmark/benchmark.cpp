#include "benchmark.hpp"
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace rtm {

static std::string fmt_time(double s) {
    char b[64];
    if      (s < 1e-3) std::snprintf(b, sizeof b, "%10.3f us", s * 1e6);
    else if (s < 1.0)  std::snprintf(b, sizeof b, "%10.3f ms", s * 1e3);
    else               std::snprintf(b, sizeof b, "%10.3f s ", s);
    return std::string(b);
}

void print_benchmark_report(std::ostream& os, const BenchmarkContext& c) {
    const double flops_per_point = 6.0 * (c.order / 2) + 5.0;   // stencil only
    const double pts   = (double)(c.nx + 2 * c.nb) * (double)(c.nz + 2 * c.nb);
    // 2 propagations (forward + backward) per shot
    const double gflop = pts * c.nt * 2.0 * c.nshots * flops_per_point * 1e-9;

    os << "\n========================================\n"
       << "RTM BENCHMARK\n"
       << "========================================\n\n"
       << "Grid:\n"
       << "    NX             = " << c.nx << "  (dx = " << c.dx << " m)\n"
       << "    NZ             = " << c.nz << "  (dz = " << c.dz << " m)\n"
       << "    Sponge NB      = " << c.nb << "\n"
       << "    Extended       = " << (c.nx + 2 * c.nb) << " x " << (c.nz + 2 * c.nb) << "\n"
       << "    Spatial order  = " << c.order << "\n\n"
       << "Time steps:\n"
       << "    NT             = " << c.nt << "  (dt = " << c.dt * 1e3 << " ms, T = "
       << c.nt * c.dt << " s)\n\n"
       << "Shots:\n"
       << "    NSHOTS         = " << c.nshots << "\n\n"
       << "Wavefield storage:\n"
       << "    Snapshots      = " << (c.snapshot_bytes / (1024.0 * 1024.0)) << " MiB\n\n"
       << "Propagation work:\n"
       << "    Stencil GFLOP  = " << gflop << "\n\n";

    const double ref_total = c.results.empty() ? 0.0 : c.results[0].t.total;

    for (const auto& r : c.results) {
        os << r.name << ":\n"
           << "    Data loading:  " << fmt_time(r.t.io)       << "\n";
        if (r.gpu) {
            os << "    H2D:           " << fmt_time(r.t.h2d)    << "\n"
               << "    Kernel:        " << fmt_time(r.t.kernel) << "\n"
               << "    D2H:           " << fmt_time(r.t.d2h)    << "\n";
        }
        os << "    Forward:       " << fmt_time(r.t.forward)  << "\n"
           << "    Backward:      " << fmt_time(r.t.backward) << "   (imaging excluded)\n"
           << "    Imaging:       " << fmt_time(r.t.imaging)  << "\n"
           << "    Total:         " << fmt_time(r.t.total)    << "\n";
        const double prop = r.t.forward + r.t.backward;
        if (prop > 0.0) {
            os << "    Stencil rate:  " << (gflop / prop) << " GFLOP/s\n";
            os << "    Throughput:    " << (pts * c.nt * 2.0 * c.nshots / prop * 1e-9)
               << " GPts/s (extended grid points x steps / propagation time)\n";
        }
        if (r.peak_device_bytes > 0)
            os << "    Device memory: " << (r.peak_device_bytes / (1024.0 * 1024.0))
               << " MiB peak in use (cudaMemGetInfo, includes the CUDA context)\n";
        if (ref_total > 0.0 && r.t.total > 0.0 && &r != &c.results[0])
            os << "    Speedup:       " << (ref_total / r.t.total) << " x\n";
        os << "\n";
    }
    os << "========================================\n";
}

// =============================================================================
// CSV output
// =============================================================================
std::string host_name() {
    char buffer[256] = {0};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0) return "unknown-host";
    return std::string(buffer);
}

std::string cpu_model_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) break;
            std::string name = line.substr(colon + 1);
            while (!name.empty() && name.front() == ' ') name.erase(name.begin());
            return name;
        }
    }
    return "unknown-cpu";
}

std::string date_now_iso() {
    const std::time_t now = std::time(nullptr);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    return std::string(buffer);
}

// Commas inside a name would break the CSV columns.
static std::string csv_safe(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    return text;
}

static const char* BENCHMARK_CSV_HEADER =
    "date,host,cpu_name,threads,gpu_name,gpus,engine,dataset,"
    "nx,nz,nb,order,nt,dt,nshots,store_interval,"
    "t_io,t_h2d,t_forward,t_backward,t_imaging,t_d2h,t_total,"
    "stencil_gflop,stencil_gflops,snapshot_mib,gpts_per_s,device_mem_mib\n";

// A CSV written by an older build has fewer columns. Rewrite it in place with
// the current header and empty values for the new columns, so old and new
// rows stay readable by the plotting scripts. Refuses anything that is not
// an older version of this same header.
static void upgrade_benchmark_csv_header(const std::string& path) {
    std::ifstream in(path);
    if (!in) return;
    std::string old_header;
    if (!std::getline(in, old_header)) return;
    std::string new_header = BENCHMARK_CSV_HEADER;
    new_header.pop_back();   // the '\n'
    if (old_header == new_header) return;
    if (new_header.compare(0, old_header.size(), old_header) != 0)
        throw std::runtime_error("unexpected header in " + path +
                                 " (not an older benchmark CSV); move it aside");

    const auto count_columns = [](const std::string& line) {
        return (int)std::count(line.begin(), line.end(), ',') + 1;
    };
    const int missing_columns = count_columns(new_header) - count_columns(old_header);
    std::vector<std::string> rows;
    std::string line;
    while (std::getline(in, line))
        if (!line.empty()) rows.push_back(line + std::string((std::size_t)missing_columns, ','));
    in.close();

    std::ofstream out(path, std::ios::trunc);
    out << new_header << '\n';
    for (const std::string& row : rows) out << row << '\n';
}

void append_benchmark_csv(const std::string& path, const BenchmarkContext& c,
                          const EngineResult& r, const RunInfo& info) {
    // Write the header only when the file is new or empty.
    bool need_header = true;
    {
        std::ifstream existing(path);
        if (existing && existing.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }
    if (!need_header) upgrade_benchmark_csv_header(path);
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot append benchmark csv: " + path);
    if (need_header) out << BENCHMARK_CSV_HEADER;

    const double flops_per_point = 6.0 * (c.order / 2) + 5.0;
    const double points = (double)(c.nx + 2 * c.nb) * (double)(c.nz + 2 * c.nb);
    const double gflop  = points * c.nt * 2.0 * c.nshots * flops_per_point * 1e-9;
    const double propagation_seconds = r.t.forward + r.t.backward;
    const double gflops = propagation_seconds > 0.0 ? gflop / propagation_seconds : 0.0;
    const double gpts_per_s = propagation_seconds > 0.0
        ? points * c.nt * 2.0 * c.nshots / propagation_seconds * 1e-9 : 0.0;
    const double device_mem_mib = r.peak_device_bytes / (1024.0 * 1024.0);

    out << std::setprecision(9)
        << date_now_iso() << ',' << csv_safe(info.host) << ',' << csv_safe(info.cpu_name) << ','
        << info.threads << ',' << csv_safe(info.gpu_name) << ',' << info.gpus << ','
        << csv_safe(info.engine) << ',' << csv_safe(info.dataset) << ','
        << c.nx << ',' << c.nz << ',' << c.nb << ',' << c.order << ',' << c.nt << ','
        << c.dt << ',' << c.nshots << ',' << c.store_interval << ','
        << r.t.io << ',' << r.t.h2d << ',' << r.t.forward << ',' << r.t.backward << ','
        << r.t.imaging << ',' << r.t.d2h << ',' << r.t.total << ','
        << gflop << ',' << gflops << ',' << (c.snapshot_bytes / (1024.0 * 1024.0)) << ','
        << gpts_per_s << ',' << device_mem_mib << '\n';
}

} // namespace rtm