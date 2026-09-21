#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "comparison.hpp"
#include "rtm_factory.hpp"
#include "rtm_io.hpp"


using namespace rtm;

static void usage() {
    std::printf(
"rtm — 2D acoustic Reverse Time Migration\n\n"
"Engine:\n"
"  --engine NAME          which implementation to run   (default cpu)\n"
"  --list-engines         print the engines in this build and exit\n"
"  --gpus N               GPUs for the multi-GPU engine   (default 1)\n\n"
"Required:\n"
"  --velocity PATH        float32 velocity model (m/s)\n"
"  --shots PATH           shot gathers (.bin = RTMS raw, .sgy/.segy = SEG-Y)\n"
"  --output PATH          migrated image, float32, image[ix*nz+iz]\n\n"
"Velocity geometry (read from PATH.hdr when present; CLI overrides):\n"
"  --nx N  --nz N  --dx M  --dz M  --ox M  --oz M\n"
"  --vel-layout zfast|xfast\n\n"
"Modelling / migration parameters:\n"
"  --order N              spatial FD order 2|4|6|8|10|12   (default 8)\n"
"  --nb N                 sponge width in cells            (default 60)\n"
"  --sponge-alpha F       Cerjan coefficient               (default 0.0053)\n"
"  --f0 F                 Ricker peak frequency, Hz        (default 12)\n"
"  --store-interval N     save source wavefield every N steps (default 1)\n"
"  --nt N                 truncate the time axis           (default: from data)\n"
"  --max-shots N          migrate only the first N shots\n"
"  --mute-direct V        direct-wave mute velocity, m/s (0 = off, default 0)\n\n"
"Outputs:\n"
"  --illumination         also write <output>_illum.bin (illumination-compensated)\n"
"  --filter laplacian     also write <output>_filtered.bin\n"
"  --write-illum-map      also write <output>_illummap.bin\n"
"  --benchmark PATH       append the benchmark report to PATH\n"
"  --benchmark-csv PATH   append one CSV row to PATH (input of the plots)\n"
"  --dataset NAME         dataset label in the CSV (default: shots file stem)\n\n"
"SEG-Y only:\n"
"  --segy-src-depth M     source depth   (default 10)\n"
"  --segy-rec-depth M     receiver depth (default 10)\n\n"
"  --quiet                less chatter\n");
}

// "data/real/marmousi_shots.bin" -> "marmousi_shots"
static std::string shots_file_stem(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

int main(int argc, char** argv) try {
    std::string vel_path, shots_path, out_path, bench_path, hdr_path, filter = "none";
    std::string engine_name = "cpu", bench_csv_path, dataset_label;
    int  requested_gpus = 1;
    io::VelocityHeader vh;
    RTMParams par;
    int  nt_override = 0, max_shots = 0;
    bool have_nx = false, have_nz = false, have_dx = false, have_dz = false;
    bool have_ox = false, have_oz = false, have_layout = false;
    bool want_illum = false, want_illum_map = false;
    float mute_v = 0.0f, segy_sz = 10.0f, segy_rz = 10.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if      (a == "--velocity")       vel_path   = next();
        else if (a == "--shots")          shots_path = next();
        else if (a == "--output")         out_path   = next();
        else if (a == "--vel-header")     hdr_path = next();
        else if (a == "--nx")           { vh.nx = std::stoi(next()); have_nx = true; }
        else if (a == "--nz")           { vh.nz = std::stoi(next()); have_nz = true; }
        else if (a == "--dx")           { vh.dx = std::stof(next()); have_dx = true; }
        else if (a == "--dz")           { vh.dz = std::stof(next()); have_dz = true; }
        else if (a == "--ox")           { vh.ox = std::stof(next()); have_ox = true; }
        else if (a == "--oz")           { vh.oz = std::stof(next()); have_oz = true; }
        else if (a == "--vel-layout")   { vh.layout = next(); have_layout = true; }
        else if (a == "--order")          par.order = std::stoi(next());
        else if (a == "--nb")             par.nb = std::stoi(next());
        else if (a == "--sponge-alpha")   par.sponge_alpha = std::stof(next());
        else if (a == "--f0")             par.f0 = std::stof(next());
        else if (a == "--store-interval") par.store_interval = std::stoi(next());
        else if (a == "--nt")             nt_override = std::stoi(next());
        else if (a == "--max-shots")      max_shots = std::stoi(next());
        else if (a == "--mute-direct")    mute_v = std::stof(next());
        else if (a == "--illumination")   want_illum = true;
        else if (a == "--write-illum-map")want_illum_map = true;
        else if (a == "--filter")         filter = next();
        else if (a == "--benchmark")      bench_path = next();
        else if (a == "--segy-src-depth") segy_sz = std::stof(next());
        else if (a == "--segy-rec-depth") segy_rz = std::stof(next());
        else if (a == "--quiet")          par.verbose = false;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--engine")         engine_name = next();
        else if (a == "--list-engines") {
            for (const std::string& name : list_engines()) std::printf("%s\n", name.c_str());
            return 0;
        }
        else if (a == "--gpus")           requested_gpus = std::stoi(next());
        else if (a == "--benchmark-csv")  bench_csv_path = next();
        else if (a == "--dataset")        dataset_label = next();
        else throw std::runtime_error("unknown option: " + a);
    }
    if (vel_path.empty() || shots_path.empty() || out_path.empty()) {
        usage();
        throw std::runtime_error("--velocity, --shots and --output are required");
    }

    // ---------------- STAGE 1: DATA LOADING -------------------------------
    std::unique_ptr<RTMEngine> engine = make_engine(engine_name, requested_gpus);
    Timer io_timer;

    // Auto-detected sidecar. Precedence (documented in the README): an explicit
    // CLI flag wins, then --vel-header, then <velocity>.hdr. Testing a field
    // against its default value is NOT a substitute for tracking whether the
    // user actually set it: "--ox 0" and "--vel-layout zfast" are legitimate
    // explicit choices and must not be silently replaced by the sidecar.
    {
        io::VelocityHeader sidecar;
        bool found = false;
        if (!hdr_path.empty()) {
            if (!io::read_header_file(hdr_path, sidecar))
                throw std::runtime_error("cannot read velocity header: " + hdr_path);
            found = true;
        } else {
            found = io::read_header_file(vel_path + ".hdr", sidecar);
        }
        if (found) {
            if (!have_nx)     vh.nx     = sidecar.nx;
            if (!have_nz)     vh.nz     = sidecar.nz;
            if (!have_dx)     vh.dx     = sidecar.dx;
            if (!have_dz)     vh.dz     = sidecar.dz;
            if (!have_ox)     vh.ox     = sidecar.ox;
            if (!have_oz)     vh.oz     = sidecar.oz;
            if (!have_layout) vh.layout = sidecar.layout;
        }
    }
    VelocityModel model = io::read_velocity_raw(vel_path, vh);
    std::vector<ShotRecord> shots = io::read_shots(shots_path, segy_sz, segy_rz);
    if (max_shots > 0 && (int)shots.size() > max_shots)
        shots.resize((std::size_t)max_shots);

    if (mute_v > 0.0f)
        for (auto& s : shots) mute_direct_wave(s, mute_v, 1.0f / par.f0, 1.0f / par.f0);

    engine->times.io = io_timer.elapsed();

    TimeAxis ta;
    ta.nt = (nt_override > 0) ? std::min(nt_override, shots[0].nt) : shots[0].nt;
    ta.dt = shots[0].dt;

    // ---------------- STAGE 2: SETUP + SANITY CHECKS ----------------------
    engine->setup(model, par, ta);
    const Grid& g = engine->grid();

    const float dt_max = max_stable_dt(model.vmax(), g, par.order);
    const float ppw    = points_per_wavelength(model.vmin(), par.f0, g);

    std::printf("\n=== MODEL ===\n"
                "  nx = %d, nz = %d, dx = %.2f m, dz = %.2f m, origin = (%.1f, %.1f)\n"
                "  vmin = %.1f m/s, vmax = %.1f m/s\n"
                "  nt = %d, dt = %.6f s, T = %.3f s\n"
                "  order = %d, nb = %d, f0 = %.2f Hz, store_interval = %d\n"
                "  shots = %zu, receivers/shot = %d\n"
                "  CFL: dt = %.6f s, dt_max = %.6f s (ratio %.3f) %s\n"
                "  Dispersion: %.2f points per shortest wavelength %s\n"
                "  Snapshot memory: %.1f MiB\n\n",
                g.nx, g.nz, g.dx, g.dz, g.ox, g.oz,
                model.vmin(), model.vmax(), ta.nt, ta.dt, ta.nt * ta.dt,
                par.order, par.nb, par.f0, par.store_interval,
                shots.size(), shots[0].nrec(),
                ta.dt, dt_max, ta.dt / dt_max,
                (ta.dt <= dt_max ? "OK" : "*** UNSTABLE ***"),
                ppw, (ppw >= 4.0f ? "OK" : "*** DISPERSIVE ***"),
                engine->snapshot_bytes() / (1024.0 * 1024.0));

    if (ta.dt > dt_max)
        throw std::runtime_error("time step violates the CFL condition — "
                                 "reduce dt, reduce the FD order, or coarsen f0/dx");
    if (ppw < 3.0f)
        std::fprintf(stderr, "WARNING: fewer than 3 points per wavelength; the "
                             "image will be visibly dispersive.\n");

    // ---------------- STAGE 3: MIGRATION ----------------------------------
    std::vector<float> image, illum;
    std::printf("=== MIGRATION (%s) ===\n", engine->name());
    engine->migrate(shots, image, illum);

    // ---------------- STAGE 4: OUTPUT -------------------------------------
    io::write_raw_floats(out_path, image);   // RAW zero-lag XCorr = the reference
    std::printf("\nwrote %s  (%d x %d float32, image[ix*nz+iz])\n",
                out_path.c_str(), g.nx, g.nz);

    const std::size_t dot = out_path.find_last_of('.');
    const std::string stem = (dot == std::string::npos) ? out_path
                                                        : out_path.substr(0, dot);
    if (want_illum_map) io::write_raw_floats(stem + "_illummap.bin", illum);
    if (want_illum) {
        std::vector<float> tmp = image;
        illumination_compensation(tmp, illum, 1e-4f);
        io::write_raw_floats(stem + "_illum.bin", tmp);
        std::printf("wrote %s_illum.bin\n", stem.c_str());
    }
    if (filter == "laplacian") {
        std::vector<float> tmp = image;
        if (want_illum) illumination_compensation(tmp, illum, 1e-4f);
        laplacian_filter(tmp, g);
        io::write_raw_floats(stem + "_filtered.bin", tmp);
        std::printf("wrote %s_filtered.bin\n", stem.c_str());
    }

    // ---------------- STAGE 5: BENCHMARK ----------------------------------
    BenchmarkContext ctx;
    ctx.nx = g.nx; ctx.nz = g.nz; ctx.nb = g.nb;
    ctx.dx = g.dx; ctx.dz = g.dz;
    ctx.nt = ta.nt; ctx.dt = ta.dt;
    ctx.nshots = (int)shots.size();
    ctx.order = par.order;
    ctx.store_interval = par.store_interval;
    ctx.snapshot_bytes = engine->snapshot_bytes();
    ctx.results.push_back({engine->name(), engine->times, engine->is_gpu()});
    print_benchmark_report(std::cout, ctx);
    if (!bench_path.empty()) {
        std::ofstream bf(bench_path, std::ios::app);
        print_benchmark_report(bf, ctx);
    }
    if (!bench_csv_path.empty()) {
        RunInfo info;
        info.host     = host_name();
        info.cpu_name = cpu_model_name();
        info.gpu_name = engine->device_name();
        info.engine   = engine_name;
        info.dataset  = dataset_label.empty() ? shots_file_stem(shots_path) : dataset_label;
        info.threads  = engine->num_threads();
        info.gpus     = engine->num_devices();
        append_benchmark_csv(bench_csv_path, ctx, ctx.results.back(), info);
        std::printf("appended benchmark row to %s\n", bench_csv_path.c_str());
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "\nERROR: %s\n", e.what());
    return 1;
}