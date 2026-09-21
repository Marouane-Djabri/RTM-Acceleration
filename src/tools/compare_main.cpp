#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "benchmark.hpp"
#include "comparison.hpp"
#include "rtm_io.hpp"
#include <fstream>

using namespace rtm;

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::printf("usage: rtm_compare REFERENCE.bin TEST.bin [--nx N --nz N]\n"
                    "                   [--csv PATH --engine NAME --dataset NAME]\n"
                    "  REFERENCE is always the CPU reference image.\n"
                    "  --csv appends one row (metrics + PASS/FAIL) for the plots.\n");
        return 1;
    }
    std::string ref_path = argv[1], test_path = argv[2];
    int nx = 0, nz = 0;
    std::string csv_path, engine_name = "unknown", dataset_label = "unknown";
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--nx"      && i + 1 < argc) nx = std::stoi(argv[++i]);
        else if (a == "--nz"      && i + 1 < argc) nz = std::stoi(argv[++i]);
        else if (a == "--csv"     && i + 1 < argc) csv_path      = argv[++i];
        else if (a == "--engine"  && i + 1 < argc) engine_name   = argv[++i];
        else if (a == "--dataset" && i + 1 < argc) dataset_label = argv[++i];
        else throw std::runtime_error("unknown option: " + a);
    }
    const auto ref  = io::read_raw_floats(ref_path);
    const auto test = io::read_raw_floats(test_path);
    const auto m    = compare_images(ref, test);
    print_metrics(std::cout, m, nx, nz);

    // -------------------------------------------------------------------------
    // NOTE ON THRESHOLDS (Phase 7).
    // No threshold is enforced here on purpose. Pick one and write it down:
    //
    //   * Same precision, same order of operations (e.g. loop interchange,
    //     blocking, no reassociation): expect bit-identical results.
    //   * FP32 with reassociated/vectorized/parallel accumulation, or FMA
    //     contraction: relative L2 typically 1e-7..1e-5 for this problem size.
    //     A useful gate is  l2_relative < 1e-5  AND  correlation > 0.9999.
    //   * FP32 vs FP64: differences are dominated by the FP32 run itself, so
    //     compare against the FP64 result, not the other way round.
    //   * Anything above ~1e-3 relative L2 is a BUG (halo, race, indexing),
    //     not rounding.
    // -------------------------------------------------------------------------
    const double gate_l2 = 1e-5, gate_corr = 0.9999;
    const bool pass = (m.l2_relative < gate_l2) && (m.correlation > gate_corr);
    std::printf("\nDefault gate (documented, not a scientific law): "
                "L2rel < %.0e and corr > %.4f  =>  %s\n",
                gate_l2, gate_corr, pass ? "PASS" : "FAIL");

    if (!csv_path.empty()) {
        bool need_header = true;
        {
            std::ifstream existing(csv_path);
            if (existing && existing.peek() != std::ifstream::traits_type::eof())
                need_header = false;
        }
        std::ofstream out(csv_path, std::ios::app);
        if (!out) throw std::runtime_error("cannot append compare csv: " + csv_path);
        if (need_header)
            out << "date,host,engine,dataset,max_abs,mean_abs,rms,l2_relative,"
                   "max_relative,correlation,pass\n";
        out.precision(9);
        out << date_now_iso() << ',' << host_name() << ',' << engine_name << ','
            << dataset_label << ',' << m.max_abs << ',' << m.mean_abs << ',' << m.rms << ','
            << m.l2_relative << ',' << m.max_relative << ',' << m.correlation << ','
            << (pass ? 1 : 0) << '\n';
        std::printf("appended compare row to %s\n", csv_path.c_str());
    }
    return pass ? 0 : 2;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}