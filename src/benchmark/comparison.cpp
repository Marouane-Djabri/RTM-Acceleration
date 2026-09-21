#include "comparison.hpp"
#include <cmath>
#include <stdexcept>

namespace rtm {

ErrorMetrics compare_images(const std::vector<float>& ref,
                            const std::vector<float>& test) {
    if (ref.size() != test.size())
        throw std::runtime_error("image size mismatch in comparison");
    ErrorMetrics m;
    m.n = ref.size();
    if (m.n == 0) return m;

    double sum_abs = 0, sum_sq = 0, ref_sq = 0, cross = 0, test_sq = 0;
    for (std::size_t i = 0; i < m.n; ++i) {
        const double a = ref[i], b = test[i], d = a - b;
        const double ad = std::fabs(d);
        if (ad > m.max_abs) { m.max_abs = ad; m.argmax = i; }
        sum_abs += ad;
        sum_sq  += d * d;
        ref_sq  += a * a;
        test_sq += b * b;
        cross   += a * b;
        m.ref_max_abs  = std::max(m.ref_max_abs,  std::fabs(a));
        m.test_max_abs = std::max(m.test_max_abs, std::fabs(b));
    }
    m.mean_abs     = sum_abs / (double)m.n;
    m.rms          = std::sqrt(sum_sq / (double)m.n);
    m.l2_relative  = (ref_sq > 0) ? std::sqrt(sum_sq / ref_sq) : 0.0;
    m.max_relative = (m.ref_max_abs > 0) ? m.max_abs / m.ref_max_abs : 0.0;
    const double den = std::sqrt(ref_sq * test_sq);
    m.correlation  = (den > 0) ? cross / den : 0.0;
    return m;
}

void print_metrics(std::ostream& os, const ErrorMetrics& m, int nx, int nz) {
    os << "\n---------- NUMERICAL COMPARISON ----------\n"
       << "  samples                 : " << m.n << "\n"
       << "  max |ref|               : " << m.ref_max_abs  << "\n"
       << "  max |test|              : " << m.test_max_abs << "\n"
       << "  max absolute error      : " << m.max_abs      << "\n"
       << "  mean absolute error     : " << m.mean_abs     << "\n"
       << "  RMS error               : " << m.rms          << "\n"
       << "  relative L2 error       : " << m.l2_relative  << "\n"
       << "  max error / max |ref|   : " << m.max_relative << "\n"
       << "  normalized correlation  : " << m.correlation  << "\n";
    if (nx > 0 && nz > 0)
        os << "  worst sample at (ix,iz) : (" << (int)(m.argmax / (std::size_t)nz)
           << ", " << (int)(m.argmax % (std::size_t)nz) << ")\n";
    os << "------------------------------------------\n";
}

} // namespace rtm