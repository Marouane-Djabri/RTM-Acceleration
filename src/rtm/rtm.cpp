#include "rtm_engine.hpp"
#include "benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rtm {

// =============================================================================
// Central finite-difference coefficients for the SECOND derivative.
// Standard Taylor-optimal coefficients; each set sums to zero.
//   d2f/dx2 ~ (1/h^2) * ( c[0]*f_i + sum_{k=1..half} c[k]*(f_{i+k} + f_{i-k}) )
// =============================================================================
static const float C2 [] = {-2.0f, 1.0f};
static const float C4 [] = {-2.5f, 4.0f/3.0f, -1.0f/12.0f};
static const float C6 [] = {-49.0f/18.0f, 1.5f, -0.15f, 1.0f/90.0f};
static const float C8 [] = {-205.0f/72.0f, 8.0f/5.0f, -1.0f/5.0f,
                            8.0f/315.0f, -1.0f/560.0f};
static const float C10[] = {-5269.0f/1800.0f, 5.0f/3.0f, -5.0f/21.0f,
                            5.0f/126.0f, -5.0f/1008.0f, 1.0f/3150.0f};
static const float C12[] = {-5369.0f/1800.0f, 12.0f/7.0f, -15.0f/56.0f,
                            10.0f/189.0f, -1.0f/112.0f, 2.0f/1925.0f,
                            -1.0f/16632.0f};

const float* stencil_coefficients(int order) {
    switch (order) {
        case 2:  return C2;
        case 4:  return C4;
        case 6:  return C6;
        case 8:  return C8;
        case 10: return C10;
        case 12: return C12;
        default: throw std::runtime_error(
            "unsupported spatial order (use 2, 4, 6, 8, 10 or 12)");
    }
}

int stencil_half(int order) {
    stencil_coefficients(order);   // validates
    return order / 2;
}

float stencil_abs_sum(int order) {
    const float* c = stencil_coefficients(order);
    const int h = order / 2;
    float s = std::fabs(c[0]);
    for (int k = 1; k <= h; ++k) s += 2.0f * std::fabs(c[k]);
    return s;
}

// Ricker wavelet, t measured relative to the peak.
float ricker(float t, float f0) {
    const float a = (float)(M_PI * M_PI) * f0 * f0 * t * t;
    return (1.0f - 2.0f * a) * std::exp(-a);
}

// von Neumann stability limit for 2nd-order-in-time / 2M-order-in-space:
//   dt <= 2 / ( vmax * sqrt( S*(1/dx^2 + 1/dz^2) ) ),  S = |c0| + 2*sum|ck|
float max_stable_dt(float vmax, const Grid& g, int order) {
    const float S = stencil_abs_sum(order);
    return 2.0f / (vmax * std::sqrt(S * (1.0f/(g.dx*g.dx) + 1.0f/(g.dz*g.dz))));
}

// Grid points per shortest wavelength. Ricker energy reaches ~2.5*f0.
// 8th order needs >= ~4-5 points/wavelength to keep numerical dispersion small.
float points_per_wavelength(float vmin, float f0, const Grid& g) {
    const float fmax = 2.5f * f0;
    const float lambda_min = vmin / fmax;
    return lambda_min / std::max(g.dx, g.dz);
}

float VelocityModel::vmin() const {
    return v.empty() ? 0.0f : *std::min_element(v.begin(), v.end());
}
float VelocityModel::vmax() const {
    return v.empty() ? 0.0f : *std::max_element(v.begin(), v.end());
}

// =============================================================================
// Default multi-shot migration loop. Shot-parallel by construction: the shot
// loop is embarrassingly parallel (MPI / multi-GPU slot, Phase 4).
// =============================================================================
void RTMEngine::migrate(const std::vector<ShotRecord>& shots,
                        std::vector<float>& image,
                        std::vector<float>& illumination) {
    Timer total;
    const std::size_t nxz = g_.n_interior();
    image.assign(nxz, 0.0f);
    illumination.assign(nxz, 0.0f);
    snap_.assign((std::size_t)nsnap_ * nxz, 0.0f);

    for (std::size_t is = 0; is < shots.size(); ++is) {
        if (par_.verbose) {
            std::printf("  shot %2zu / %2zu   sx=%8.1f m  sz=%6.1f m  nrec=%d\n",
                        is + 1, shots.size(), shots[is].sx, shots[is].sz,
                        shots[is].nrec());
            std::fflush(stdout);
        }
        forward_propagation(shots[is], &snap_, nullptr);
        backward_propagation(shots[is], snap_, image, illumination);
    }
    times.total += total.elapsed();
}

void RTMEngine::model_shot(ShotRecord& shot) {
    shot.nt = ta_.nt;
    shot.dt = ta_.dt;
    std::vector<float> rec((std::size_t)shot.nrec() * ta_.nt, 0.0f);
    forward_propagation(shot, nullptr, &rec);
    shot.traces = std::move(rec);
}

} // namespace rtm
