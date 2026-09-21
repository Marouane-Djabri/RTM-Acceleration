// =============================================================================
// IMAGING CONDITION
//        v
// computational kernel  (imaging)
//        v
// CUDA CANDIDATE #3  <-- elementwise fused multiply-add over the image grid
// =============================================================================
#include "rtm_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace rtm {

// -----------------------------------------------------------------------------
// KERNEL 7 — zero-lag cross-correlation imaging condition:
//
//     I(x,z)  += p_src(x,z,t) * p_rec(x,z,t)
//     S(x,z)  += p_src(x,z,t)^2          (source illumination, for Phase-7
//                                         amplitude normalization)
//
// Perfectly parallel over (x,z). On GPU this fuses into the tail of the
// backward stencil kernel, which removes an entire wavefield-sized read.
// -----------------------------------------------------------------------------
void CPUReferenceRTM::imaging(const float* fwd_snapshot,
                              const float* bwd_extended,
                              std::vector<float>& image,
                              std::vector<float>& illumination) {
    const int nx = g_.nx, nz = g_.nz, nb = g_.nb, nze = g_.nze();

    for (int ix = 0; ix < nx; ++ix) {
        const float* bcol = bwd_extended + (std::size_t)(ix + nb) * nze + nb;
        for (int iz = 0; iz < nz; ++iz) {
            const std::size_t i = (std::size_t)ix * nz + iz;
            const float f = fwd_snapshot[i];
            image[i]        += f * bcol[iz];
            illumination[i] += f * f;
        }
    }
}

// -----------------------------------------------------------------------------
// POST-PROCESSING (applied once, after all shots).
// -----------------------------------------------------------------------------

// Source-illumination compensation: I / (S + eps*max(S)).
// Balances amplitudes; does not change reflector positions.
void illumination_compensation(std::vector<float>& image,
                               const std::vector<float>& illumination,
                               float eps_relative) {
    float smax = 0.0f;
    for (float s : illumination) smax = std::max(smax, s);
    const float eps = eps_relative * smax + 1e-30f;
    for (std::size_t i = 0; i < image.size(); ++i)
        image[i] /= (illumination[i] + eps);
}

// Laplacian image filter. The raw zero-lag correlation is contaminated by
// low-wavenumber "rabbit-ear" artifacts from the source/receiver wavefields
// travelling together. A Laplacian is the cheapest standard high-pass that
// suppresses them (Youn & Zhou 2001; Zhang & Sun 2009).
// Applied ONLY to the optional *_filtered.bin output — never to the reference.
void laplacian_filter(std::vector<float>& image, const Grid& g) {
    std::vector<float> out(image.size(), 0.0f);
    const float idx2 = 1.0f / (g.dx * g.dx);
    const float idz2 = 1.0f / (g.dz * g.dz);
    for (int ix = 1; ix < g.nx - 1; ++ix) {
        for (int iz = 1; iz < g.nz - 1; ++iz) {
            const std::size_t i = (std::size_t)ix * g.nz + iz;
            out[i] = idx2 * (image[i + g.nz] - 2.0f * image[i] + image[i - g.nz])
                   + idz2 * (image[i + 1   ] - 2.0f * image[i] + image[i - 1   ]);
        }
    }
    image.swap(out);
}

} // namespace rtm