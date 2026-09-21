#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace rtm {

constexpr int MAX_HALF = 6;   // supports spatial orders 2..12

// -----------------------------------------------------------------------------
// Grid layout convention (IMPORTANT — the same for every array in the project):
//   index = ix * nz + iz     ->  DEPTH IS THE FASTEST-VARYING AXIS.
// The "interior" grid is the physical model (nx * nz).
// The "extended" grid adds nb absorbing cells on all four sides (nxe * nze).
// -----------------------------------------------------------------------------
struct Grid {
    int   nx = 0, nz = 0;          // interior samples
    float dx = 10.0f, dz = 10.0f;  // metres
    float ox = 0.0f, oz = 0.0f;    // origin of interior grid, metres
    int   nb = 60;                 // absorbing pad width in cells
    int   half = 4;                // stencil half width (order / 2)

    int nxe() const { return nx + 2 * nb; }
    int nze() const { return nz + 2 * nb; }
    std::size_t n_interior() const { return (std::size_t)nx * nz; }
    std::size_t n_extended() const { return (std::size_t)nxe() * nze(); }
    std::size_t eidx(int ixe, int ize) const { return (std::size_t)ixe * nze() + ize; }
    std::size_t iidx(int ix,  int iz ) const { return (std::size_t)ix  * nz     + iz;  }
};

struct TimeAxis { int nt = 0; float dt = 0.0f; };

struct RTMParams {
    int   nb             = 60;      // sponge width
    int   order          = 8;       // spatial FD order
    float sponge_alpha   = 0.0053f; // Cerjan coefficient
    int   store_interval = 1;       // save source wavefield every N steps
    float f0             = 12.0f;   // Ricker peak frequency (Hz)
    bool  verbose        = true;
};

struct VelocityModel {
    Grid grid;
    std::vector<float> v;   // interior, m/s, v[ix*nz + iz]
    float vmin() const;
    float vmax() const;
};

struct ShotRecord {
    float sx = 0.0f, sz = 0.0f;      // source position, metres
    std::vector<float> rx, rz;       // receiver positions, metres
    int   nt = 0;
    float dt = 0.0f;
    std::vector<float> traces;       // nrec * nt, TRACE-MAJOR: traces[ir*nt + it]
    int nrec() const { return (int)rx.size(); }
    void allocate() { traces.assign((std::size_t)nrec() * nt, 0.0f); }
};

// ---- shared numerics (src/rtm/rtm.cpp) --------------------------------------
const float* stencil_coefficients(int order);  // returns half+1 coefficients
int   stencil_half(int order);                 // order / 2
float stencil_abs_sum(int order);              // |c0| + 2*sum|ck|, for CFL
float ricker(float t, float f0);               // t relative to the peak
float max_stable_dt(float vmax, const Grid& g, int order);
float points_per_wavelength(float vmin, float f0, const Grid& g);

} // namespace rtm  