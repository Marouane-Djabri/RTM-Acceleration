#include "rtm_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace rtm {

void CPUReferenceRTM::setup(const VelocityModel& model, const RTMParams& par,
                            const TimeAxis& ta) {
    par_ = par;
    ta_  = ta;
    g_       = model.grid;
    g_.nb    = par.nb;
    g_.half  = stencil_half(par.order);

    if (g_.nb < g_.half + 1)
        throw std::runtime_error("--nb must be at least order/2 + 1");
    if (model.v.size() != g_.n_interior())
        throw std::runtime_error("velocity model size does not match nx*nz");
    if (par.store_interval < 1)
        throw std::runtime_error("--store-interval must be >= 1");

    nsnap_ = (ta.nt - 1) / par.store_interval + 1;

    // this is important for me
    const int nxe = g_.nxe(), nze = g_.nze(), nb = g_.nb;

    // -------------------------------------------------------------------------
    // 1. Extend the velocity model onto the padded grid by edge replication and
    //    pre-multiply into (v*dt)^2 — the only form the stencil ever needs.
    // -------------------------------------------------------------------------
    vdt2_.assign(g_.n_extended(), 0.0f);
    const float dt2 = ta.dt * ta.dt;
    for (int ixe = 0; ixe < nxe; ++ixe) {
        const int ix = std::min(std::max(ixe - nb, 0), g_.nx - 1);
        for (int ize = 0; ize < nze; ++ize) {
            const int iz = std::min(std::max(ize - nb, 0), g_.nz - 1);
            const float v = model.v[g_.iidx(ix, iz)];
            vdt2_[g_.eidx(ixe, ize)] = v * v * dt2;
        }
    }

    // -------------------------------------------------------------------------
    // 2. Cerjan (1985) sponge: w(i) = exp(-(alpha*(nb-i))^2) inside the pad,
    //    1.0 in the interior. Separable, so the 2D map is the product of the
    //    two 1D tapers. Applied to BOTH p^n and p^{n+1} every step.
    //    Absorbing on ALL FOUR sides (no free surface — see README assumptions).
    // -------------------------------------------------------------------------
    auto taper = [&](int i, int n_extended) {
        const int d = std::min(i, n_extended - 1 - i);   // distance to the edge
        if (d >= nb) return 1.0f;
        const float a = par.sponge_alpha * (float)(nb - d);
        return std::exp(-a * a);
    };
    std::vector<float> wx(nxe), wz(nze);
    for (int i = 0; i < nxe; ++i) wx[i] = taper(i, nxe);
    for (int i = 0; i < nze; ++i) wz[i] = taper(i, nze);
    sponge_.assign(g_.n_extended(), 1.0f);
    for (int ixe = 0; ixe < nxe; ++ixe)
        for (int ize = 0; ize < nze; ++ize)
            sponge_[g_.eidx(ixe, ize)] = wx[ixe] * wz[ize];

    // -------------------------------------------------------------------------
    // 3. Ricker source wavelet, delayed by t0 = 1.2/f0 so that f(0) ~ 0.
    // -------------------------------------------------------------------------
    wavelet_.resize(ta.nt);
    const float t0 = 1.2f / par.f0;
    for (int it = 0; it < ta.nt; ++it)
        wavelet_[it] = ricker((float)it * ta.dt - t0, par.f0);

    // -------------------------------------------------------------------------
    // 4. Stencil coefficients, pre-divided by the grid spacing.
    // -------------------------------------------------------------------------
    const float* c = stencil_coefficients(par.order);
    const float idx2 = 1.0f / (g_.dx * g_.dx);
    const float idz2 = 1.0f / (g_.dz * g_.dz);
    c0_ = c[0] * (idx2 + idz2);
    for (int k = 1; k <= g_.half; ++k) { cx_[k] = c[k] * idx2; cz_[k] = c[k] * idz2; }

    // -------------------------------------------------------------------------
    // 5. Wavefield buffers. The backward pass reuses the same three buffers,
    //    so peak wavefield memory is 3 * nxe * nze * 4 bytes.
    // -------------------------------------------------------------------------
    pp_.assign(g_.n_extended(), 0.0f);
    pc_.assign(g_.n_extended(), 0.0f);
    pn_.assign(g_.n_extended(), 0.0f);
}

// Map physical (x,z) in metres to a linear index on the EXTENDED grid.
// Nearest-grid-point injection (no sinc interpolation — see README assumptions).
//
// Coordinates outside the model are CLAMPED to the edge, which is the right
// behaviour for a receiver a fraction of a cell off the end of the line. But a
// unit or scalar mistake — SEG-Y `scalco`, or the AGL Marmousi convention of
// storing X multiplied by 1000 — clamps *every* trace onto the same edge cell
// and still produces a plausible-looking image. Silence there is a trap, so
// count the clamps and say so once per shot.
void CPUReferenceRTM::map_geometry(const ShotRecord& shot) {
    int nclamped = 0;
    auto to_index = [&](float x, float z) -> std::size_t {
        const int ix0 = (int)std::lround((x - g_.ox) / g_.dx);
        const int iz0 = (int)std::lround((z - g_.oz) / g_.dz);
        const int ix  = std::min(std::max(ix0, 0), g_.nx - 1);
        const int iz  = std::min(std::max(iz0, 0), g_.nz - 1);
        if (ix != ix0 || iz != iz0) ++nclamped;
        return g_.eidx(ix + g_.nb, iz + g_.nb);
    };
    src_index_ = to_index(shot.sx, shot.sz);
    rec_index_.resize((std::size_t)shot.nrec());
    for (int ir = 0; ir < shot.nrec(); ++ir)
        rec_index_[ir] = to_index(shot.rx[ir], shot.rz[ir]);

    if (nclamped > 0 && par_.verbose) {
        std::fprintf(stderr,
            "WARNING: %d of %d source/receiver positions fell outside the "
            "%.1f x %.1f m model and were clamped to the edge.\n"
            "         Check the coordinate units (SEG-Y scalco, or coordinates "
            "stored pre-multiplied) before trusting this image.\n",
            nclamped, shot.nrec() + 1,
            (g_.nx - 1) * g_.dx, (g_.nz - 1) * g_.dz);
    }
}

} // namespace rtm