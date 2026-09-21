#include "rtm_io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rtm {

// -----------------------------------------------------------------------------
// Direct-arrival mute. The direct wave carries no reflectivity information and
// produces a very strong low-wavenumber artifact in the cross-correlation image.
// Everything before  t = offset/v_surface + t_pad  is zeroed, with a raised
// cosine taper of length t_taper to avoid creating a step discontinuity.
// -----------------------------------------------------------------------------
void mute_direct_wave(ShotRecord& shot, float v_surface, float t_pad,
                      float t_taper) {
    if (v_surface <= 0.0f) return;
    for (int ir = 0; ir < shot.nrec(); ++ir) {
        const float dx  = shot.rx[ir] - shot.sx;
        const float dz  = shot.rz[ir] - shot.sz;
        const float off = std::sqrt(dx * dx + dz * dz);
        const float t0  = off / v_surface + t_pad;
        const int   i0  = (int)(t0 / shot.dt);
        const int   i1  = i0 + std::max(1, (int)(t_taper / shot.dt));
        float* tr = shot.traces.data() + (std::size_t)ir * shot.nt;
        for (int it = 0; it < shot.nt && it < i1; ++it) {
            if (it <= i0) { tr[it] = 0.0f; continue; }
            const float u = (float)(it - i0) / (float)(i1 - i0);
            tr[it] *= 0.5f * (1.0f - std::cos((float)M_PI * u));
        }
    }
}

namespace io {

// =============================================================================
// RAW SHOT FORMAT  ("RTMS", version 1, LITTLE-endian)
//
//  offset  type        field
//  ------  ----------  ---------------------------------------------------
//    0     char[4]     "RTMS"
//    4     int32       version = 1
//    8     int32       nshots
//   12     int32       nrec        (receivers per shot, constant)
//   16     int32       nt          (samples per trace)
//   20     float32     dt          (seconds)
//   24     int32       flags       (0)
//   28     int32       reserved    (0)
//  ------  ----------  --- then, for each shot, contiguously: -------------
//          float32     sx, sz                     (metres)
//          float32     rx[nrec], rz[nrec]         (metres)
//          float32     data[nrec*nt]              TRACE-MAJOR: d[ir*nt + it]
// =============================================================================
static const char RTMS_MAGIC[4] = {'R', 'T', 'M', 'S'};

void write_shots_raw(const std::string& path, const std::vector<ShotRecord>& shots) {
    if (shots.empty()) throw std::runtime_error("no shots to write");
    const int32_t nshots = (int32_t)shots.size();
    const int32_t nrec   = shots[0].nrec();
    const int32_t nt     = shots[0].nt;
    const float   dt     = shots[0].dt;
    for (const auto& s : shots)
        if (s.nrec() != nrec || s.nt != nt)
            throw std::runtime_error("RAW format requires constant nrec and nt");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write shots: " + path);
    const int32_t version = 1, flags = 0, reserved = 0;
    f.write(RTMS_MAGIC, 4);
    f.write((const char*)&version,  4);
    f.write((const char*)&nshots,   4);
    f.write((const char*)&nrec,     4);
    f.write((const char*)&nt,       4);
    f.write((const char*)&dt,       4);
    f.write((const char*)&flags,    4);
    f.write((const char*)&reserved, 4);
    for (const auto& s : shots) {
        f.write((const char*)&s.sx, 4);
        f.write((const char*)&s.sz, 4);
        f.write((const char*)s.rx.data(), (std::streamsize)(nrec * sizeof(float)));
        f.write((const char*)s.rz.data(), (std::streamsize)(nrec * sizeof(float)));
        f.write((const char*)s.traces.data(),
                (std::streamsize)(s.traces.size() * sizeof(float)));
    }
}

std::vector<ShotRecord> read_shots_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open shots: " + path);
    char magic[4];
    int32_t version, nshots, nrec, nt, flags, reserved;
    float dt;
    f.read(magic, 4);
    if (std::memcmp(magic, RTMS_MAGIC, 4) != 0)
        throw std::runtime_error(path + ": not an RTMS file (bad magic)");
    f.read((char*)&version, 4);
    if (version != 1) throw std::runtime_error("unsupported RTMS version");
    f.read((char*)&nshots, 4);
    f.read((char*)&nrec,   4);
    f.read((char*)&nt,     4);
    f.read((char*)&dt,     4);
    f.read((char*)&flags,  4);
    f.read((char*)&reserved, 4);
    if (nshots <= 0 || nrec <= 0 || nt <= 0 || !(dt > 0.0f))
        throw std::runtime_error("RTMS header contains invalid values");

    std::vector<ShotRecord> shots((std::size_t)nshots);
    for (auto& s : shots) {
        s.nt = nt; s.dt = dt;
        s.rx.resize(nrec); s.rz.resize(nrec);
        s.traces.resize((std::size_t)nrec * nt);
        f.read((char*)&s.sx, 4);
        f.read((char*)&s.sz, 4);
        f.read((char*)s.rx.data(), (std::streamsize)(nrec * sizeof(float)));
        f.read((char*)s.rz.data(), (std::streamsize)(nrec * sizeof(float)));
        f.read((char*)s.traces.data(),
               (std::streamsize)(s.traces.size() * sizeof(float)));
        if (!f) throw std::runtime_error("RTMS file truncated: " + path);
    }
    return shots;
}

// =============================================================================
// MINIMAL SEG-Y READER (rev 0/1)
//
// Supported : 3200-byte textual + 400-byte binary header, 240-byte trace
//             headers, constant trace length, big-endian, sample formats
//             1 (IBM float), 2 (int32), 3 (int16), 5 (IEEE float32).
// NOT       : extended textual headers, variable trace length, rev2 features.
//             If your data needs those, convert to the RAW format instead.
//
// Header fields used (1-based byte positions, SEG-Y standard):
//   binary 3217-3218 : sample interval (microseconds)
//   binary 3221-3222 : samples per trace
//   binary 3225-3226 : data sample format code
//   trace     9-12   : field record number (used to group traces into shots)
//   trace    71-72   : coordinate scalar
//   trace    73-76   : source X
//   trace    81-84   : group (receiver) X
// Depths are NOT taken from the headers (too many conventions) — pass them
// with --segy-src-depth / --segy-rec-depth.
// =============================================================================
static int16_t be16(const unsigned char* p) {
    return (int16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}
static int32_t be32(const unsigned char* p) {
    return (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                     ((uint32_t)p[2] << 8)  | (uint32_t)p[3]);
}
static float ibm_to_ieee(uint32_t v) {
    if (v == 0) return 0.0f;
    const int sign = (v >> 31) & 1;
    const int expo = (int)((v >> 24) & 0x7f) - 64;      // excess-64, base 16
    const uint32_t frac = v & 0x00ffffffu;
    const double m = (double)frac / 16777216.0;         // 2^24
    const double x = m * std::pow(16.0, (double)expo);
    return (float)(sign ? -x : x);
}
static float apply_scalco(int32_t coord, int16_t scalco) {
    if (scalco > 0) return (float)coord * (float)scalco;
    if (scalco < 0) return (float)coord / (float)(-scalco);
    return (float)coord;
}

std::vector<ShotRecord> read_shots_segy(const std::string& path,
                                        float src_depth, float rec_depth) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open SEG-Y: " + path);
    const std::streamsize file_bytes = f.tellg();
    if (file_bytes < 3600) throw std::runtime_error("SEG-Y too short: " + path);
    f.seekg(3200);
    unsigned char bh[400];
    f.read((char*)bh, 400);

    const int dt_us  = be16(bh + 16);   // 3217-3218
    const int ns     = be16(bh + 20);   // 3221-3222
    const int format = be16(bh + 24);   // 3225-3226
    if (ns <= 0) throw std::runtime_error("SEG-Y: invalid samples per trace");
    if (dt_us <= 0) throw std::runtime_error("SEG-Y: invalid sample interval");

    int bps;
    switch (format) {
        case 1: case 2: case 5: bps = 4; break;
        case 3:                 bps = 2; break;
        default: {
            std::ostringstream m;
            m << "SEG-Y sample format " << format
              << " is not supported (use 1, 2, 3 or 5)";
            throw std::runtime_error(m.str());
        }
    }
    const std::streamsize trace_bytes = 240 + (std::streamsize)ns * bps;
    const long ntraces = (long)((file_bytes - 3600) / trace_bytes);
    if (ntraces <= 0) throw std::runtime_error("SEG-Y contains no traces");

    const float dt = (float)dt_us * 1e-6f;
    std::vector<ShotRecord> shots;
    std::vector<unsigned char> buf((std::size_t)trace_bytes);

    int32_t cur_fldr = 0;
    float   cur_sx   = 0.0f;
    bool    started  = false;

    f.seekg(3600);
    for (long i = 0; i < ntraces; ++i) {
        f.read((char*)buf.data(), trace_bytes);
        if (!f) break;
        const unsigned char* th = buf.data();
        const int32_t fldr   = be32(th + 8);
        const int16_t scalco = be16(th + 70);
        const float   sx     = apply_scalco(be32(th + 72), scalco);
        const float   gx     = apply_scalco(be32(th + 80), scalco);

        const bool new_shot = !started ||
                              (fldr != cur_fldr) ||
                              (fldr == 0 && sx != cur_sx);
        if (new_shot) {
            shots.emplace_back();
            shots.back().nt = ns;
            shots.back().dt = dt;
            shots.back().sx = sx;
            shots.back().sz = src_depth;
            cur_fldr = fldr; cur_sx = sx; started = true;
        }
        ShotRecord& s = shots.back();
        s.rx.push_back(gx);
        s.rz.push_back(rec_depth);

        const unsigned char* sp = th + 240;
        std::size_t base = s.traces.size();
        s.traces.resize(base + (std::size_t)ns);
        for (int k = 0; k < ns; ++k) {
            float v = 0.0f;
            switch (format) {
                case 1: v = ibm_to_ieee((uint32_t)be32(sp + 4 * k)); break;
                case 5: { const int32_t bits = be32(sp + 4 * k);
                          std::memcpy(&v, &bits, 4); } break;
                case 2: v = (float)be32(sp + 4 * k); break;
                case 3: v = (float)be16(sp + 2 * k); break;
            }
            s.traces[base + (std::size_t)k] = v;
        }
    }
    if (shots.empty()) throw std::runtime_error("SEG-Y: no shots decoded");
    return shots;
}

std::vector<ShotRecord> read_shots(const std::string& path,
                                   float src_depth, float rec_depth) {
    const std::size_t dot = path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (ext == ".sgy" || ext == ".segy")
        return read_shots_segy(path, src_depth, rec_depth);
    return read_shots_raw(path);
}

} // namespace io
} // namespace rtm