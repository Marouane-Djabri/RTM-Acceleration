#pragma once
#include <string>
#include <vector>
#include "rtm_types.hpp"

namespace rtm {

// Simple direct-arrival mute (applied to data before migration).
void mute_direct_wave(ShotRecord& shot, float v_surface, float t_pad, float t_taper);

namespace io {

// ---- velocity model ---------------------------------------------------------
struct VelocityHeader {
    int   nx = 0, nz = 0;
    float dx = 10.0f, dz = 10.0f, ox = 0.0f, oz = 0.0f;
    std::string layout = "zfast";   // "zfast" (v[ix*nz+iz]) or "xfast" (v[iz*nx+ix])
};
bool read_header_file(const std::string& path, VelocityHeader& h);
void write_header_file(const std::string& path, const VelocityHeader& h);

VelocityModel read_velocity_raw(const std::string& path, const VelocityHeader& h);
void          write_velocity_raw(const std::string& path, const VelocityModel& m);

// ---- shot gathers -----------------------------------------------------------
void                    write_shots_raw(const std::string& path,
                                        const std::vector<ShotRecord>& shots);
std::vector<ShotRecord> read_shots_raw(const std::string& path);
std::vector<ShotRecord> read_shots_segy(const std::string& path,
                                        float src_depth, float rec_depth);
// Dispatch on extension: .sgy/.segy -> SEG-Y, anything else -> RAW.
std::vector<ShotRecord> read_shots(const std::string& path,
                                   float src_depth, float rec_depth);

// ---- plain float32 arrays (images) -----------------------------------------
void               write_raw_floats(const std::string& path, const std::vector<float>& v);
std::vector<float> read_raw_floats(const std::string& path);

} // namespace io
} // namespace rtm