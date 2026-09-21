#include "rtm_io.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rtm {
namespace io {

bool read_header_file(const std::string& path, VelocityHeader& h) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t c = line.find('#');
        if (c != std::string::npos) line = line.substr(0, c);
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq), val = line.substr(eq + 1);
        auto trim = [](std::string& s) {
            while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
        };
        trim(key); trim(val);
        if      (key == "nx")     h.nx = std::stoi(val);
        else if (key == "nz")     h.nz = std::stoi(val);
        else if (key == "dx")     h.dx = std::stof(val);
        else if (key == "dz")     h.dz = std::stof(val);
        else if (key == "ox")     h.ox = std::stof(val);
        else if (key == "oz")     h.oz = std::stof(val);
        else if (key == "layout") h.layout = val;
    }
    return true;
}

void write_header_file(const std::string& path, const VelocityHeader& h) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write header: " + path);
    f << "# RTM velocity model header\n"
      << "# data file: float32, little-endian, no header\n"
      << "nx = "     << h.nx     << "\n"
      << "nz = "     << h.nz     << "\n"
      << "dx = "     << h.dx     << "\n"
      << "dz = "     << h.dz     << "\n"
      << "ox = "     << h.ox     << "\n"
      << "oz = "     << h.oz     << "\n"
      << "layout = " << h.layout << "   # zfast: v[ix*nz+iz]  |  xfast: v[iz*nx+ix]\n";
}

VelocityModel read_velocity_raw(const std::string& path, const VelocityHeader& h) {
    if (h.nx <= 0 || h.nz <= 0)
        throw std::runtime_error("velocity nx/nz not set (use a .hdr file or --nx/--nz)");

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open velocity file: " + path);
    const std::streamsize bytes = f.tellg();
    const std::size_t n = (std::size_t)h.nx * h.nz;
    if ((std::size_t)bytes != n * sizeof(float)) {
        std::ostringstream m;
        m << "velocity file size mismatch: " << bytes << " bytes, expected "
          << n * sizeof(float) << " (nx=" << h.nx << ", nz=" << h.nz << ", float32)";
        throw std::runtime_error(m.str());
    }
    f.seekg(0);
    std::vector<float> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), bytes);

    VelocityModel m;
    m.grid.nx = h.nx; m.grid.nz = h.nz;
    m.grid.dx = h.dx; m.grid.dz = h.dz;
    m.grid.ox = h.ox; m.grid.oz = h.oz;
    m.v.resize(n);

    if (h.layout == "xfast") {          // v[iz*nx + ix] -> v[ix*nz + iz]
        for (int iz = 0; iz < h.nz; ++iz)
            for (int ix = 0; ix < h.nx; ++ix)
                m.v[(std::size_t)ix * h.nz + iz] = raw[(std::size_t)iz * h.nx + ix];
    } else {
        m.v = std::move(raw);
    }

    for (float v : m.v)
        if (!(v > 0.0f) || v > 20000.0f)
            throw std::runtime_error("velocity model contains a non-physical value");
    return m;
}

void write_velocity_raw(const std::string& path, const VelocityModel& m) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write velocity file: " + path);
    f.write(reinterpret_cast<const char*>(m.v.data()),
            (std::streamsize)(m.v.size() * sizeof(float)));
}

void write_raw_floats(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write: " + path);
    f.write(reinterpret_cast<const char*>(v.data()),
            (std::streamsize)(v.size() * sizeof(float)));
}

std::vector<float> read_raw_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open: " + path);
    const std::streamsize bytes = f.tellg();
    if (bytes % (std::streamsize)sizeof(float) != 0)
        throw std::runtime_error("file is not a float32 array: " + path);
    f.seekg(0);
    std::vector<float> v((std::size_t)bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

} // namespace io
} // namespace rtm