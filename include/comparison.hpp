#pragma once
#include <cstdio>
#include <ostream>
#include <vector>

namespace rtm {

// Phase 7 error metrics. `ref` is always the CPU reference image.
struct ErrorMetrics {
    std::size_t n         = 0;
    double max_abs        = 0;  // max |ref - test|
    double mean_abs       = 0;  // mean |ref - test|
    double rms            = 0;  // sqrt(mean (ref-test)^2)
    double l2_relative    = 0;  // ||ref-test||_2 / ||ref||_2
    double max_relative   = 0;  // max|ref-test| / max|ref|      <- scale-free
    double correlation    = 0;  // normalized zero-lag correlation, 1.0 == identical
    double ref_max_abs    = 0;
    double test_max_abs   = 0;
    std::size_t argmax    = 0;  // linear index of the worst sample
};

ErrorMetrics compare_images(const std::vector<float>& ref,
                            const std::vector<float>& test);
void print_metrics(std::ostream& os, const ErrorMetrics& m,
                   int nx = 0, int nz = 0);

} // namespace rtm