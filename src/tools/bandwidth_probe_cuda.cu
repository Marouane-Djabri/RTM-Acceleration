// GPU half of the bandwidth probe: a plain copy kernel, best of `repeats`,
// timed with CUDA events. Copy = 1 read + 1 write per element, which is the
// same traffic pattern as the stencil's dominant streams.
#include <cuda_runtime.h>
#include <string>

namespace rtm {

namespace {
__global__ void copy_kernel(const float* src, float* dst, std::size_t n) {
    const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
} // namespace

bool probe_gpu_copy_bandwidth(std::size_t bytes_per_buffer, int repeats,
                              std::string& gpu_name, double& gbps, std::string& error) {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        error = (status == cudaSuccess) ? "no CUDA device" : cudaGetErrorString(status);
        return false;
    }
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    gpu_name = properties.name;

    const std::size_t elements = bytes_per_buffer / sizeof(float);
    float* d_src = nullptr;
    float* d_dst = nullptr;
    if (cudaMalloc(&d_src, bytes_per_buffer) != cudaSuccess ||
        cudaMalloc(&d_dst, bytes_per_buffer) != cudaSuccess) {
        error = "cudaMalloc failed";
        cudaFree(d_src); cudaFree(d_dst);
        return false;
    }
    cudaMemset(d_src, 0, bytes_per_buffer);

    const int threads = 256;
    const int blocks  = (int)((elements + threads - 1) / threads);
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    float best_milliseconds = 1e30f;
    for (int repeat = 0; repeat < repeats + 1; ++repeat) {   // +1: warm-up
        cudaEventRecord(start_event);
        copy_kernel<<<blocks, threads>>>(d_src, d_dst, elements);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        if (repeat > 0 && milliseconds < best_milliseconds) best_milliseconds = milliseconds;
    }
    status = cudaGetLastError();

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_src);
    cudaFree(d_dst);
    if (status != cudaSuccess) { error = cudaGetErrorString(status); return false; }

    const double bytes_moved = 2.0 * (double)bytes_per_buffer;
    gbps = bytes_moved / (best_milliseconds * 1e-3) / 1e9;
    return true;
}

} // namespace rtm
