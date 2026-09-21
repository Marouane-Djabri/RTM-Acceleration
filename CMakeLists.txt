cmake_minimum_required(VERSION 3.18)
project(RTM_Hardware_Acceleration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The reference build deliberately avoids -O3/-march=native/-ffast-math:
# the reference image must be bit-reproducible across machines.
# -g -fno-omit-frame-pointer are required for usable `perf` call graphs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -fno-omit-frame-pointer")
endif()

option(RTM_FAST_REFERENCE "Allow -O3 -march=native for the reference build" OFF)
if(RTM_FAST_REFERENCE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -g -fno-omit-frame-pointer")
endif()

# OpenMP: used by the optimized CPU engine (Phase 1) and the bandwidth probe.
find_package(OpenMP)

# ---------------------------------------------------------------------------
# CUDA (docs/OPTIMIZATION_PLAN.md §2.4). Must be enabled BEFORE the targets are
# created so CMAKE_CUDA_ARCHITECTURES seeds every target's CUDA_ARCHITECTURES.
#   local compile check : no GPU, so a fixed list of common RunPod cards
#                         (75 T4, 80 A100, 86 RTX 30xx/A4000/A5000, 89 RTX 40xx/L4)
#   on the pod          : scripts/pod/setup.sh passes -DCMAKE_CUDA_ARCHITECTURES=native
# ---------------------------------------------------------------------------
option(RTM_WITH_CUDA "Build the CUDA engines and the GPU bandwidth probe" ON)
set(RTM_CUDA_ENABLED OFF)
if(RTM_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(RTM_CUDA_ENABLED ON)
  else()
    message(WARNING "CUDA compiler not found; building CPU engines only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Core library: reference engine, I/O, benchmark, engine factory.
# ---------------------------------------------------------------------------
add_library(rtm_core STATIC
  src/rtm/rtm.cpp
  src/rtm/rtm_cpu.cpp
  src/rtm/factory.cpp
  src/rtm/forward.cpp
  src/rtm/backward.cpp
  src/rtm/imaging.cpp
  src/io/velocity_io.cpp
  src/io/seismic_io.cpp
  src/benchmark/benchmark.cpp
  src/benchmark/comparison.cpp)
target_include_directories(rtm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimized CPU engine (Phase 1): its sources alone get -O3 -march=native so
# the reference objects keep their reproducible -O2.
file(GLOB RTM_CPU_OPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_opt/*.cpp)
if(RTM_CPU_OPT_SOURCES)
  target_sources(rtm_core PRIVATE ${RTM_CPU_OPT_SOURCES})
  target_include_directories(rtm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_opt)
  target_compile_definitions(rtm_core PUBLIC RTM_WITH_CPU_OPT)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${RTM_CPU_OPT_SOURCES} PROPERTIES
      COMPILE_OPTIONS "-O3;-march=native;-fopt-info-vec-optimized")
  endif()
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(rtm_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(RTM_CUDA_ENABLED)
  target_sources(rtm_core PRIVATE
    src/cuda/rtm_cuda.cu
    src/cuda/propagation_cuda.cu
    src/cuda/imaging_cuda.cu)
  target_compile_definitions(rtm_core PUBLIC RTM_WITH_CUDA)
  set_target_properties(rtm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(rtm_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(rtm                 src/main.cpp)
add_executable(rtm_synth           src/tools/make_synthetic.cpp)
add_executable(rtm_compare         src/tools/compare_main.cpp)
add_executable(rtm_bandwidth_probe src/tools/bandwidth_probe.cpp)
foreach(t rtm rtm_synth rtm_compare rtm_bandwidth_probe)
  target_link_libraries(${t} PRIVATE rtm_core)
endforeach()
if(RTM_CUDA_ENABLED)
  target_sources(rtm_bandwidth_probe PRIVATE src/tools/bandwidth_probe_cuda.cu)
endif()

file(MAKE_DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/results
  ${CMAKE_CURRENT_SOURCE_DIR}/results/ref
  ${CMAKE_CURRENT_SOURCE_DIR}/results/images
  ${CMAKE_CURRENT_SOURCE_DIR}/results/plots
  ${CMAKE_CURRENT_SOURCE_DIR}/data/synthetic
  ${CMAKE_CURRENT_SOURCE_DIR}/data/real)
