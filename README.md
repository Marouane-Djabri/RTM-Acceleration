# RTM-Hardware-Acceleration — 2D acoustic RTM, CPU reference baseline

A transparent, dependency-free 2D acoustic Reverse Time Migration in C++17,
built as a **correctness reference** for CPU/CUDA optimization research.
Every numerical operation lives in plain, readable source files.

## 1. What algorithm is implemented

Standard **shot-profile acoustic Reverse Time Migration** with a zero-lag
cross-correlation imaging condition:

1. Forward-propagate a Ricker source wavelet from the shot position, storing the
   source wavefield `p_s(x,z,t)`.
2. Back-propagate the recorded receiver traces from `t = T` to `t = 0`, giving
   `p_r(x,z,t)`.
3. Correlate at zero lag and stack over shots:
   `I(x,z) = Σ_shots Σ_t p_s(x,z,t) · p_r(x,z,t)`
4. Accumulate the source illumination `S(x,z) = Σ p_s²` for optional amplitude
   normalization.

## 2. Numerical method

Constant-density acoustic wave equation

    (1/c²) ∂²p/∂t² = ∇²p + f(t)·δ(x−xs, z−zs)

* **Time discretization** — explicit second-order leapfrog:
  `p^{n+1} = 2p^n − p^{n−1} + (cΔt)²·∇²p^n`
* **Space discretization** — central finite differences of order 2/4/6/8/10/12
  (default 8). Taylor-optimal coefficients, `src/rtm/rtm.cpp`.
* **Boundaries** — Cerjan (1985) exponential sponge, `nb` cells on all four
  sides, applied to `p^n` and `p^{n+1}` at every step.
* **Stability (enforced at startup)**
  `Δt ≤ 2 / (v_max·sqrt(S·(1/dx² + 1/dz²)))`, `S = |c₀| + 2Σ|c_k|`
  (S = 6.5016 for order 8). The program refuses to run if violated.
* **Dispersion (warned at startup)** — points per shortest wavelength using
  `f_max ≈ 2.5·f₀`. Keep ≥ 4 for order 8.
* **Source/receiver injection** — nearest grid point, scaled by `(cΔt)²`.

## 3. Assumptions

* 2D, acoustic, **constant density**. No elastic effects, no anisotropy, no
  attenuation, no free-surface multiples.
* **Absorbing top boundary** (no free surface). Data must be free-surface-
  multiple-free, or the multiples will migrate as false reflectors.
* Velocity model is the **migration velocity** and is assumed correct; there is
  no velocity analysis in this project.
* Nearest-grid-point injection — source/receiver coordinates are snapped to the
  grid. Sub-grid sinc interpolation is not implemented.
* Zero-lag cross-correlation only. No deconvolution imaging condition, no
  angle gathers, no least-squares RTM.
* Data are assumed already sampled at a `Δt` satisfying the CFL condition. No
  resampling is performed.
* Single precision (`float`) throughout. Switching to FP64 requires changing
  the `float` types in the kernels (a deliberate Phase-9 experiment).

## 4/5/6. Data formats

### Velocity model (`--velocity`)
Raw **float32, little-endian, no embedded header**, metres per second.
Default layout `zfast`: `v[ix*nz + iz]` — depth is the fastest axis.
Use `--vel-layout xfast` if your file is `v[iz*nx + ix]`.

Geometry comes from a sidecar text file `<velocity>.hdr` (auto-detected) or
from CLI flags (`--nx --nz --dx --dz --ox --oz --vel-layout`), CLI wins:

```
nx = 201
nz = 151
dx = 10
dz = 10
ox = 0
oz = 0
layout = zfast
```

### Shot gathers (`--shots`)

**A. RAW "RTMS" format** (`.bin`) — the recommended path.
Little-endian, 32-byte header, then each shot contiguously:

| offset | type      | field                                       |
|--------|-----------|---------------------------------------------|
| 0      | char[4]   | `"RTMS"`                                    |
| 4      | int32     | version = 1                                 |
| 8      | int32     | nshots                                      |
| 12     | int32     | nrec (constant across shots)                |
| 16     | int32     | nt                                          |
| 20     | float32   | dt (seconds)                                |
| 24     | int32     | flags = 0                                   |
| 28     | int32     | reserved = 0                                |

then per shot: `float32 sx, sz`, `float32 rx[nrec]`, `float32 rz[nrec]`,
`float32 data[nrec*nt]` **trace-major**: `d[ir*nt + it]`.
Coordinates are in metres in the same frame as the velocity model origin.

**B. SEG-Y** (`.sgy` / `.segy`) — a minimal built-in reader:
rev 0/1, big-endian, 3200-byte textual + 400-byte binary header, 240-byte trace
headers, **constant trace length**, sample formats 1 (IBM float), 2 (int32),
3 (int16), 5 (IEEE float32). Traces are grouped into shots by field record
number (bytes 9–12); source X from 73–76, receiver X from 81–84, coordinate
scalar from 71–72. Depths are **not** read from headers — pass
`--segy-src-depth` / `--segy-rec-depth`. Extended textual headers, variable
trace length and rev-2 features are not supported: convert to RAW in that case.

### Output image
Raw float32, `image[ix*nz + iz]`, same `nx*nz` grid as the velocity model.

## 7. Compile

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
The reference is built with `-O2 -g -fno-omit-frame-pointer` — **not** `-O3`,
`-march=native` or `-ffast-math`, so the reference image is reproducible across
machines. `-DRTM_FAST_REFERENCE=ON` relaxes this (do not use it for the image
you compare against).

## 8. Run

```bash
./build/rtm \
    --velocity data/synthetic/velocity.bin \
    --shots    data/synthetic/shots.bin \
    --output   results/reference_rtm_image.bin \
    --order 8 --nb 60 --f0 12 --store-interval 2 \
    --mute-direct 1500 --illumination --filter laplacian \
    --benchmark results/benchmark.txt
```
`--help` lists every option.

`--output` always receives the **raw zero-lag cross-correlation** — that is the
reference quantity for numerical comparison. Illumination compensation and the
Laplacian filter are written to separate `_illum.bin` / `_filtered.bin` files so
that post-processing never contaminates the baseline.

## 9. Generate the synthetic dataset

```bash
./build/rtm_synth --outdir data/synthetic \
    --nx 201 --nz 151 --dx 10 --dz 10 --nt 1400 --dt 0.001 --f0 12 --nshots 4
```
Three layers (1500 / 2000 / 2500 m/s) with a dipping second interface plus a
60×60 m, 2800 m/s box diffractor at (1200 m, 700 m). Four surface shots, 101
receivers at 20 m spacing.

The gathers are modelled with the **same** propagator used for migration — an
inverse crime. That is intentional: it validates the pipeline, not the physics.
`data/synthetic/` is validation data; put real data in `data/real/`.

**What a correct image looks like:** two continuous reflectors at ~400 m and at
the dipping ~900 m interface, and a focused blob at the diffractor. If you see
concentric rings, the direct wave is not muted; if you see wraparound energy
from the sides, increase `--nb`; if the image is high-frequency noise, the CFL
or dispersion check is being ignored.

## 10. Benchmark

Every run prints the report and appends it to `--benchmark`. Stages are timed
independently with `std::chrono::steady_clock`:
data loading / forward / backward (imaging excluded) / imaging / total, plus
reserved H2D, Kernel, D2H rows for the future CUDA engines. The report also
prints the stencil GFLOP count and the achieved GFLOP/s.

## 11. Where the bottlenecks are

| Function | File | Share | Why |
|---|---|---|---|
| `CPUReferenceRTM::fd_time_step` | `src/rtm/forward.cpp` | **~85–95 %** | called `2·nshots·nt` times over `nxe·nze` points |
| `CPUReferenceRTM::apply_sponge` | `src/rtm/forward.cpp` | ~3–8 % | two extra full passes over the wavefield per step |
| `CPUReferenceRTM::imaging` | `src/rtm/imaging.cpp` | ~2–5 % | one pass over the image grid per stored step |
| `save_snapshot` + snapshot writes | `src/rtm/forward.cpp` | ~1–5 % | pure memory traffic, grows with `store_interval = 1` |
| injection / recording | forward/backward | < 1 % | `O(nrec)` per step |

`fd_time_step` does `6·half + 5` flops per point (29 at order 8) while touching
~16 bytes per point → arithmetic intensity ≈ **1.8 flop/byte**. Against a
typical machine balance of 10–20 flop/byte this is firmly **memory-bandwidth
bound**, which is the single most important fact for planning optimizations.

## 12. What to modify for CPU optimization

Create `CPUOptimizedRTM : public RTMEngine` and touch only:

1. `fd_time_step` — cache blocking over z, `restrict` + `__builtin_assume_aligned`,
   explicit AVX2/AVX-512, temporal blocking, OpenMP over `ix`.
2. `apply_sponge` — fuse into `fd_time_step` (removes two wavefield passes).
3. `imaging` — fuse into the tail of the backward step.
4. `RTMEngine::migrate` — parallelize the **shot loop** (embarrassingly parallel;
   needs a private image buffer per thread plus a reduction).
5. Storage — replace full snapshot storage with boundary saving + reverse
   reconstruction, or Griewank checkpointing.

Do **not** modify `CPUReferenceRTM`; it defines the truth.

## 13. What to port to CUDA

See "CUDA roadmap" in the delivery notes. Order: `fd_time_step` →
fuse `apply_sponge` → fuse `imaging` → keep wavefields device-resident →
snapshot traffic.

## Correctness workflow

```bash
./build/rtm_compare results/reference_rtm_image.bin \
                    results/cuda_rtm_image.bin --nx 201 --nz 151
```
Reports max abs error, mean abs error, RMS, relative L2, max-relative and
normalized correlation. The default gate (`L2rel < 1e-5`, `corr > 0.9999`) is a
**documented convention, not a scientific law** — see the comment block in
`src/tools/compare_main.cpp` for the rationale and change it deliberately.

## Profiling

```bash
perf record -g --call-graph dwarf ./build/rtm ...   # then: perf report
perf stat -e cycles,instructions,cache-misses,LLC-load-misses ./build/rtm ...
nsys profile ./build/rtm ...        # once CUDA exists
ncu --set full ./build/rtm ...
```
The build keeps frame pointers and debug info, and each stage is a distinct
function, so `perf report` attributes time directly to the kernels above.