# CUDA Optimization Plan

Step-by-step plan for building the first CUDA engine (`CUDARTM`), validating
it against `CPUReferenceRTM`, and then iterating toward a fast one. Written to
be followed top to bottom; every phase ends with an acceptance test.

---

## 0. Situation and ground rules

### Hardware reality (checked 2026-09-20)

| | |
|---|---|
| This machine | WSL2, 12 CPU cores, **no NVIDIA GPU** (Windows reports only *Intel(R) Graphics*) |
| CUDA toolkit | `nvcc` 13.4 at `/usr/local/cuda` — **compiles fine**, but `cudaGetDeviceCount` fails: *"CUDA driver version is insufficient"* (there is no driver because there is no GPU) |
| Consequence | Write and compile CUDA **here**; **execute** on a remote GPU (Google Colab T4/L4/A100, Kaggle, or a lab machine). See §7. |

Everything in §1–§2 works locally. From §3 on, each "run" step happens on the
remote GPU.

### Ground rules (from README §12 — do not bend these)

1. `CPUReferenceRTM` is never edited. It defines the truth.
2. Every engine must pass `rtm_compare` against the reference image
   (`L2rel < 1e-5` and `corr > 0.9999`). A PASS is the *only* definition of
   "works".
3. One change at a time. Measure before and after. If a step doesn't pass the
   gate, do not move on.
4. Correctness first (V0), then speed (V1+). A fast wrong engine is worth zero.

### The development loop

```
edit .cu / .hpp locally
   └─> cmake --build build            (local: catches compile errors, no GPU needed)
        └─> git commit + push          (or zip upload)
             └─> remote: pull, build, run rtm --engine cuda
                  └─> rtm_compare  ─── PASS? ──> record benchmark, next step
                                    └── FAIL? ──> §4 debugging ladder
```

---

## 1. Freeze the reference (local, ~30 min)

The reference image is what every CUDA result is compared against. Build it
once, keep a copy where nothing will overwrite it.

```bash
cd ~/RTM/RTM-Hardware-Acceleration
scripts/run_synthetic.sh                # builds, models 4 shots, migrates twice, checks reproducibility
mkdir -p results/ref
cp results/reference_rtm_image.bin      results/ref/synthetic_ref.bin
cp results/benchmark.txt                results/ref/synthetic_cpu_benchmark.txt
```

Also make a *small* Marmousi reference for the gate — the full 12-shot CPU run
is long, and the gate only needs a few shots:

```bash
./build/rtm --velocity data/real/marmousi_vp_12.5m.bin --shots data/real/marmousi_shots.bin \
    --output results/ref/marmousi_2shots_ref.bin \
    --order 8 --nb 50 --f0 10 --store-interval 10 --mute-direct 1500 --max-shots 2 \
    --benchmark results/ref/marmousi_2shots_cpu_benchmark.txt
```

Write down from the benchmark reports: **Forward**, **Backward**, **Total**
seconds and **Stencil rate** GFLOP/s. These are the numbers you beat.

**Acceptance:** `results/ref/` contains both reference images and both
benchmark files. Commit them (they are small: 121 KB and 1.5 MB).

---

## 2. Plumbing (local, ~1–2 h, compiles without a GPU)

Nothing here is "optimization"; it is the scaffolding that lets an engine be
selected, built and timed. Do it first so that from §3 on you only think about
kernels.

### 2.1 CMake — enable the CUDA language

`CMakeLists.txt` already has a commented "PHASE 8 SLOT" (lines 48–60). Turn it
on, with two additions: an option to force it off, and the target architecture.

```cmake
option(RTM_WITH_CUDA "Build the CUDA engine" ON)
if(RTM_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 89)    # T4, A100, L4 — covers Colab/Kaggle
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    target_sources(rtm_core PRIVATE
      src/cuda/rtm_cuda.cu
      src/cuda/propagation_cuda.cu
      src/cuda/imaging_cuda.cu)
    target_compile_definitions(rtm_core PUBLIC RTM_WITH_CUDA)
    set_target_properties(rtm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(rtm_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  else()
    message(WARNING "CUDA compiler not found; building CPU engines only")
  endif()
endif()
```

`-lineinfo` costs nothing and lets `nsys`/`ncu`/`compute-sanitizer` point at
source lines. The `-O2` flags at the top of the file are `CMAKE_CXX_FLAGS_*` and
do not touch `.cu` files; nvcc defaults to `-O3` for device code, which is what
you want.

### 2.2 The engine header — `include/rtm_cuda.hpp`

Mirror `rtm_cpu.hpp` one-to-one. Same public overrides, same private table
names with a `d_` prefix for device pointers. Sketch (fill in, don't copy):

```cpp
#pragma once
#include "rtm_engine.hpp"

namespace rtm {

class CUDARTM : public RTMEngine {
public:
    ~CUDARTM() override;                     // cudaFree everything
    const char* name() const override { return "CUDA naive"; }

    void setup(const VelocityModel&, const RTMParams&, const TimeAxis&) override;
    void forward_propagation(const ShotRecord&, std::vector<float>* snapshots,
                             std::vector<float>* recorded) override;
    void backward_propagation(const ShotRecord&, const std::vector<float>& snapshots,
                              std::vector<float>& image,
                              std::vector<float>& illumination) override;
    void imaging(const float*, const float*, std::vector<float>&,
                 std::vector<float>&) override;      // host-side fallback, see §3.5
    void migrate(const std::vector<ShotRecord>&, std::vector<float>& image,
                 std::vector<float>& illumination) override;   // device-resident loop

private:
    void map_geometry(const ShotRecord&);    // host: same as CPU, then upload rec_index

    // device tables (extended grid unless noted)
    float* d_vdt2_   = nullptr;
    float* d_sponge_ = nullptr;
    float* d_wavelet_= nullptr;              // nt
    float* d_pp_ = nullptr, *d_pc_ = nullptr, *d_pn_ = nullptr;
    float* d_snap_   = nullptr;              // nsnap * nx*nz   (interior)
    float* d_traces_ = nullptr;              // nrec * nt, trace-major
    float* d_image_  = nullptr, *d_illum_ = nullptr;   // nx*nz (interior)
    int*   d_rec_index_ = nullptr;           // nrec, extended-grid indices
    int    src_index_ = 0;
    int    nrec_cap_  = 0;                   // current allocation size of d_traces_/d_rec_index_
};

} // namespace rtm
```

Stencil coefficients (`c0`, `cx[1..half]`, `cz[1..half]`, `half`, `nxe`,
`nze`, `nb`) go into `__constant__` memory in `propagation_cuda.cu`, uploaded
once in `setup`.

### 2.3 Source files

```
src/cuda/
├── cuda_check.hpp          CUDA_CHECK macro (below) + a kernel-launch check helper
├── rtm_cuda.cu             setup(), ~CUDARTM(), map_geometry(), migrate()
├── propagation_cuda.cu     kernels 1–6, forward_propagation(), backward_propagation()
└── imaging_cuda.cu         kernel 7 + the imaging() override
```

The error macro — use it on **every** CUDA API call, no exceptions:

```cpp
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess)                                                  \
            throw std::runtime_error(std::string(#call) + ": " +                \
                                     cudaGetErrorString(e_) + " (" +            \
                                     __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
    } while (0)
// after every kernel launch, in debug builds:
#define CUDA_CHECK_KERNEL() do { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
```

Wrap the kernel check in `#ifdef RTM_CUDA_SYNC_DEBUG` once V0 passes — a
`cudaDeviceSynchronize` after every launch kills performance but is the
fastest way to find *which* kernel faults.

### 2.4 Engine selection — `--engine` flag

Add to `src/main.cpp` **and** `src/tools/make_synthetic.cpp` (the second one
gives you a forward-only test, see §4.2). Cleanest: one small factory used by
both.

```cpp
// include/rtm_factory.hpp
std::unique_ptr<RTMEngine> make_engine(const std::string& name);   // "cpu" | "cuda"
```

```cpp
// src/rtm/factory.cpp
#include "rtm_cpu.hpp"
#ifdef RTM_WITH_CUDA
#include "rtm_cuda.hpp"
#endif
std::unique_ptr<RTMEngine> make_engine(const std::string& name) {
    if (name == "cpu")  return std::make_unique<CPUReferenceRTM>();
#ifdef RTM_WITH_CUDA
    if (name == "cuda") return std::make_unique<CUDARTM>();
#endif
    throw std::runtime_error("unknown or unavailable engine: " + name);
}
```

In `main.cpp`: replace `CPUReferenceRTM engine;` (line 98) with
`auto engine = make_engine(engine_name);`, change the `engine.` uses to
`engine->`, parse `--engine NAME` (default `cpu`), add it to `usage()`, and set
`gpu = true` in the `EngineResult` when the engine is not the CPU one so the
report prints the H2D/Kernel/D2H rows (`benchmark.cpp:44-48`).

### 2.5 Acceptance for §2

- `cmake -S . -B build && cmake --build build -j` succeeds **on this machine**
  with the three `.cu` files containing only stubs that throw
  `"not implemented"`.
- `./build/rtm --engine cpu ...` produces a bit-identical image to before
  (run `rtm_compare` — this proves the factory change didn't touch the
  reference).
- `./build/rtm --engine cuda ...` fails here with a *clear* message from
  `CUDA_CHECK(cudaSetDevice(0))` — the "driver version is insufficient" error —
  not a segfault.
- Commit: `"Add CUDA build plumbing and --engine flag"`.

---

## 3. V0 — the "naive mirror" engine (the version to finish first)

**Principle:** reproduce `CPUReferenceRTM` on the GPU *as literally as
possible*. Same loop structure, same index convention (`index = ix*nze + iz`,
depth fastest), same order of floating-point operations per point. One thread
per grid point, plain global-memory loads, no shared memory, no fusion. It will
already be 10–50× faster than the CPU reference simply because it's parallel,
and — more importantly — it is the version you can *debug* against the CPU.

### 3.1 Device memory plan

| buffer | size (floats) | grid | lifetime |
|---|---|---|---|
| `d_vdt2_`, `d_sponge_` | `nxe·nze` | extended | setup → destructor |
| `d_pp_`, `d_pc_`, `d_pn_` | `nxe·nze` each | extended | setup → destructor |
| `d_wavelet_` | `nt` | — | setup |
| `d_snap_` | `nsnap·nx·nz` | interior | setup (this is the big one) |
| `d_rec_index_` | `nrec` | — | (re)allocated in `map_geometry` if nrec grows |
| `d_traces_` | `nrec·nt` | — | uploaded per shot in `backward_propagation` |
| `d_image_`, `d_illum_` | `nx·nz` | interior | `migrate` |

Sizes for reference: extended grid for Marmousi is 1461 × 381 = 2.2 MB per
buffer; snapshots at `--store-interval 10` are 280 × 1361 × 281 × 4 B = 428 MB;
synthetic at `--store-interval 2` is 85 MB. All fit easily on a 16 GB T4.
**Keep snapshots device-resident from day one** — never copy them to the host.
That's the single biggest structural win over a "port line by line" approach
and it costs nothing extra.

### 3.2 Host-side precompute (in `CUDARTM::setup`)

`CPUReferenceRTM::setup` (`src/rtm/rtm_cpu.cpp:9-87`) builds four tables on the
host: `vdt2_`, `sponge_`, `wavelet_`, and the scaled stencil coefficients.
Reproduce those four loops **verbatim** in `CUDARTM::setup` (same formulas,
same float arithmetic — copy them, do not "improve" them), then
`cudaMemcpy` each to its `d_` buffer and `cudaMemcpyToSymbol` the coefficients
into `__constant__` memory. Keep the same validation (`nb >= half+1`, model
size, `store_interval >= 1`) and compute `nsnap_` the same way.

Why not share code with the CPU engine? Because the rule is "don't touch
`CPUReferenceRTM`", and 40 duplicated lines are cheaper than the risk. If the
two tables ever differ you will see it as a FAIL in §4.1 immediately.

### 3.3 Thread mapping — the one decision that matters for V0

Memory is `index = ix * nze + iz`: **consecutive `iz` are consecutive in
memory**. So `threadIdx.x` must map to `iz`, never to `ix`, or every warp load
is a strided (uncoalesced) access and the kernel runs ~10× slower than it
should.

```
block = dim3(32, 8)           // 32 threads along z (one warp = 128 contiguous bytes), 8 along x
iz = half + blockIdx.x * 32 + threadIdx.x
ix = half + blockIdx.y *  8 + threadIdx.y
grid  = dim3(ceil((nze - 2*half) / 32.0), ceil((nxe - 2*half) / 8.0))
if (iz >= nze - half || ix >= nxe - half) return;        // guard, exactly the CPU loop bounds
```

Same mapping for every 2D kernel (sponge, snapshot copy, imaging), only with
their own bounds (`0..n_extended` for the sponge; `0..nx, 0..nz` interior for
snapshot and imaging).

### 3.4 Kernel table

Each kernel mirrors one CPU function. Keep the *body* the same as the CPU
loop body; only the loop header becomes the thread index.

| # | kernel | mirrors | threads | launch | notes |
|---|---|---|---|---|---|
| 1 | `k_fd_time_step(pp, pc, pn)` | `fd_time_step` (`forward.cpp:29-50`) | one per interior-of-extended point | 2D, §3.3 | read `c0, cx[], cz[], half` from `__constant__`; loop `k = 1..half` exactly as CPU; `pn[i] = 2*pc[i] - pp[i] + vdt2[i]*lap` in **that** order |
| 2 | `k_sponge(pc, pn, sponge, n)` | `apply_sponge` (`forward.cpp:56-63`) | one per extended point | 1D, 256/block | elementwise; two writes |
| 3 | `k_inject_source(pn, vdt2, idx, amp)` | `inject_source` (`forward.cpp:70-72`) | 1 | `<<<1,1>>>` | pass `wavelet_[it]` as a scalar argument (host has the wavelet), or index `d_wavelet_[it]` |
| 4 | `k_inject_receivers(pn, vdt2, rec_index, traces, nt, it, nrec)` | `inject_receivers` (`backward.cpp:20-27`) | `nrec` | 1D | **use `atomicAdd`**: two receivers clamped to the same cell must both add, as the sequential CPU `+=` does |
| 5 | `k_record_traces(pc, rec_index, out, nt, it, nrec)` | `record_traces` (`forward.cpp:75-79`) | `nrec` | 1D | gather; `out[ir*nt + it] = pc[rec_index[ir]]` |
| 6 | snapshot copy | `save_snapshot` (`forward.cpp:83-88`) | — | `cudaMemcpy2DAsync` | the interior is a sub-rectangle of the extended grid: `dst pitch = nz·4`, `src pitch = nze·4`, `width = nz·4 B`, `height = nx`, source offset `eidx(nb, nb)`. No kernel needed. |
| 7 | `k_imaging(snap_slot, pc, image, illum)` | `imaging` (`imaging.cpp:25-40`) | one per interior point | 2D | `f = snap[ix*nz+iz]; b = pc[(ix+nb)*nze + iz+nb]; image += f*b; illum += f*f`. No atomics — one thread owns one point. |

Kernel 1 pseudo-code (this is the shape, not the code):

```
i = ix*nze + iz
lap = c0 * pc[i]
for k in 1..half:
    lap += cx[k] * (pc[i + k*nze] + pc[i - k*nze]) + cz[k] * (pc[i + k] + pc[i - k])
pn[i] = 2*pc[i] - pp[i] + vdt2[i] * lap
```

### 3.5 Drivers

**`forward_propagation(shot, snapshots, recorded)`**

```
map_geometry(shot)                       // host: compute src_index_, rec_index → upload d_rec_index_
cudaMemset(d_pp_, d_pc_, d_pn_) to 0
if recorded: allocate/zero d_rec (nrec*nt)
for it = 0 .. nt-1:
    k_fd_time_step<<<>>>(d_pp_, d_pc_, d_pn_)
    k_inject_source<<<1,1>>>(d_pn_, wavelet_[it])
    k_sponge<<<>>>(d_pc_, d_pn_)
    swap(d_pp_, d_pc_); swap(d_pc_, d_pn_)          // pointer swap, same as vector::swap
    if recorded: k_record_traces<<<>>>(d_pc_, ..., it, ...)
    if it % store == 0: cudaMemcpy2DAsync(d_snap_ + (it/store)*nx*nz  <-  d_pc_ interior)
cudaDeviceSynchronize()
if recorded: cudaMemcpy D2H into *recorded            // makes rtm_synth --engine cuda work
```

The host `snapshots` pointer is **ignored** (device-resident). Document that in
the header comment — it's a deliberate deviation the base class allows
(`rtm_engine.hpp:51-53`).

**`backward_propagation(shot, snapshots, image, illum)`**

```
map_geometry(shot)
upload shot.traces → d_traces_  (H2D, time it into times.h2d)
cudaMemset(d_pp_, d_pc_, d_pn_) to 0
for it = nt-1 .. 0:
    k_fd_time_step<<<>>>(d_pp_, d_pc_, d_pn_)
    k_inject_receivers<<<>>>(d_pn_, d_traces_, it)
    k_sponge<<<>>>(d_pc_, d_pn_)
    swap; swap
    if it % store == 0: k_imaging<<<>>>(d_snap_ + (it/store)*nx*nz, d_pc_, d_image_, d_illum_)
cudaDeviceSynchronize()
```

Host `snapshots`/`image`/`illum` arguments are ignored here too; the device
copies are the live ones.

**`migrate(shots, image, illum)`** — override the base version
(`rtm.cpp:83-103`) because the base one passes host buffers:

```
cudaMalloc + cudaMemset d_image_, d_illum_ (nx*nz)
for each shot: forward_propagation(shot, nullptr, nullptr); backward_propagation(shot, {}, image, illum)
cudaMemcpy D2H d_image_ → image, d_illum_ → illum   (time into times.d2h)
times.total += ...
```

**`imaging(...)` override** — required by the interface. Implement it as a
thin host wrapper (upload both inputs, run kernel 7, download) *or* leave it
throwing "use migrate()". The base `migrate` never calls it once you override
`migrate`. Pick the throw for V0; it's honest.

### 3.6 Timing rules

Kernels are asynchronous. A host `Timer` around a launch measures nothing.
Either `cudaDeviceSynchronize()` before every `elapsed()` read, or use
`cudaEventRecord`/`cudaEventElapsedTime` pairs. For V0: synchronize at the end
of each driver and time the whole driver — that fills `times.forward` and
`times.backward` correctly. Fill `times.h2d` (traces upload) and `times.d2h`
(final image download) with events around those copies. `times.kernel` can
stay 0 in V0.

### 3.7 Invariants checklist — tape this above your monitor

- [ ] Stencil threads cover **only** `[half, n-half)` in both axes; the outer
      `half` ring stays 0 forever (`forward.cpp:48-49`).
- [ ] Sponge multiplies **both** `pc` and `pn`, every step, over the **whole**
      extended grid.
- [ ] After the two swaps, `d_pc_` is "the field at time index `it`" — the
      record/snapshot/imaging kernels all read `d_pc_`, never `d_pn_`.
- [ ] Forward stores at `it % store == 0` into slot `it / store`; backward
      reads slot `it / store` at the same `it`. Both loops use the same
      `store`.
- [ ] `inject_receivers` uses `atomicAdd`.
- [ ] Every kernel guards its bounds; every CUDA call is wrapped in
      `CUDA_CHECK`.
- [ ] `map_geometry` is called at the start of **both** drivers (the receiver
      set can change per shot in SEG-Y data).
- [ ] The engine zeroes the three wavefield buffers at the start of both
      drivers — a stale wavefield from the previous shot is a classic
      "image looks almost right" bug.

### 3.8 Acceptance for §3

Remote GPU, synthetic dataset:

```bash
./build/rtm --engine cuda --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output results/cuda_v0_image.bin --order 8 --nb 60 --f0 12 --store-interval 2 --mute-direct 1500 \
    --benchmark results/benchmark_cuda_v0.txt
./build/rtm_compare results/ref/synthetic_ref.bin results/cuda_v0_image.bin --nx 201 --nz 151
```

→ `PASS`. Then the same on `marmousi_2shots_ref.bin` with `--max-shots 2`.
Commit as `"CUDARTM v0: naive mirror, passes gate"` and record the speedup.

---

## 4. Validation ladder (what to do when it does not PASS)

### 4.1 First run: expect *near bit-identity*

nvcc contracts `a*b + c` into FMA by default; GCC at `-O2` without
`-march=native` does not. That alone gives `L2rel ~ 1e-7 … 1e-6`. To separate
"rounding" from "bug", build V0 once with

```cmake
target_compile_options(rtm_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
```

With FMA off and the same per-point operation order, the images should be
**bit-identical or within 1e-7**. If `L2rel > 1e-5` with FMA off, it is a bug,
full stop (`compare_main.cpp:41-42` says the same).

### 4.2 Bisect by stage: the forward-only test

`rtm_synth --engine cuda` exercises **only** `forward_propagation` with
`recorded != nullptr`. Model the synthetic shots with both engines and compare
the traces:

```bash
./build/rtm_synth --engine cpu  --outdir /tmp/cpu  --nshots 1 --nt 600
./build/rtm_synth --engine cuda --outdir /tmp/cuda --nshots 1 --nt 600
data/synthetic/venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from read_shots import read_shots
import numpy as np
_, a = read_shots("/tmp/cpu/shots.bin"); _, b = read_shots("/tmp/cuda/shots.bin")
d = np.asarray(a.traces[0]) - np.asarray(b.traces[0])
print("max|diff| =", np.abs(d).max(), " rel L2 =", np.linalg.norm(d)/np.linalg.norm(a.traces[0]))
EOF
```

- Traces match, image doesn't → bug is in backward/imaging/migrate.
- Traces don't match → bug is in kernels 1–3/5 or setup tables. Reduce further:
  `--nt 1` (only injection), `--nt 2` (one stencil step), `--order 2`
  (`half = 1`, easiest to hand-check).

### 4.3 Tools

```bash
compute-sanitizer --tool memcheck  ./build/rtm --engine cuda ... --max-shots 1 --nt 50
compute-sanitizer --tool racecheck ./build/rtm --engine cuda ... --max-shots 1 --nt 50
```

`memcheck` catches out-of-bounds halo reads instantly; `racecheck` catches a
missing `atomicAdd`. Run both with a tiny `--nt`; they are 100× slower than
native.

### 4.4 Symptom → cause

| symptom | most likely cause |
|---|---|
| `illegal memory access` | stencil launched without the `half` guard, or `rec_index` not uploaded / wrong size |
| Image all zeros | `d_image_` never downloaded, or `imaging` launched on `d_pn_` instead of `d_pc_` |
| Image shifted a few cells | `nb` offset missing in kernel 7 or in the `cudaMemcpy2D` source pointer |
| Image "almost right", `L2rel ~ 1e-2` | wavefield buffers not zeroed between shots, or forward/backward `store` slot mismatch |
| Rings around receivers | forgot `--mute-direct` on one of the two runs (not a CUDA bug) |
| `L2rel ~ 1e-3` only on Marmousi, PASS on synthetic | atomics missing in `inject_receivers` — Marmousi has 341 receivers, more clamping collisions |
| Runs 10× slower than expected | `threadIdx.x` mapped to `ix` (uncoalesced), see §3.3 |
| Times are ~0 | missing `cudaDeviceSynchronize()` before `Timer::elapsed()` |

---

## 5. Benchmarking V0 properly

Once PASS, run the full 12-shot Marmousi job on the GPU and the CPU reference
for the same job (the CPU run is long — do it once, overnight if needed, or
compare per-shot times from a `--max-shots 2` run and scale).

Record, per engine:

| metric | where |
|---|---|
| Forward / Backward / Imaging / Total | benchmark report |
| Stencil GFLOP/s | benchmark report |
| **Effective bandwidth** = `bytes_per_point × points × steps / time` | compute by hand: ~16 B/point/step for kernel 1 (`forward.cpp:21`) |
| % of peak bandwidth | T4: 320 GB/s; L4: 300 GB/s; A100: 1 555 GB/s |

The last row is the number that tells you how far V0 is from the roofline and
therefore how much V1–V3 can gain. Expect V0 at maybe 20–40 % of peak: the
sponge and imaging passes are separate kernels re-reading the whole wavefield.

---

## 6. After V0 passes: the optimization ladder

Each version is a new commit, must PASS the gate, and gets a row in a results
table. Never skip the gate because "it's just a fusion".

| version | change | why | expected gain | gate |
|---|---|---|---|---|
| **V1** | Fuse `k_sponge` into `k_fd_time_step` (multiply `pn[i]` and `pc[i]` by `sponge[i]` in the same thread) | removes two full read+write passes over the wavefield per step (`forward.cpp:53-54`) | 1.3–1.5× | bit-identical to V0 (same ops, same order — `pc[i] *= w` then `pn[i] *= w`) |
| **V2** | Fuse `k_imaging` into the backward stencil (the thread that computed the point also does `image += f*b`) | removes another wavefield-sized read per stored step | 1.1–1.2× at `store=2`, less at `store=10` | bit-identical |
| **V3** | Shared-memory tiling: each block loads its `(32+2h) × (8+2h)` tile of `pc` into `__shared__`, stencil reads from shared | the naive kernel reads each point `1 + 2·order` times from L1/L2; tiling reads it ~once from global | 1.5–2.5× on the stencil | bit-identical |
| **V3b** | Register queue along z instead of / in addition to shared tiles (each thread walks a column, keeps `2h+1` values in registers) | classic 3D-stencil trick; in 2D with z fastest it turns the z-derivative into register reuse | try after V3, measure | bit-identical |
| **V4** | Streams: overlap the per-shot traces H2D and the final D2H with compute; pinned host memory (`cudaMallocHost`) for the trace buffer | hides the H2D row of the report | small unless nrec·nt is large | bit-identical |
| **V5** | Snapshot traffic: replace stored snapshots with boundary saving + reverse-time reconstruction, or checkpointing | removes the 428 MB write/read stream and the `store_interval` compromise | memory ↓ 100×, time ~neutral | `L2rel` gate (reconstruction is *not* bit-identical) |
| **V6** | Multi-shot / multi-GPU: shots are independent (`rtm.cpp:80-81`) | scaling, not per-shot speed | linear in GPUs | bit-identical per shot; stack order affects the last bits |

Write the results table in `docs/RESULTS.md` as you go: version, dataset,
Total s, Stencil GFLOP/s, effective GB/s, % peak, speedup vs CPU, gate result.

---

## 7. Remote GPU recipe (Colab / Kaggle)

Colab free tier gives a T4 (`sm_75`, 16 GB, CUDA 12.x preinstalled).
Kaggle gives a T4 ×2 or P100. Do **not** rely on CUDA-13-only APIs in the
code; everything in this plan is CUDA 11-era API.

Get the repo there. The repo has no remote yet; add one:

```bash
# once, locally
git remote add origin git@github.com:<you>/RTM-Hardware-Acceleration.git
git push -u origin master
```

Colab notebook, one cell:

```bash
%%bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
git clone https://github.com/<you>/RTM-Hardware-Acceleration.git || (cd RTM-Hardware-Acceleration && git pull)
cd RTM-Hardware-Acceleration
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
# synthetic data is generated on the fly (seconds); Marmousi shots must be uploaded or re-modelled
./build/rtm_synth --outdir data/synthetic --nx 201 --nz 151 --dx 10 --dz 10 --nt 1400 --dt 0.001 --f0 12 --nshots 4
./build/rtm --engine cuda --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output results/cuda_image.bin --order 8 --nb 60 --f0 12 --store-interval 2 --mute-direct 1500 \
    --benchmark results/benchmark_cuda.txt
./build/rtm_compare results/ref/synthetic_ref.bin results/cuda_image.bin --nx 201 --nz 151
```

For Marmousi on Colab, either upload `data/real/marmousi_vp_12.5m.bin(+.hdr)`
(1.5 MB) and re-run `rtm_synth --velocity ... --engine cuda` there, or upload
the 46 MB shots file to Drive once and mount it. Keep the reference images in
git so `rtm_compare` always has its truth.

If a lab machine with an NVIDIA GPU is reachable over SSH, prefer it — the
loop is faster than Colab and the data can just stay there.

---

## 8. Today's checklist

Time-boxed for one working day. The goal is a **PASS on the synthetic set from
`--engine cuda`** and a recorded speedup; everything in §6 is for later.

| when | what | done when |
|---|---|---|
| 0:00–0:30 | §1 freeze the reference | `results/ref/` committed |
| 0:30–2:00 | §2 plumbing: CMake, header, stubs, factory, `--engine` | builds locally; `--engine cpu` still bit-identical; `--engine cuda` errors cleanly |
| 2:00–2:30 | §7 set up GitHub remote + a Colab notebook that clones and builds | remote build succeeds, `nvidia-smi` shows a GPU |
| 2:30–4:30 | §3.2–3.4 `setup` + kernels 1, 2, 3, 5 + `forward_propagation` | `rtm_synth --engine cuda` produces traces; §4.2 test shows `max|diff| < 1e-5` |
| 4:30–6:00 | kernels 4, 6, 7 + `backward_propagation` + `migrate` | `rtm --engine cuda` writes an image |
| 6:00–7:00 | §3.8 gate; §4 if it fails | `rtm_compare` → PASS |
| 7:00–7:30 | §5 benchmark, write the first row of `docs/RESULTS.md`, commit | speedup number written down |

If §3 isn't passing by hour 7, stop and run the §4.2 forward-only test with
`--order 2 --nt 2` — that reduces the problem to something you can verify by
hand in ten minutes.
