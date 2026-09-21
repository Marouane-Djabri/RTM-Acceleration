# Optimization Plan — fixed dataset, optimized CPU reference, CUDA ladder with hand-offs, scalability study

Goal: a sequence of CUDA optimizations of the 2D acoustic RTM where every rung
is a separately selectable engine, passes the correctness gate against
`CPUReferenceRTM` on **one fixed Marmousi dataset**, and adds one bar to a
speedup chart against the CPU — then a study showing the solution scales with
more GPUs and faster GPUs.

Kernel-level CUDA design (memory plan, thread mapping, kernel table, debugging
ladder) is in [CUDA_PLAN.md](CUDA_PLAN.md) §3–§4.

---

## 0. Working agreement

| topic | agreed |
|---|---|
| Who does what | **Claude:** writes all code (engines, kernels, scripts), **compiles locally** to verify it builds (g++ + nvcc 13.4; no GPU here), writes a runbook per rung. **You:** run everything on RunPod pods, send results back. |
| Nothing is executed locally | No benchmark, no reference, no dataset generation runs on the local machine. Local = edit + compile only. |
| Code style | Explicit variable names, simple readable code, no cleverness the feature doesn't need. Working + correct beats elegant. |
| Correctness truth | `CPUReferenceRTM` — never modified. Its Marmousi image, computed **once on the pod**, is the frozen reference. |
| Gate | `rtm_compare`: `L2rel < 1e-5` and `corr > 0.9999` against the frozen reference. Fused/tiled rungs additionally checked for bit-identity with their predecessor. |
| Dataset | **One** dataset for every engine, gate and bar: Marmousi, **12 shots**, fixed parameters (§1). Never regenerated after it is fixed. |
| CPU comparison | **No CPU ladder.** One optimized CPU engine `cpu-opt` (OpenMP + fused passes + vectorization) written directly. Charts show two lines: `cpu` (1×, truth) and `cpu-opt` (best CPU). |
| Per CUDA rung | Implement → compile → **stop** → hand over: "finished" + a short runbook (`docs/runbook/<engine>.md`) with the exact pod commands. You run it, send back `results/`. I chart and continue to the next rung. |
| Hosts | All bars come from **the same pod** (its CPU for `cpu` / `cpu-opt`, its GPU for `cuda-*`), so one chart, one honest 1×. Scaling runs use additional pods (§4). |

---

## 1. The fixed dataset (done once, never again)

Parameters live in **one file**, `scripts/dataset_marmousi.env`, sourced by
the generation script and by the benchmark script so nothing can drift:

```
MARMOUSI_VEL=data/real/marmousi_vp_12.5m.bin     # 1361 x 281, 12.5 m, from segy_to_raw.py --decimate 10
MARMOUSI_SHOTS=data/real/marmousi_shots.bin      # RTMS, 12 shots x 341 receivers x 2800 samples
NSHOTS=12   NT=2800   DT=0.001   F0=10
ORDER=8     NB=50     STORE=10   MUTE=1500   REC_DX=50   SRC_Z=25   REC_Z=25
```

* The file that already exists locally (`data/real/marmousi_shots.bin`, 12
  shots, 45.9 MB, generated 2026-09-20 by `rtm_synth` with exactly these
  parameters) **is** the dataset. It is rsync'd to the pod's persistent volume
  once. Its SHA-256 is recorded in `data/real/marmousi_shots.sha256`; the pod
  setup script verifies it so every run provably uses the same bytes.
* `scripts/make_marmousi_dataset.sh` (fixed parameters, no env overrides) is
  kept only so the dataset can be rebuilt from the SEG-Y if it is ever lost.
  `run_marmousi.sh` is retired.
* The reference image `results/ref/marmousi_ref.bin` is produced once on the
  pod by `--engine cpu` during setup and committed (1.5 MB) with its benchmark
  CSV row.

Why 12 shots: already generated, 46 MB, each GPU rung runs it in well under a
minute, the CPU reference in ~15–30 min once. Enough shots for a stable timing
and a realistic image.

---

## 2. Phase 0 — Infrastructure (Claude: write + compile; you: pod setup)

### 2.1 Engine registry
* `include/rtm_factory.hpp`, `src/rtm/factory.cpp`: `make_engine(name)`, `list_engines()`; CUDA engines under `#ifdef RTM_WITH_CUDA`.
* `src/main.cpp`, `src/tools/make_synthetic.cpp`: `--engine NAME` (default `cpu`), `--list-engines`, `--gpus N` (multi-GPU engine only).

### 2.2 Machine-readable output
* `rtm --benchmark-csv PATH`: one row per run —
  `date, host, cpu_name, threads, gpu_name, gpus, engine, dataset, nx, nz, nb, order, nt, nshots, store_interval, t_io, t_h2d, t_forward, t_backward, t_imaging, t_d2h, t_total, stencil_gflops, snapshot_mib`.
* `rtm_compare --csv PATH --engine NAME`: `date, host, engine, max_abs, mean_abs, rms, l2_relative, correlation, pass`.

### 2.3 Bandwidth probe — `src/tools/bandwidth_probe.cpp/.cu`
Measures achievable memory bandwidth (CPU OpenMP triad, GPU copy kernel; 256 MB each) → `results/ref/<host>_bandwidth.csv`. Feeds the roofline chart and the device-scaling prediction. Runs during pod setup.

### 2.4 CMake
CUDA language enabled (`RTM_WITH_CUDA` option, `CMAKE_CUDA_ARCHITECTURES` default `native` on the pod; local compile check uses `75;80;86;89`), `-lineinfo`, OpenMP found and linked for `cpu-opt`, `-O3 -march=native` applied **only** to `src/cpu_opt/*.cpp` (the reference keeps `-O2`).

### 2.5 Pod-side scripts — `scripts/pod/`
| script | does |
|---|---|
| `setup.sh` | `apt-get install cmake g++ git` if missing; build; verify dataset SHA-256; run bandwidth probe; run `--engine cpu` on the dataset **once** → `results/ref/marmousi_ref.bin` + reference CSV rows. Skips any step whose output already exists. |
| `bench.sh ENGINE [--gpus N]` | runs `rtm --engine ENGINE` with the fixed parameters → benchmark CSV → `rtm_compare` vs the reference → compare CSV → prints `ENGINE  total_s  speedup_vs_cpu  speedup_vs_cpu-opt  L2rel  PASS/FAIL`. |
| `bench_all.sh` | `bench.sh` for every engine in `--list-engines`. |
| `plot.sh` | `pip install matplotlib` if needed; runs `scripts/plot_benchmark.py` → `results/plots/*.png` + `docs/RESULTS.md`. |

### 2.6 Local-side scripts — `scripts/local/`
* `sync_to_pod.sh <ssh-alias>`: rsync repo + dataset to `/workspace/RTM` (excludes `build/`, `venv/`).
* `sync_from_pod.sh <ssh-alias>`: rsync `results/` back.

### 2.7 `scripts/plot_benchmark.py`
| chart | shows |
|---|---|
| `speedup.png` | speedup vs `cpu` per engine, log scale; dashed line at `cpu-opt`; hatched red if gate FAILed |
| `stages.png` | stacked forward / backward / imaging / H2D / D2H per engine |
| `bandwidth.png` | achieved stencil GB/s as % of the host's measured peak |
| `scaling_*.png` | §4 |

### Phase 0 hand-off
Runbook `docs/runbook/00_setup.md`: create pod (CUDA 12.x devel image, persistent `/workspace`, SSH), `sync_to_pod.sh`, `ssh pod scripts/pod/setup.sh`, `sync_from_pod.sh`. You send back `results/`. Acceptance: reference image + CSV rows exist; probe CSV exists; `speedup.png` shows one bar.

---

## 3. Phase 1 — `cpu-opt`, the optimized CPU comparison (Claude: write + compile)

One engine, `src/cpu_opt/cpu_opt.hpp/.cpp`, written directly with everything
that helps on a multi-core CPU, no intermediate rungs:

* Precompute tables copied verbatim from `CPUReferenceRTM::setup` (byte-identical `vdt2`, `sponge`, `wavelet`, coefficients).
* `#pragma omp parallel for` over `ix` in the stencil; sponge multiplied **inside** the stencil loop; imaging fused into the backward stencil pass on stored steps (removes 2–3 full wavefield passes per step).
* `__restrict__` wavefield pointers, 64-byte-aligned buffers, inner `iz` loop written so GCC auto-vectorizes; file compiled with `-O3 -march=native` (no fast-math). Vectorization verified with `-fopt-info-vec` in the local compile.
* `threads` column in the CSV = `omp_get_max_threads()`.

Gate: L2rel (FMA/vectorization change rounding). Expected on a pod CPU: 5–25× depending on its core count.

Hand-off: `docs/runbook/01_cpu_opt.md` → you run `bench.sh cpu-opt` + `plot.sh` → send `results/`. Chart now has `cpu` and `cpu-opt`.

---

## 4. Phase 2 — CUDA ladder (Claude: write + compile; you: run each rung)

One `CUDARTM` class with a `CudaVariant` selecting the kernel set; one kernel file per rung. Design in CUDA_PLAN.md §3.

```
src/cuda/
├── cuda_check.hpp          CUDA_CHECK macro, kernel-launch check
├── rtm_cuda.hpp/.cu        CUDARTM: setup, migrate, drivers, variant switch, cudaEvent timing
├── kernels_common.cu       inject source / receivers (atomicAdd), record traces, standalone imaging
├── kernels_v0_naive.cu     G0
├── kernels_v1_fused.cu     G1
├── kernels_v2_imaging.cu   G2
├── kernels_v3_shared.cu    G3
├── kernels_v4_regqueue.cu  G4
└── rtm_cuda_multi.hpp/.cu  S1 multi-GPU driver
```

| rung | engine | change | removes | gate |
|---|---|---|---|---|
| **G0** | `cuda-v0` | Naive mirror: one thread per point, `threadIdx.x ↔ iz`, separate kernels, snapshots device-resident, image downloaded once per `migrate`. | serial execution | L2rel |
| **G1** | `cuda-v1` | Sponge fused into the stencil kernel. | 2 wavefield passes/step | bit-identical to G0 |
| **G2** | `cuda-v2` | Imaging fused into the backward stencil on stored steps. | 1 wavefield read per stored step | bit-identical to G1 |
| **G3** | `cuda-v3` | Shared-memory tiling of `p_cur` with halo. | redundant global/L2 reads | bit-identical to G2 |
| **G4** | `cuda-v4` | Register queue along z, x-neighbours from shared memory. | shared-memory traffic | bit-identical to G3 |
| **G5** (stretch) | `cuda-v5` | Two streams + pinned host memory. | H2D/D2H bars | bit-identical to G4 |
| **G6** (stretch) | `cuda-v6` | Boundary saving + backward reconstruction of the source wavefield. | snapshot stream, `store_interval` compromise | L2rel |

**Hand-off per rung** — `docs/runbook/1x_<engine>.md`, always the same shape:

```
1. sync:      scripts/local/sync_to_pod.sh pod
2. build:     ssh pod "cd /workspace/RTM && cmake --build build -j"
3. run:       ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cuda-vN"
4. plot:      ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
5. return:    scripts/local/sync_from_pod.sh pod
6. send me:   the bench.sh summary line, and results/ (CSVs + plots)
If it FAILs or crashes: the exact commands to rerun with compute-sanitizer and --max-shots 1, and what output to paste back.
```

I don't start the next rung until the current one is PASS on the pod and its bar is on the chart.

---

## 5. Phase 3 — Scalability (Claude: write + compile; you: run on the pods described)

Built on the best passing single-GPU rung.

| study | engine / run | pod needed | chart | shows |
|---|---|---|---|---|
| **S1 strong scaling** | `cuda-multi --gpus 1,2,4`: one host thread per device, each with its own `CUDARTM`, shots dealt round-robin, private images summed in fixed order. Dataset: the fixed 12 shots (3 per GPU at 4 GPUs) **and** a 48-shot set modelled once on the pod (`scripts/pod/make_scaling_dataset.sh`, fixed params, checksum recorded). | 2× or 4× of the same modest card | `scaling_gpus.png` | time & speedup vs #GPUs, ideal line, efficiency % |
| **S2 weak scaling** | `cuda-multi` with 12 shots/GPU (12, 24, 48 shots for 1, 2, 4 GPUs). | same multi-GPU pod | `scaling_weak.png` | time vs #GPUs should be flat |
| **S3 device scaling** | Stencil is bandwidth-bound → **predict** the faster card's time from the two measured bandwidths, then run `setup.sh` + `bench.sh` for `cpu`, `cuda-v0`, best rung on it. | a faster card, short session | `scaling_devices.png` | measured time vs measured bandwidth per device with the predicted point hollow; straight line = scales with hardware as the roofline says |
| **S4 problem-size scaling** | best rung on Marmousi at 25 / 12.5 / 6.25 m (2 shots each, modelled on the pod, fixed params). | the main pod | `scaling_gridsize.png` | throughput (Mpoint-updates/s) vs grid size; flat = no cache cliff |

Each study is its own hand-off with its own runbook (`docs/runbook/2x_<study>.md`), including how to create the pod it needs.

---

## 6. Phase 4 — Results
* You run `bench_all.sh` once from a clean build on the main pod so all bars come from one commit; `sync_from_pod.sh`.
* I produce the final figures and `docs/RESULTS.md`: table + one paragraph per rung (what changed, what it removed, what the chart shows, gate result) + one section per scaling study.

---

## 7. Execution order

| step | Claude delivers | you run | result |
|---|---|---|---|
| 1 | Phase 0 code + `runbook/00_setup.md` | pod setup | reference frozen, 1 bar |
| 2 | `cpu-opt` + `runbook/01_cpu_opt.md` | `bench.sh cpu-opt` | best-CPU line |
| 3 | `cuda-v0` + runbook | `bench.sh cuda-v0` | first GPU bar |
| 4–7 | `cuda-v1` … `cuda-v4`, one at a time, each with runbook | one `bench.sh` each | one bar each |
| 8 | `cuda-multi` + S1/S2 runbook | multi-GPU pod | 2 scaling charts |
| 9 | S3 + S4 runbooks (no new engine) | faster-card pod; grid sweep on main pod | 2 scaling charts |
| 10 | stretch `cuda-v5`, `cuda-v6` if wanted | one `bench.sh` each | bars |
| 11 | final `RESULTS.md` + figures | one clean `bench_all.sh` | done |

---

## 8. Time estimate

My time is implementation + local compile + runbook. Your time is pod runs.

| step | Claude (implement + compile + runbook) | your pod time |
|---|---|---|
| 1 Phase 0 | 40–60 min | setup ≈ 5 min + CPU reference 15–30 min (once) |
| 2 `cpu-opt` | 20–30 min | 2–5 min |
| 3 `cuda-v0` | 45–60 min (+ fix cycles if it fails on the pod) | 1–2 min per attempt |
| 4–7 `cuda-v1`…`v4` | 20–40 min each | 1–2 min each |
| 8 `cuda-multi` + S1/S2 | 40–60 min | 10 min (incl. modelling 48 shots) |
| 9 S3 + S4 | 20–30 min | 15 min across two pods |
| 10 stretch | 30 min (v5) + 60–90 min (v6) | 2 min each |
| 11 results | 30 min | 5 min |
| **core (1–9, 11)** | **≈ 5–7 h** | **≈ 1–1.5 h** |

Debug cycles for a failing CUDA rung: ~10–20 min of my time each plus one pod run of yours.

---

## 9. Definition of done for any rung
- [ ] `--list-engines` shows it; local build passes (`g++` and `nvcc`).
- [ ] Runbook written; hand-off message sent.
- [ ] `bench.sh <engine>` on the pod → PASS; bit-identity vs predecessor where required.
- [ ] Bar in `speedup.png`; stage chart shows the expected stage shrinking.
- [ ] Row + "why" paragraph in `docs/RESULTS.md`.
- [ ] Committed with a message naming the engine and its speedup.
