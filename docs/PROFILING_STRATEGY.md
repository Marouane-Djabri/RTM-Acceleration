# Profiling Strategy — Nsight Systems + Nsight Compute

Source request: `~/RTM/profiling.md`

> **Part A. For each kernel:** profile with Nsight Systems and Nsight Compute,
> show **the same metrics** for every version, and identify the **optimization
> points** and the **blocking points**.
>
> **Part B. Then for Devito, cuQRTM and Rotax:** use Nsight Compute and Nsight
> Systems to show **how much more data we can process** and **how much seismic
> processing can be accelerated**.
>
> *Scope decision (2026-09-25): cuQRTM and Rotax are dropped; Part B compares our
> engine with Devito only.*

This document has two jobs:

1. **For Claude**, it is the implementation plan. §3 lists every code and script
   change, in order, with the files involved.
2. **For you**, it is the operating manual. §4–§8 say exactly which command to
   run on the pod, which file comes back, where to click in each GUI, and which
   number to copy into the report.

---

## 0. What you will have at the end

| # | Deliverable | Produced by | Goes in the report as |
|---|---|---|---|
| D1 | Timeline screenshot for each engine (v0…v4), one shot, zoomed to ~5 time steps | Nsight Systems GUI | "What the GPU is doing": gaps, launch count, kernels per step |
| D2 | **Per-step GPU time breakdown** (stencil / sponge / imaging / injection / memcpy) per engine | `nsys stats` → `scripts/profile_extract.py` | Stacked bar chart |
| D3 | **The kernel metrics table**: the same ~15 metrics for the main kernel of every version | `ncu` → `scripts/profile_extract.py` | Main table of the performance chapter |
| D4 | **Roofline chart** with v0…v4 as points | Nsight Compute GUI (screenshot) + our own plot from the same numbers | "Why it is faster / why it stopped getting faster" |
| D5 | Warp-stall breakdown per version | Nsight Compute, *Warp State Statistics* | Proof of the blocking point of each version |
| D6 | "Optimization points / blocking points" table (§4.7) | You + me, from D1–D5 | Conclusion of Part A |
| D7 | **Capacity curve**: device memory vs grid size, with the 24 GB line, per snapshot policy | capacity sweep + `nsys --cuda-memory-usage` | "How big a model fits on one GPU" |
| D8 | **Efficiency curve**: GPts/s and % of peak DRAM bandwidth vs grid size | capacity sweep + `ncu` | "How well the GPU is used as the problem grows" |
| D9 | Same D7/D8/D3 numbers for Devito-GPU | §6 | Comparison chapter |
| D10 | Survey-level projection (hours, $, kWh for 1000 shots at 6.25 m and 2.5 m, CPU vs GPU) | `scripts/project_survey.py` | "What this means for a seismic processing company" |
| D11 | **Roofline vs stencil order** (4…16), our engine and Devito, at dx = 2.5 m | order sweep S2 + `ncu` | "What happens when the problem is made compute-heavier" |

---

## 1. Starting point: what the numbers already say

From `results/benchmarks.csv` (RTX 3090, marmousi12 = 1361×281 grid, 12 shots,
2800 steps, order 8). Second run of each engine (warm):

| engine | total (s) | vs cpu (211.6 s) |
|---|---|---|
| cuda-v0 | 1.836 | 115× |
| cuda-v1 | 1.421 | 149× |
| cuda-v2 | 1.427 | 148× |
| cuda-v3 | 1.478 | 143× |
| cuda-v4 | 1.512 | 140× |

Back-of-envelope throughput for v1:
`(1461 × 381 extended points) × 2800 steps × 2 (fwd+bwd) × 12 shots = 37.4 G point-updates`
→ `37.4 G / 1.421 s ≈ 26 GPts/s`.

Minimum DRAM traffic of the v1 kernel with perfect neighbour reuse: read `p_prev`,
`p_cur`, `vdt2`, `sponge` and write `p_cur`, `p_next` = **24 bytes per point**.
`26 GPts/s × 24 B = 630 GB/s`, which is **~75 % of the 843 GB/s** our bandwidth
probe measured. Two hypotheses follow, and the profiling must confirm or refute
them:

* **H1: v1 is already bandwidth-bound.** v3 (shared memory) and v4 (warp shuffle)
  cannot help because they reduce *neighbour re-reads*, and L1/L2 already absorbs
  those. On an RTX 3090 the L2 is 6 MB and one wavefield array is only 2.2 MB, so
  a large part of the working set stays in L2. **Expected ncu evidence:** L2 hit rate
  is high for v1, DRAM throughput is close to peak, and v3 shows *Stall Barrier* plus
  more instructions per point.
* **H2: at this grid size, launch overhead is visible.** Each time step launches
  3–4 kernels, and one step costs about 20 µs of GPU time. That puts the CPU launch
  rate close to the GPU execution rate. **Expected nsys evidence:** small gaps
  between kernels on the *CUDA HW* row, and `cudaLaunchKernel` taking a large share
  of the *CUDA API* summary.
* **H3: the grid is too small for the GPU.** H1 and H2 both change with grid size:
  at larger grids the L2 no longer holds the working set, and launch overhead
  becomes negligible. The scaling sweep in Part B (§5) tests this.

---

## 2. Tool setup

### 2.1 On the pod (where profiling runs)

`nsys` and `ncu` ship with the CUDA toolkit in every *devel* image:

```bash
ssh pod "nsys --version; ncu --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
```

If `nsys` is missing: `apt-get install -y nsight-systems-cli` (NVIDIA apt repo) or
use `/usr/local/cuda/bin/nsys`. Note both version numbers for §2.2.

**Permission test for performance counters (do this first).** Many container
hosts block GPU counters. Test in ~10 seconds:

```bash
ssh pod "cd /workspace/RTM && ncu --metrics gpu__time_duration.sum -k regex:k_fd_time_step -c 1 \
   ./build/rtm --engine cuda-v1 --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
   --output /tmp/x.bin --order 8 --nb 60 --f0 12 --nt 50 --max-shots 1 --quiet"
```

| Result | Meaning | Do |
|---|---|---|
| A table with `gpu__time_duration.sum` | Counters available | Continue |
| `ERR_NVGPUCTRPERM` | The host driver restricts counters to admin; this cannot be fixed from inside the container | Keep this pod for the nsys work. For the ncu work, use a host that exposes counters: a full VM (e.g. Lambda, or a cloud VM with an NVIDIA GPU) where you can run as root or set `NVreg_RestrictProfilingToAdminUsers=0`, or try another RunPod host / Colab / Kaggle. **Record which GPU the ncu data comes from**, because every ncu number is GPU-specific. |

`nsys --gpu-metrics-devices` (the *SM Active / DRAM bandwidth* rows) needs the same
permission. Without it nsys still records the full timeline and only loses those
rows.

### 2.2 On your laptop (where you look at the reports)

Profile on the pod, read in the GUI locally. The GUIs do not need an NVIDIA GPU.

* Install **Nsight Systems** and **Nsight Compute** on **Windows**, not in WSL.
  Both are free downloads from developer.nvidia.com and need an NVIDIA developer
  login. The native Windows GUIs are much smoother than running them through WSLg.
* **The GUI version must be ≥ the pod CLI version** from §2.1. An older GUI refuses
  newer reports.
* The report files come back through `scripts/local/sync_from_pod.sh` (after
  modification M5) into `results/profiles/`. From Windows, open them at
  `\\wsl$\<distro>\home\samir\RTM\RTM-Hardware-Acceleration\results\profiles\`.

---

## 3. Code changes (the implementation plan for Claude)

Constraints: `CPUReferenceRTM` stays untouched. Code stays simple, with explicit
names. No engine changes its numerical output, so every gate still passes.

| # | Change | Files | Why |
|---|---|---|---|
| **M1** | **NVTX ranges.** An RAII helper `NvtxRange range("forward shot 3")` that compiles to nothing when `RTM_WITH_NVTX` is off. Ranges: `setup`, `shot N` > `forward` / `backward`, `download image`, plus `read inputs` / `write output` in `main.cpp`. NVTX v3 is header-only in CUDA 12/13 (`#include <nvtx3/nvToolsExt.h>`), so the CMake change is just an option plus the include path. | new `src/cuda/nvtx_range.hpp`, `rtm_cuda.cu`, `propagation_cuda.cu`, `main.cpp`, `CMakeLists.txt` | Nsight Systems then shows named phases on the timeline. Nsight Compute can filter kernels by range (`--nvtx-include`), for example to profile only kernels inside `backward`. |
| **M2** | **Device-memory accounting.** Call `cudaMemGetInfo` after `setup()` and after the per-shot allocations. Record peak used MiB in a new CSV column `device_mem_mib`, **appended at the end** so existing rows stay readable. Also add a `gpts_per_s` column. Print both in the text report. | `rtm_cuda.cu`, `benchmark.cpp/.hpp`, `rtm_engine.hpp` (StageTimes), `summary.py`, `plot_benchmark.py` | Gives capacity numbers for every run, without a profiler (needed for D7). |
| **M3** | **Profiling scripts** that fix every flag so every engine is profiled identically: `scripts/pod/profile_nsys.sh ENGINE [DATASET_ENV]`, `scripts/pod/profile_ncu.sh ENGINE [DATASET_ENV]`, `scripts/pod/profile_ladder.sh` (loops cuda-v0…v4). Outputs go to `results/profiles/<dataset>/{nsys,ncu}/<engine>.{nsys-rep,ncu-rep}` plus CSV exports next to them. | new scripts | Every engine gets identical flags, as the "same metrics" requirement demands. |
| **M4** | **Extraction + plots.** `scripts/profile_extract.py` reads `nsys stats` CSVs and `ncu --page raw --csv` exports and writes `results/profiles/kernel_metrics.csv` (D3), `step_breakdown.csv` (D2) and plots: stacked per-step bars, metric bars per version, and our own roofline (D4). | new script | Report figures come straight from data, without copying numbers by hand. |
| **M5** | `sync_from_pod.sh`: also pull `profiles/***` and `capacity*.csv`. | `scripts/local/sync_from_pod.sh` | So the reports come back to the laptop. |
| **M6** | **Scaled datasets.** `scripts/pod/make_scaled_dataset.sh DECIMATION` writes `data/scaled/dx<dx>/` (velocity, 2 shots, `dataset.env`) from the compressed SEG-Y shipped in `data/real/`. It decimates the same 1.25 m SEG-Y model by 10, 5, 4, 3, 2 and 1 (dx = 12.5, 6.25, 5, 3.75, 2.5, 1.25 m), scales `DT` for CFL and `NT` to keep the same 2.8 s record, **scales `F0` = 10 Hz × 12.5/dx to keep points per wavelength constant **, and models the shots with **`--engine cuda-v1`**, because the CPU is far too slow at fine grids. The same files feed our engine and Devito (§5.2). | new script, reuses `segy_to_raw.py`, `rtm_synth` | Input for sweeps S1/S2 (D7, D8, D11), for both codes. |
| **M7** | **Capacity + order sweeps.** `scripts/pod/capacity_sweep.sh ENGINE` loops over the scaled datasets (S1) and two snapshot policies (§5.2). For each run it executes a timing run, one `nsys` run with memory tracking and one short `ncu` run of the stencil kernel. It appends to `results/scaling.csv` (shared with Devito), logs GPU power with `nvidia-smi`, and records OOM as a data point instead of crashing the loop. `scripts/pod/order_sweep.sh ENGINE` runs S2 (orders 4/8/12 at dx = 2.5 m). | new scripts | D7, D8, D11 |
| **M8** | **Devito on the GPU, instrumentation and scaling.** `marmousi_rtm_devito.py` gains `--nvtx` (ranges per shot / forward / backward), `--nt` (short ncu runs), `--run-json` (timings + Devito's own GPts/s, GFlops/s, OI), a CFL check per order, and JIT compilation forced before the timed loop. New `scripts/pod/setup_devito_gpu.sh` (NVIDIA HPC SDK via apt + Devito venv + GPU smoke test) and `scripts/pod/devito_sweep.sh grid|order` (rows in `results/scaling.csv` with `code = devito`). 3D (S3) is not implemented; it waits for your decision. | `scripts/devito/…`, new scripts | Devito in Nsight (GPU only), on the same scaled inputs as our engine |
| **M11** | `scripts/project_survey.py`: turns measured s/shot, GPts/s and power into survey-level hours, cost and energy (§7). | new script | D10 |
| **M12** | **`scripts/pod/run_all.sh`**: every phase in order on the pod, resumable, with a status table at the end (§8). | new script | one command for the whole pod session |

(M9/M10, the cuQRTM and Rotax harnesses, are dropped with those codes.)

**Status (2026-09-25): M1–M8, M11 and M12 are implemented** and checked locally as
far as possible without a GPU: build, CPU runs, CSV upgrade, extraction on sample
exports, and the Devito script on CPU. The first pod run is the first real GPU test
of the new scripts.

---

## 4. Part A — Per-kernel profiling (the CUDA ladder v0…v4)

### 4.1 Which kernels exist, and which ones "the kernel" means

Kernel names as they appear in Nsight (namespace `rtm::`):

| engine | per forward step | per backward step | on stored backward steps (every 10th) |
|---|---|---|---|
| cuda-v0 | `k_fd_time_step`, `k_inject_source`, `k_sponge` | `k_fd_time_step`, `k_inject_receivers`, `k_sponge` | + `k_imaging` |
| cuda-v1 | `k_fd_time_step_v1`, `k_inject_source` | `k_fd_time_step_v1`, `k_inject_receivers` | + `k_imaging` |
| cuda-v2 | same as v1 | same as v1 | `k_fd_time_step_v2_image` replaces the stencil, + `k_image_unique_receivers` |
| cuda-v3 | `k_fd_time_step_v3`, `k_inject_source` | `k_fd_time_step_v3`, `k_inject_receivers` | `k_fd_time_step_v3_image` + `k_image_unique_receivers` |
| cuda-v4 | `k_fd_time_step_v4`, … | `k_fd_time_step_v4`, … | `k_fd_time_step_v4_image` + … |

Two levels of comparison, both required:

* **Level 1, "per time step" (Nsight Systems).** Sum of *all* kernels in one step.
  This is the fair comparison, because v0 → v1 wins by *removing* a kernel.
* **Level 2, "main kernel" (Nsight Compute).** The plain stencil kernel of each
  version (`k_fd_time_step`, `_v1`, `_v1` for v2, `_v3`, `_v4`), plus the imaging
  variant (`_v2_image`, `_v3_image`, `_v4_image`) as a second row. For v0, also
  profile `k_sponge` and `k_imaging`, so the table shows what fusion removed.

### 4.2 Nsight Systems — the command

The profiling run uses the fixed dataset, **all 12 shots**, and the same flags as
`bench.sh`. The profiler adds little overhead, so this timeline is the real one.
The script (M3) runs:

```bash
nsys profile \
  --output results/profiles/marmousi12/nsys/cuda-v1 --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage true \
  --gpu-metrics-devices all \
  --stats false \
  ./build/rtm --engine cuda-v1 <fixed dataset flags from dataset_marmousi.env> --quiet
```

(If your nsys version rejects `--gpu-metrics-devices`, run `nsys profile --help | grep gpu-metrics`; older
versions call it `--gpu-metrics-device`. If counters are blocked (§2.1), the script drops the flag.)

Then it exports the summary tables to CSV:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
  --format csv --output results/profiles/marmousi12/nsys/cuda-v1 \
  results/profiles/marmousi12/nsys/cuda-v1.nsys-rep
```

**What you run:**

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && scripts/pod/setup.sh"              # rebuilds with NVTX
ssh pod "cd /workspace/RTM && scripts/pod/profile_ladder.sh nsys" # ~2 min for all five
scripts/local/sync_from_pod.sh pod
```

### 4.3 Nsight Systems GUI — where to look

Open `results/profiles/marmousi12/nsys/cuda-v1.nsys-rep` (File → Open).

**Step 1: find your way around the timeline.** The left tree has these rows:

* `Processes → [pid] rtm → Threads → rtm` — the CPU thread. Under it:
  * **CUDA API** — every `cudaLaunchKernel` and `cudaMemcpy` call on the CPU side.
  * **NVTX** — our ranges from M1: `setup`, `shot 1`, `forward`, `backward`…
* `CUDA HW (… NVIDIA GeForce RTX 3090)` — what actually ran on the GPU:
  * **Kernels** — one box per kernel execution.
  * **Memory** — memcpy/memset boxes (H2D green, D2H red, D2D and memset in other colours).
  * **Memory usage** — the device-memory curve (from `--cuda-memory-usage`).
* `GPU Metrics` (only if counters were allowed) — **SM Active**, **SM Issue**,
  **DRAM Read/Write Bandwidth**, **PCIe** as percentage graphs.

**Step 2: whole-run view (screenshot for D1a).** Zoom out fully. You should see
12 pairs of `forward`/`backward` NVTX blocks, with the Kernels row solid under
them. Look for:

| What you see | What it means |
|---|---|
| A long gap before `shot 1` | CUDA context creation plus file reading (`t_io` in the CSV). Report it as setup; it does not scale with shots. |
| Kernels row solid at this zoom | GPU busy; no large idle phases. |
| Tall H2D blocks per shot on the Memory row | Trace upload. It should be tiny; confirm it. |

**Step 3: zoom to ~5 time steps (screenshot for D1b; this is the key picture).**
Inside one `forward` range, drag-select a region about 100 µs wide, then
right-click → *Zoom into Selection*. Repeat until individual kernels are visible.
Then:

* Count the kernels per step, and hover each one for name and duration.
* **Look at the gaps between kernels on the Kernels row.** Click the gap and read
  its width in the bottom bar.
* Compare the **CUDA API** row directly above: if the `cudaLaunchKernel` boxes
  are as wide as the GPU kernels, **the CPU can barely launch as fast as the GPU
  runs**. This is hypothesis H2. It is a blocking point that no kernel
  optimization fixes. The fix is CUDA Graphs or fewer, fused launches.
* Do the same inside a `backward` range, at a stored step (the one with the
  extra imaging kernel).

**Step 4: numbers (bottom pane, needed for D2).** In the drop-down at the top of
the bottom pane (default *Events View*), select **Stats System View**. The report
list on the left includes:

| Report | Read | Use |
|---|---|---|
| **CUDA GPU Kernel Summary** | per kernel name: *Time (%)*, *Total Time*, *Instances*, *Avg*, *Med* | Share of GPU time per kernel; divide Total by (steps × shots) for per-step cost (D2) |
| **CUDA API Summary** | `cudaLaunchKernel` total time and count | Launch overhead (H2) |
| **CUDA GPU MemOps Summary (by Time / by Size)** | H2D / D2H / memset totals | Transfer share (expected tiny) |
| **NVTX Range Summary** | `forward`, `backward` totals | Cross-check with the CSV `t_forward` / `t_backward` |

The same tables are already in the CSVs from §4.2; the GUI is only for looking.

**Step 5: automatic advice.** Bottom-pane drop-down → **Expert Systems View**.
Relevant rules: *GPU Gaps*, *GPU Low Utilization*, *CUDA Synchronous Memcpy*,
*Pageable memory*. Screenshot any that fire. A rule firing on "GPU gaps" is direct
evidence for H2.

**Step 6: memory (for D7 at this size).** Expand `CUDA HW → Memory usage`. Hover
the plateau and note the peak (should be ≈ snapshots 408 MiB + 5 wavefield/model
arrays + traces). Compare with the `device_mem_mib` column from M2.

**Step 7 (optional): jump to Nsight Compute.** Right-click a kernel → *Analyze the
Selected Kernel with NVIDIA Nsight Compute*. This only works if the GUI machine
can run the kernel, which yours cannot. Use §4.4 instead.

### 4.4 Nsight Compute — the command

ncu replays each profiled kernel many times (about 40 passes with `--set full`),
so it profiles a **few chosen launches**, not the whole run. Use one shot and
600 steps. After 200 warm-up launches the timing is stable, and the data values
do not change the cost of a stencil.

The script (M3) runs two captures per engine:

```bash
# (a) main stencil kernel, forward, 3 launches after 200 warm-up launches
ncu --set full --import-source yes \
    --kernel-name 'regex:(^|::)k_fd_time_step_v1(\(|$)' \
    --launch-skip 200 --launch-count 3 \
    --export results/profiles/marmousi12/ncu/cuda-v1_stencil --force-overwrite \
    ./build/rtm --engine cuda-v1 <fixed dataset flags> --nt 600 --max-shots 1 --quiet

# (b) the imaging kernel (k_imaging for v0/v1, k_fd_time_step_vN_image for v2-v4),
#     backward; plus k_sponge for v0 (skip 200). One capture per kernel.
ncu --set full --import-source yes \
    --kernel-name 'regex:(^|::)k_imaging(\(|$)' --launch-skip 5 --launch-count 3 \
    --export results/profiles/marmousi12/ncu/cuda-v1_imaging --force-overwrite \
    ./build/rtm ... (same)
```

Then it exports the raw metrics:
`ncu --import X.ncu-rep --page raw --csv > X.csv`.

Three flags matter for honest numbers:

* **`--import-source yes`**: embeds the `.cu` source in the report so the Source
  page works on your laptop. `-lineinfo` is already in `CMakeLists.txt`.
* **`--cache-control all` (the default)** flushes L1/L2 before each replay. That
  is right for comparing kernels with each other, but it **understates L2 reuse
  between consecutive time steps**, which matters here (H1). So the script adds a
  **second, light pass** with `--cache-control none --section SpeedOfLight
  --section MemoryWorkloadAnalysis` and saves it as `*_warm.ncu-rep`. Report both,
  labelled *cold* and *warm*.
* **`--clock-control base` (the default)** locks the GPU at base clock, so ncu
  durations are **longer than nsys durations**. Use ncu for *ratios and
  percentages*, and nsys for *time*. Do not mix them in one table column.

**What you run:**

```bash
ssh pod "cd /workspace/RTM && scripts/pod/profile_ladder.sh ncu"   # ~5-10 min for all five
scripts/local/sync_from_pod.sh pod
python3 scripts/profile_extract.py     # locally → kernel_metrics.csv + plots
```

### 4.5 Nsight Compute GUI — where to look

Open `results/profiles/marmousi12/ncu/cuda-v0_stencil.ncu-rep`.

**Orientation.** The top toolbar has a **Page** drop-down: *Summary · Details ·
Source · Context · Raw · Session*. Next to it, a **Result** drop-down selects which
of the 3 captured launches you are viewing (they should be nearly identical; if
not, profile more launches).

**Baseline comparison (the most useful feature for the ladder).**

1. Open `cuda-v0_stencil.ncu-rep`, go to the *Details* page, and click **Add
   Baseline** in the toolbar. v0 is now the reference.
2. File → Open `cuda-v1_stencil.ncu-rep` (and v3, v4). Every number on the
   *Details* page now shows a **diff against v0** as a coloured bar and a
   percentage.
3. Screenshot the *GPU Speed Of Light Throughput* section in this state. It shows
   the whole ladder's progress on one screen.

**Details page, section by section.** Each section has a triangle to expand it and
may show a rule result (⚠ with *Est. Speedup*). Read the rule text; it is
often the diagnosis.

| Section | Read | Tells you |
|---|---|---|
| **GPU Speed Of Light Throughput** | *Memory Throughput %*, *DRAM Throughput %*, *Compute (SM) Throughput %*, *Duration* | Which wall the kernel is against. Memory % ≫ Compute % ⇒ memory-bound. **Both low ⇒ latency-bound / not enough parallelism** (small grid, or stalls). |
| ↳ **Roofline chart** (inside SOL, or *GPU Speed Of Light Roofline Chart*) | Point position vs the DRAM slope and the FP32 ceiling | Arithmetic intensity and distance to the roof. Hover the point for exact FLOP/s and FLOP/byte. **Screenshot this for D4**. |
| **Memory Workload Analysis** | **Memory Chart** (diagram: kernel ↔ L1/shared ↔ L2 ↔ DRAM, with bytes and hit rates on each arrow); tables: *L1/TEX Hit Rate*, *L2 Hit Rate*, *Mem Busy*, *Max Bandwidth* | Where the bytes come from. **v3 should show traffic moving from L1 to Shared Memory.** Compare DRAM bytes per launch with the 24 B/pt estimate. |
| **Compute Workload Analysis** | *Executed IPC*, pipe utilization | Confirms the kernel is not compute-bound (FMA pipe low) |
| **Launch Statistics** | *Registers Per Thread*, *Static Shared Memory Per Block*, *Grid Size*, *Block Size*, *Waves Per SM* | The resource cost of each version (v3/v4 add shared memory, possibly registers). **Waves per SM < ~2 ⇒ the grid is too small for the GPU** (H3). |
| **Occupancy** | *Theoretical* vs *Achieved Occupancy*, *Block Limit Registers / Shared Mem / Warps* | What limits resident warps. The occupancy-vs-registers/shared-memory graphs show the headroom. |
| **Scheduler Statistics** | *Active Warps / Eligible Warps / Issued Warp per scheduler*, *No Eligible %* | High *No Eligible* ⇒ warps are waiting; the next section says on what. |
| **Warp State Statistics** | Bar chart of **stall reasons** (cycles per issued instruction) | **The blocking point (D5):** *Stall Long Scoreboard* = waiting on global/L2 memory → memory-bound. *Stall Barrier* = waiting at `__syncthreads()` (v3/v4). *Stall MIO Throttle / Short Scoreboard* = shared-memory or shuffle pressure (v3/v4). *Stall Wait / Not Selected* = fine. |
| **Source Counters** | *Branch Instructions*, *Branch Efficiency*, *Divergent branches*; **top 5 stall locations** (click → jumps to Source) | Divergence from boundary checks and the receiver skip in `_image` kernels |
| **Instruction Statistics** | *Executed Instructions*; SASS opcode mix | v3/v4 execute **more instructions per point** (clamped loads, halo logic, shuffles). This is the cost side of their trade. |

**Source page.** Set *View* to **Source and SASS**, and choose **Warp Stall
Sampling (All Samples)** as the navigation metric. The hottest `.cu` lines light
up. Expected hot lines: the `p_cur[i ± kx]` loads in the k-loop (v0/v1), the
`load_tile` halo branch and `__syncthreads()` (v3), and the `safe_load` of
`p_prev`/`vdt2`/`sponge` plus the shuffles (v4). Screenshot one per version for
the appendix.

**Raw page.** Every metric by its exact name, with a search box. Use it to check
any number in `kernel_metrics.csv`.

### 4.6 "The same metrics" — the table for every kernel (D3)

`profile_extract.py` pulls exactly these metrics from every ncu report (metric
names as they appear on the *Raw* page). This is your table from
`PresentaionRapportPlan.md`, made precise:

| Metric (report label) | ncu metric name(s) | GUI location | Formula / note |
|---|---|---|---|
| Kernel duration | `gpu__time_duration.sum` | SOL → Duration | at base clock (see §4.4) |
| Time per step, all kernels | — (nsys) | Stats → CUDA GPU Kernel Summary | Σ kernel time / (nt × 2 × shots) |
| DRAM throughput, % of peak | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | SOL | |
| DRAM bytes (read / write) | `dram__bytes_read.sum`, `dram__bytes_write.sum` | Memory Workload → Memory Chart | |
| **Effective bandwidth, GB/s** | (read + write bytes) / duration | computed | also vs the 843 GB/s probe |
| **Bytes per grid point** | (read + write bytes) / points updated | computed | the 24 B/pt vs 16 B/pt argument (§4.7) |
| SM (compute) throughput % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SOL | |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy | |
| Theoretical occupancy | `sm__maximum_warps_per_active_cycle_pct` | Occupancy | |
| Registers / thread | `launch__registers_per_thread` | Launch Statistics | |
| Shared memory / block | `launch__shared_mem_per_block_static` | Launch Statistics | 0 for v0–v2 |
| Waves per SM | `launch__waves_per_multiprocessor` | Launch Statistics | < 2 ⇒ grid too small |
| L1/TEX hit rate | `l1tex__t_sector_hit_rate.pct` | Memory Workload | |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | Memory Workload | report cold **and** warm |
| Global load efficiency | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Memory Workload tables | ideal = 4 sectors/request (coalesced 32-bit) |
| Shared bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Memory Workload → Shared Memory table | v3/v4 only |
| Warp execution efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` / 32 | Instruction / Scheduler stats | < 100 % ⇒ divergent or partially filled warps |
| Branch efficiency | `smsp__sass_average_branch_targets_threads_uniform.pct` | Source Counters | |
| Executed instructions / point | `smsp__inst_executed.sum` / points | Instruction Statistics | cost of v3/v4 extra logic |
| FP32 FLOPs | `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + …_fmul_… + 2×…_ffma_…` | Roofline (hover) | |
| **Arithmetic intensity** | FLOPs / DRAM bytes | Roofline | FLOP/byte |
| Top stall reason + share | `smsp__average_warp_latency_issue_stalled_*` (largest) | Warp State Statistics | the blocking point |
| GPU utilization (SM active over the run) | nsys GPU metrics *SM Active* | nsys timeline | needs counters |
| H2D / D2H time and bytes | — (nsys) | Stats → CUDA GPU MemOps Summary | |
| CPU launch time share | — (nsys) | Stats → CUDA API Summary (`cudaLaunchKernel`) | H2 |

The resulting table has **rows = kernels** (v0 stencil, v0 sponge, v0 imaging,
v1, v2-image, v3, v3-image, v4, v4-image) and **columns = the metrics above**.
That one table satisfies "show the same metrics for each kernel".

### 4.7 From metrics to conclusions: optimization points vs blocking points

Use this decision table on each version:

| Observation (where) | Diagnosis | Optimization point it suggests |
|---|---|---|
| DRAM % ≥ ~75 %, Long Scoreboard dominant, AI far left on the roofline | **Memory-bandwidth-bound** (expected for v0–v2) | Reduce bytes per point: do not rewrite `p_cur` (apply the sponge only in the boundary strips), fold `vdt2·sponge` tables, compute the sponge from coordinates, FP16 storage for tables/snapshots (accuracy check needed). 24 → 16 B/pt ⇒ up to 1.5× |
| Two kernels each streaming the whole grid (nsys: `k_sponge` ≈ `k_fd_time_step` duration) | **Redundant grid pass** (v0) | Fusion (done in v1). Quantify: v0→v1 per-step time ratio vs byte ratio 36/24 = 1.5 |
| Gaps between kernels, `cudaLaunchKernel` ≈ kernel duration | **Launch-bound** (H2, small grids) | CUDA Graphs, or merge `k_inject_source` into the stencil kernel (no 1-thread launch) |
| L2 hit high (warm), DRAM % moderate, v3 not faster than v1 | **Caches already provide the reuse** (H1) | Shared memory is not worth it at this size; re-test at large grids (Part B) |
| Stall Barrier / MIO Throttle high, more instructions per point (v3/v4) | **Overhead of the optimization itself** | Blocking point of v3/v4: explains why they are slower. Possible fix: larger tiles or a z-sweep with a register queue |
| Waves per SM < 2, SM Active < 100 % in nsys | **Not enough work for 82 SMs** (H3) | Bigger problems per launch, or several shots concurrently in streams |
| Occupancy limited by registers/shared memory, and Long Scoreboard high | **Latency not hidden** | Reduce registers (`__launch_bounds__`), or smaller tiles |
| Low branch efficiency in `_image` kernels | Divergence from the receiver skip / interior test | Negligible if the kernel runs on 1 step in 10; check its time share in nsys first |

**Final Part A table (D6)** has one row per version with these columns: what
changed · measured gain (per step, nsys) · metric that proves why (ncu) ·
remaining blocking point · next optimization it points to.

---

## 5. Part B-1 — Scaling our engine: "how much more can we process"

GPU memory and GPU bandwidth limit the engine independently, and the two
Nsight tools measure them separately.

### 5.1 What fills device memory

From `rtm_cuda.cu` (setup + map_geometry + migrate):

```
bytes ≈ 5 × nxe·nze·4        (vdt2, sponge, p_prev, p_cur, p_next)   + nxe·nze (receiver mask, v2+)
      + nsnap × nx·nz·4      (snapshots)   ← dominates
      + 2 × nx·nz·4          (image, illumination)
      + nrec × nt × 4        (traces)
nsnap = (nt − 1) / store_interval + 1
```

The shot count does not matter: shots run one after another and reuse the buffers.

### 5.2 How the input is scaled up (applies to our engine AND Devito)

The same scaled datasets feed our engine (§5) and Devito (§6.2), so both codes
are always compared on identical inputs. Four knobs make the compute heavier.
Each is used for a specific purpose, and one is deliberately not used for stress:

| Knob | Effect on work | Effect on memory | Used for |
|---|---|---|---|
| **Finer grid (dx, dz ↓)** | ∝ 1/dx³ in 2D: points ∝ 1/dx², and CFL makes dt ∝ dx, so nt ∝ 1/dx. **Halving dx = 8× work** | ∝ 1/dx² (snapshots) | **Sweep S1**, the main sweep: capacity + efficiency |
| **Frequency f0 ↑, together with the grid** | none by itself | none | Keeps S1 physically honest (below) |
| **Stencil order ↑ (4 → 16)** | ~linear in order (more neighbours and FLOPs per point); bytes per point almost unchanged | halo only | **Sweep S2**: moves the kernel along the roofline |
| **Longer record (nt ↑)** | linear | snapshots linear | Not swept; policy B below covers the snapshot memory |
| **More shots** | linear total time, **same GPts/s** (shots are independent) | none (buffers reused) | **Only** the survey projection (§7). Not a stress knob. |
| **3D** | enormous | enormous | **Sweep S3**, optional and Devito-only (our engine is 2D) |

**Why f0 scales with the grid.** Refining the grid at fixed f0 = 10 Hz simulates
the same waves with more points than needed. That is valid as a compute test, but
a geophysicist would call it wasted effort. In production, a finer grid exists
because the frequency is higher, and a higher frequency means a sharper image. S1
therefore keeps the **points per wavelength constant**:

```
ppw = v_min / (2.5 · f0 · dx)       (2.5·f0 ≈ highest significant frequency of a Ricker)
today: water v = 1500 m/s, f0 = 10 Hz, dx = 12.5 m  →  ppw ≈ 4.8 in the water layer
       (the slowest cells of the model are 1028 m/s → ppw ≈ 3.3 there; the engine's
        startup check reports this stricter number)
S1 rule: f0 = 10 Hz × (12.5 m / dx)  →  both numbers stay the same at every size
```

The C++ engine prints ppw (and CFL) at the start of every run, so each dataset's
log shows it. The report sentence this enables: *"same accuracy, 10× the resolution, and here
is what it costs on each code"*.

#### Sweep S1: grid + frequency (main sweep, both codes)

Datasets (M6): the same Marmousi2 model at 6 resolutions. Record length stays
2.8 s. `DT` shrinks with dx (CFL), so `NT` grows. f0 grows with 1/dx.

| dx (m) | f0 (Hz) | grid nx×nz | points (interior) | DT (ms) | NT | work vs 12.5 m | snapshots, policy A |
|---|---|---|---|---|---|---|---|
| 12.5 | 10 | 1361×281 | 0.38 M | 1.0 | 2800 | 1× | 0.43 GB |
| 6.25 | 20 | 2721×561 | 1.5 M | 0.5 | 5600 | 8× | 1.7 GB |
| 5.0 | 25 | 3401×701 | 2.4 M | 0.4 | 7000 | ~16× | 2.7 GB |
| 3.75 | 33.3 | 4534×934 | 4.2 M | 0.3 | 9334 | ~37× | 4.8 GB |
| 2.5 | 50 | 6801×1401 | 9.5 M | 0.2 | 14000 | 125× | 10.7 GB |
| 1.25 | 100 | 13601×2801 | 38 M | 0.1 | 28000 | 1000× | **42.8 GB → OOM on 24 GB** |

The shot files are modelled **once, with our GPU engine** (`rtm_synth --engine
cuda-v1`), at the scaled f0. Both codes read the same files. Modelling at 1.25 m on
CPU would take days.

A fixed-f0 variant (f0 = 10 Hz at every dx) costs exactly the same compute. It is
not run by default. Add it only if you want a "pure compute scaling" line with no
physics change.

#### Sweep S2: stencil order (roofline movement)

Fixed grid **dx = 2.5 m** (large enough to fill the GPU, so throughput is on its
plateau, and 10.7 GB fits comfortably). Two shots (throughput does not depend on
the shot count). Orders:

| order | our engine | Devito |
|---|---|---|
| 4, 8, 12 | yes (`--order`; the engine supports 2–12, `MAX_HALF = 6`) | yes (`--order` → `space_order`) |
| 16 | no | yes |

The shots stay the order-8 files from S1. Migrating with a different order than the
one used for modelling is fine for a throughput study. The existing dt stays below
the CFL limit up to order 16, because the stability sum barely grows with order.
The C++ engine checks this at startup anyway. For Devito, M8 adds the same check.

What to look for: FLOPs per point grow with order while DRAM bytes per point stay
almost flat, so **arithmetic intensity rises and the roofline point moves right**.
The question for the report is whether Devito's generated stencil (and ours at 12)
leaves the bandwidth roof and becomes compute-limited. This produces figure **D11**
(roofline with one point per order, per code).

#### Sweep S3 (optional, Devito-only): 3D

Real surveys are 3D, and in Devito 3D is a few-line change to the script. Our
engine is 2D, so this has **no comparison partner**: label it clearly as "Devito
alone". Input: Marmousi2 at 12.5 m extruded along y (ny chosen so the run fills
the GPU). In 3D the saved wavefield explodes, so this also shows why 3D RTM in
industry needs snapshot compression or checkpointing. **Decision pending (you):**
in or out of the report.

#### Snapshot policies (applied on top of S1)

Two snapshot policies:

* **Policy A, same physics:** one snapshot every 10 ms regardless of dx
  (`store_interval` = 10 ms / DT, so nsnap ≈ 281 is constant). Memory grows with
  grid points. **This finds the ceiling.**
* **Policy B, trade accuracy for capacity:** at 2.5 m and 1.25 m, snapshots every
  5, 20, 40 and 80 ms. The denser 5 ms spacing (~80 GiB at 1.25 m) keeps an
  out-of-memory ceiling on 48 GB cards such as the A40, where 10 ms at 1.25 m fits. At 2.5 m policy A still fits, so `rtm_compare` against the
  policy-A image measures the accuracy cost (chart D7b). At 1.25 m it shows which
  spacing makes the largest grid fit. **This shows how far the ceiling moves, and
  what it costs.**

Each scaled dataset has **2 shots**, because GPts/s does not depend on the shot
count, and those two shots keep the 1.25 m runs to minutes.

The CPU reference is far too slow at fine grids, so the gate at each size is
**cuda-v3 vs cuda-v1** (both are run anyway; the ladder guarantees they are
bit-identical, and both were proven against the CPU at 12.5 m).

### 5.3 What you run

```bash
ssh pod "cd /workspace/RTM && for d in 10 5 4 3 2 1; do scripts/pod/make_scaled_dataset.sh \$d; done"   # shots modelled on the GPU
ssh pod "cd /workspace/RTM && scripts/pod/capacity_sweep.sh cuda-v1 cuda-v3"   # v3: does shared memory win at large grids? (H1/H3)
ssh pod "cd /workspace/RTM && scripts/pod/order_sweep.sh cuda-v1"              # S2: orders 4, 8, 12 at dx = 2.5 m
scripts/local/sync_from_pod.sh pod
python3 scripts/profile_extract.py
```

(`run_all.sh` phases 6–8 run exactly these; §8.) Estimated GPU time on a 3090:
about 45 min for both engines, most of it the 1.25 m policy-B runs (2 shots ×
~90 s each). The order sweep takes about 5 min.

### 5.4 Where to look (per sweep point)

* **Nsight Systems:** `CUDA HW → Memory usage` row, peak value. It must match the
  `device_mem_mib` column. At the OOM point there is no report; the CSV records
  `oom` and the requested size.
* **Nsight Systems, Stats → CUDA GPU Kernel Summary:** the stencil's *Time %*
  should rise toward ~100 % as the grid grows (launch overhead fades → H2 confirmed
  if it was < ~90 % at 12.5 m).
* **Nsight Compute, SOL + Launch Statistics:** DRAM % and *Waves per SM* at each
  size. Memory Workload: L2 hit rate should **drop** once the working set exceeds
  6 MB, which is where v3 may start to beat v1.

### 5.5 Charts

* **D7 Capacity:** x = grid points, y = device memory (GB). One line per snapshot
  policy, a horizontal 24 GB line, and the largest fitting grid marked. Second
  panel: image error (L2rel vs policy A) against snapshot spacing.
* **D8 Efficiency:** x = grid points (log), left y = GPts/s, right y = DRAM % of
  peak. One line per engine (v1, v3). Expected shape: rising while the GPU is
  under-filled, then a plateau near the bandwidth roof. The plateau is **the
  processing rate of one GPU**; any drop at the largest sizes needs an
  explanation (L2 or TLB effects).

---

## 6. Part B-2 — Devito

cuQRTM and Rotax are **out of scope** (decided 2026-09-25). Devito is the only
external code in the comparison.

### 6.1 Fairness rules (write these in the report)

1. **Same input:** Devito reads exactly the files our engine reads: the scaled
   Marmousi datasets from M6, with the same nt/dt, f0, shots and receiver line.
2. **Same GPU** for every GPU number, in the same pod session.
3. **Normalize.** Report **GPts/s** (extended points × steps × 2 / time),
   **s per shot**, **peak device memory**, **largest grid that fits**, and, for the
   main kernel, **% of DRAM peak** and **arithmetic intensity**.
4. **Exclude one-off costs.** Devito's JIT compilation is forced before the timed
   loop and reported in its own column (`jit_s`). Context creation and file I/O
   are outside the timed region for both codes.
5. **State the differences in the same table:** absorbing-boundary implementation,
   generated vs hand-written kernels, and how the source wavefield is stored (see
   "saved wavefield" below).

### 6.2 Devito on the GPU

**Why the GPU backend is required.** Devito on CPU (OpenMP, runbook 15) is
invisible to Nsight Compute. For Nsight, Devito must generate GPU code: OpenACC
through the NVIDIA HPC SDK compiler `nvc`. `scripts/pod/setup_devito_gpu.sh`
installs it from NVIDIA's apt repository (**about 10 GB in `/opt`, on the
container disk**, so create the pod with a container disk of at least 30 GB).
It then writes `scripts/pod/devito_gpu.env`:

```bash
export DEVITO_PLATFORM=nvidiaX DEVITO_LANGUAGE=openacc DEVITO_ARCH=nvc DEVITO_LOGGING=PERF
```

and runs a 1-shot GPU smoke test on the synthetic model.

**What the Devito sweep does per point** (`scripts/pod/devito_sweep.sh grid|order`):

1. A **timed run** with all shots of the dataset and default profiling, so the
   timing is clean. `nvidia-smi` logs memory and power.
2. If it fitted, an **Nsight Systems run** of 1 shot with `--nvtx` (ranges `shot N`,
   `devito forward`, `devito backward`), with `DEVITO_PROFILING=advanced` so Devito
   also reports its own GPts/s, GFlops/s and operational intensity. JIT compilation
   happens before the first range, so it appears as a CPU-only stretch at the
   start of the timeline. Filter it out by selecting the NVTX ranges.
3. An **Nsight Compute run** (600 steps, 1 shot) of the kernel with the most
   GPU time in (2). That is Devito's stencil; its name is picked automatically from
   the nsys Kernel Summary.
4. One row in `results/scaling.csv` (`code = devito`), plus the image compared with
   our cuda-v1 image of the same dataset in `results/compare_scaling.csv`
   (a similarity score, not a gate).

**Saved wavefield (fairness note for D7).** Devito saves the forward wavefield on the
*extended* grid (including the 50-cell sponge), while our engine saves only the
interior. The script prints the ratio at startup. It is **1.46× at 12.5 m but only
about 1.09× at 2.5 m**, because the sponge stays 50 cells while the grid grows. So
the handicap fades exactly where capacity matters. Report the ratio next to the
capacity chart rather than changing Devito's code.

**Where to look in the GUI (Devito-specific):**
* nsys → Stats → **CUDA GPU Kernel Summary**: the kernel names are generated
  (from the operator names `Fwd` / `Adj` and source lines). The stencil is the one
  with about nt instances and most of the time.
* nsys → **Memory** row during `devito forward`: **large D2H copies during the
  forward pass mean Devito streams the saved wavefield to host memory.** That trades
  bandwidth for capacity: Devito can then run grids that do not fit in device
  memory. On the capacity chart, its line may continue past our out-of-memory
  point, slower but still running. Report that if it happens.
* nsys → **OpenACC** rows (`--trace openacc`): compute and data regions.
* ncu → same sections as §4.5. Compare Devito's stencil with our v1 on **DRAM %,
  bytes/point and AI**. Does its separate sponge equation cost an extra pass, as in
  our v0?

**What to expect:**
* **Capacity (D7):** Devito's diamonds sit above our points at small grids
  (extended-grid save) and converge at large grids.
* **Efficiency (D8):** both curves should plateau. Compare the plateau GPts/s and
  the DRAM % at the plateau.
* **Order (D11):** does Devito's arithmetic intensity grow faster than ours with
  order? Its code generator can factor terms and reuse values across iterations,
  which our hand-written loop does not.

### 6.3 Comparison table (D9)

| code | device | order | wavefield storage | peak mem @12.5 m | largest grid on the GPU | GPts/s @12.5 m | GPts/s plateau | DRAM % (main kernel) | AI | s/shot @2.5 m | JIT (s) | image corr. vs cuda-v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours cuda-v1 | | 8 | interior snapshots on device | | | | | | | | — | 1.0 |
| ours cuda-v3 | | 8 | same | | | | | | | | — | |
| Devito GPU | | 8 | extended-grid snapshots (device or streamed) | | | | | | | | | |

All numbers come from `results/scaling.csv`, `results/profiles/kernel_metrics.csv`
and `results/compare_scaling.csv`.

---

## 7. Part B-3 — "How much can seismic processing be accelerated" (D10)

`scripts/project_survey.py` (M11) projects the measured numbers to a realistic
survey. The formulas belong in the report:

```
work per shot       W  = nxe·nze · nt · 2                       (point-updates)
time per shot       t  = measured s/shot at that dx (GPU)   |   W / measured CPU GPts/s (CPU)
survey time         T  = N_shots · t / N_devices               (shots are independent → linear)
cost                C  = T · price_per_device_hour             (only if you pass a price)
energy              E  = T · mean power · N_devices            (GPU: nvidia-smi; CPU: node power, default 280 W)
max grid per device    = from the capacity curve (D7)
```

`run_all.sh` produces it for **1000 shots at 6.25 m and at 2.5 m**:
`results/survey_projection_dx6.25.md`, `results/survey_projection_dx2.5.md` and the
matching `D10_survey_projection_dx*.png`. For cost columns, re-run locally with your
pod's price:

```bash
python3 scripts/project_survey.py --survey-shots 1000 --dx 2.5 --gpu-price-per-hour 0.44 --gpus 1 8
```

The CPU line uses `cpu` and `cpu-opt` from `results/benchmarks.csv` (12.5 m),
extrapolated by work. The assumption is printed under the table: linear shot
scaling, identical parameters, no I/O bottleneck, and CPU throughput independent
of grid size.

---

## 8. Running everything at once

### 8.1 Pod requirements

| Setting | Value | Why |
|---|---|---|
| Image | CUDA 12.x/13.x **devel** (has `nvcc`, `nsys`, `ncu`) | build + profilers |
| GPU | one card; the charts are labelled with its name | 24 GB gives the capacity ceiling of §5.2 |
| Container disk | **≥ 30 GB** | NVIDIA HPC SDK for Devito (~10 GB in /opt) |
| `/workspace` volume | **≥ 20 GB** | scaled datasets (~1.5 GB), images (~2 GB), reports |
| RAM | ≥ 32 GB | Devito at 1.25 m may stream its saved wavefield to host memory |
| GPU counters | test in phase 1 | without them the ncu phases are skipped (§2.1) |

### 8.2 Commands

On the laptop, once per session:

```bash
cd ~/RTM/RTM-Hardware-Acceleration
scripts/local/sync_to_pod.sh pod
```

On the pod, detached so a dropped SSH connection does not stop it:

```bash
ssh pod "cd /workspace/RTM && nohup scripts/pod/run_all.sh > run_all.out 2>&1 &"
ssh pod "tail -f /workspace/RTM/run_all.out"        # watch; Ctrl-C only stops watching
```

The 12 phases (`scripts/pod/run_all.sh --list`):

| # | phase | what | time (RTX 3090 class) |
|---|---|---|---|
| 1 | preflight | tool versions → `results/profiles/TOOLS.txt`, rebuild, GPU-counter test | 2 min |
| 2 | setup | `setup.sh`: dataset checksum, smoke tests, bandwidth probe, CPU reference | 30 min first time, then 1 min |
| 3 | bench | `bench_all.sh`: timing + gate of every engine on marmousi12 | 5 min |
| 4 | nsys_ladder | Part A timelines, v0…v4 | 3 min |
| 5 | ncu_ladder | Part A kernel reports, v0…v4 | 10 min |
| 6 | datasets | 6 scaled datasets, shots modelled on the GPU | 10 min |
| 7 | capacity | S1 + policy B, cuda-v1 and cuda-v3, with nsys + ncu per size | 45 min |
| 8 | orders | S2, orders 4/8/12, cuda-v1 | 5 min |
| 9 | devito_setup | NVIDIA HPC SDK + Devito venv + GPU smoke test | 15–20 min |
| 10 | devito_grid | Devito on S1 | 30–60 min |
| 11 | devito_order | Devito on S2, orders 4/8/12/16 | 15 min |
| 12 | report | `profile_extract.py` + `project_survey.py` | 1 min |

Total: about 3–4 hours. Useful options:

* `--skip-devito`: phases 9–11 skipped (for example if the HPC SDK will not install).
* **Resuming:** each finished phase leaves `results/profiles/.phases/<name>.done`.
  After a crash or pod restart, run the same command again and finished phases are
  skipped.
* `--only ncu_ladder` or `--from 7`: rerun one phase, or everything from a phase.

A failed phase is recorded and the script continues; only a failed build stops it.
The last lines are a status table, one line per phase.

### 8.3 Before terminating the pod

```bash
scripts/local/sync_from_pod.sh pod
```

It pulls `results/profiles/` (all `.nsys-rep` / `.ncu-rep` reports plus their CSV
and text exports), the CSVs, the plots and the projections. It then compares
checksums of every pod file with the local copy. **Terminate only after it prints
`SAFE TO TERMINATE THE POD`.** Also note the `nsys` and `ncu` versions from
`results/profiles/TOOLS.txt`: your Windows GUIs must be at least those versions.

### 8.4 After the pod: where everything is

| What | Where |
|---|---|
| Timelines (Nsight Systems GUI) | `results/profiles/<dataset>/nsys/*.nsys-rep` |
| Kernel reports (Nsight Compute GUI) | `results/profiles/<dataset>/ncu/*.ncu-rep` |
| D3 metrics table, D2 per-step table | `results/profiles/kernel_metrics.csv`, `step_breakdown.csv`, `SUMMARY.md` |
| Every sweep run | `results/scaling.csv`, `results/compare_scaling.csv` |
| Charts D2, D3, D4, D7, D7b, D8, D10, D11 | `results/plots/profiling/` |
| Per-run logs (if something looks wrong) | `results/profiles/<dataset>/runs/`, `run_all.out` on the pod |

`<dataset>` is `marmousi12` for Part A and `marmousi_dx<dx>` for the sweeps.
Recreating the charts locally: `myVenv/bin/python scripts/profile_extract.py`.

**What to send me:** the phase summary table from the end of `run_all.out`, and
tell me when the sync is done. I read `results/profiles/**` (CSV and text exports)
directly.

---

## 9. If something goes wrong

| Symptom | Cause | Do |
|---|---|---|
| `ERR_NVGPUCTRPERM` | Counters blocked on the host | §2.1: nsys-only on this pod, ncu elsewhere |
| ncu run takes forever | Too many launches matched | Check `--kernel-name` regex and `--launch-count`; keep `--nt 600 --max-shots 1` |
| ncu shows "==PROF== No kernels were profiled" | Regex does not match; the namespace is part of the name | Run `nsys` first and copy the exact name from the Kernel Summary; try `--kernel-name-base demangled` |
| GUI: "report version newer than this tool" | Laptop GUI older than pod CLI | Update the GUI (§2.2) |
| Source page empty / "source not available" | Report made without `--import-source yes`, or build without `-lineinfo` | Rerun with the flag; confirm `-lineinfo` in the compile line (`make VERBOSE=1`) |
| nsys durations ≠ ncu durations | ncu locks base clocks and flushes caches | Expected (§4.4). Use nsys for time, ncu for ratios |
| nsys report huge / slow to open | 12 shots × ~4 kernels × 5600 steps ≈ 270 k events | Fine for the GUI; if needed, profile `--max-shots 2` for timeline screenshots |
| `run_all.sh` stopped (pod restart, SSH drop without nohup) | — | Run the same command again: finished phases are skipped (§8.2) |
| Phase `devito_setup` failed | HPC SDK download/install, disk full, or driver too old for its CUDA | Check `df -h /opt`; rerun `--only devito_setup`; or run the rest with `--skip-devito` |
| Devito: no kernels in nsys | Still running on CPU | Check `DEVITO_PLATFORM`/`DEVITO_LANGUAGE` are exported in the same shell; Devito logs the target at operator build time |
| OOM at a sweep point | Expected at the ceiling | It is a data point: the script logs requested bytes and moves on |
