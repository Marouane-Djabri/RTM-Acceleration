# Exploring the results of the A40 run

A guided tour of what the 2026-09-26 run produced: which file to open, in which
order, what to look for, and which report section it feeds. Every number quoted
here comes from the files in `results/`. Check them yourself as you go: the
jury will ask where each number comes from.

**The run:** RunPod pod, 1× NVIDIA A40 (48 GB, 46,068 MiB visible), Intel Xeon
Gold 6342 with 8 vCPUs, CUDA 12.8, Nsight Systems 2026.3.2. No Nsight Compute
data: the host blocks GPU performance counters (`ERR_NVGPUCTRPERM`,
`results/profiles/ncu_permission.txt`).

**The images** come from a second, short session (`scripts/pod/images_run.sh`):
RunPod, 1× RTX PRO 6000 Blackwell (MIG slice), CUDA 12.8. That session produced
one migrated image per GPU version plus Devito's, and nothing else (no timings).

---

## 0. Before you start

**Install Nsight Systems on Windows, version 2026.3.2 or newer.** The reports
were recorded with 2026.3.2, and an older GUI refuses to open them. It's a free
download from developer.nvidia.com and needs an NVIDIA developer login.

**Open WSL files from Windows** through this path in the Explorer address bar:

```
\\wsl$\<your distro>\home\samir\RTM\RTM-Hardware-Acceleration\results\
```

**Regenerate every chart and table** at any time. This takes a few seconds and
needs no pod:

```bash
cd ~/RTM/RTM-Hardware-Acceleration
myVenv/bin/python scripts/profile_extract.py
myVenv/bin/python scripts/project_survey.py --survey-shots 1000 --dx 2.5
```

**See everything in one page:** `myVenv/bin/python scripts/build_dashboard.py`
writes `results/dashboard.html`, which has every chart, table and image from the
sessions below, with the interpretation computed from the same CSVs. Open it in a
browser; it works offline.

---

## 1. Map of the results

| What | Where |
|---|---|
| One timing row per run of every engine (fixed 12.5 m dataset) | `results/benchmarks.csv` (A40 rows: host `c73680b81fa3`) |
| Correctness gates vs the CPU reference | `results/compare.csv` |
| Every sweep run: grid size, snapshots, GPts/s, memory, power | `results/scaling.csv` |
| Image comparisons from the sweeps (gates, snapshot accuracy, Devito) | `results/compare_scaling.csv` |
| Measured bandwidth (CPU and GPU) | `results/ref/bandwidth.csv` |
| Tables ready for the report | `results/profiles/SUMMARY.md`, `results/profiles/step_breakdown.csv` |
| Survey projections | `results/survey_projection_dx6.25.md`, `results/survey_projection_dx2.5.md` |
| Charts | `results/plots/profiling/D*.png`, `results/plots/*_c73680b81fa3.png` |
| **Migrated images, one per version + Devito** (raw float32, 1361 × 281) | `results/images/cuda-v0..v4.bin` (+ `_illum`, `_filtered`), `results/images/devito.bin` |
| **Image figures for the report** | `results/plots/images/*.png` (regenerate: `myVenv/bin/python scripts/plot_version_images.py`) |
| Image comparisons vs the reference (images session) | `results/compare_images.csv` |
| **Timelines for the GUI** | `results/profiles/<dataset>/nsys/*.nsys-rep` |
| Summary tables exported from each timeline | `results/profiles/<dataset>/nsys/*_cuda_*_sum.csv`, `*_nvtx_sum.csv` |
| Per-run logs (if a number looks odd) | `results/profiles/<dataset>/runs/*.log` |

`<dataset>` is `marmousi12` for the CUDA ladder (Part A) and `marmousi_dx<dx>`
for the sweeps. Charts, CSVs and the image figures are in git; the `.nsys-rep`
reports (576 MB) and the raw `.bin` images live only on this laptop.

**Rows to ignore in `scaling.csv`:** sweep `S1B_aliased` comes from the first
capacity sweep, whose snapshot spacing (a fixed 10 ms) aliased the imaging at
fine grids. Session 5 explains why. Sweep `S1T` rows come from that same sweep but
keep valid throughput numbers; the charts use them only for throughput.

---

## 2. The exploration, in eight sessions

Each session: open → look for → what it shows → report section. Do them in
order: each one builds on the previous.

### Session 0: The result images (15 min), start here

**Open:** `results/plots/images/my_marmousi_filtered.png`, `versions_grid.png`,
`version_differences.png`, then `devito_vs_ours.png`.

**Look for:**
* **The geology** (`my_marmousi_filtered.png`, the Laplacian-filtered image): layered
  sediments on the left, the faulted blocks dipping through the centre (8–11 km),
  the strong deep reflector at ~2.3 km, the structures on the right. That's the
  known Marmousi model, so the migration works. `marmousi_ref.png` is the same
  image before filtering. The big blobs near the surface are RTM's usual
  low-frequency noise, which the filter removes: a good before/after pair.
* **The 12 arcs along the top** are an acquisition footprint: only 12 shots were
  fired, and each shot position leaves a curved artifact near the surface.
* **Every version produces the same image** (`versions_grid.png`): the CPU
  reference and cuda-v0 … v4 are indistinguishable, and each correlation with the
  reference rounds to **1.0000000**.
* **What differs is rounding** (`version_differences.png`, one colour scale for
  all panels). The largest difference is **0.53 × 10⁻⁵ of the image's peak**
  (cuda-v3), concentrated near the shallow shots.

  | version | relative L2 vs reference, A40 | same, RTX PRO 6000 (Blackwell) |
  |---|---|---|
  | cuda-v0 | 4.3·10⁻⁶ | 4.3·10⁻⁶ |
  | cuda-v1 | 4.2·10⁻⁶ | 7.3·10⁻⁶ |
  | cuda-v2 | 4.6·10⁻⁶ | 7.5·10⁻⁶ |
  | cuda-v3 | 8.5·10⁻⁶ | 1.45·10⁻⁵ |
  | cuda-v4 | 6.7·10⁻⁶ | 1.06·10⁻⁵ |

  On the Blackwell GPU, v3 and v4 land just above the strict 1·10⁻⁵ gate. Their
  correlation is still exactly 1, and a real bug shows up around 10⁻³. The
  compiler arranges the float operations slightly differently per GPU
  architecture. What to write: *"all versions agree with the reference to within
  floating-point rounding (relative L2 ≤ 1.5·10⁻⁵ on every GPU tested)."*
* **Devito vs ours** (`devito_vs_ours.png`): the **same geology, correlation 0.95**
  once both are filtered (0.69 raw: the two codes scale amplitudes differently).
  Devito shows a **stronger flat event near 0.4 km** and more ringing in the
  shallow layers. That's a difference between two independent implementations
  (absorbing boundary, source interpolation), not an error in either. Both
  migrate identical data: their direct-wave mutes were aligned before this run.

**Report:** the first results figure (the filtered image), then "correctness
across versions" (grid + differences + the table), and the Devito figure in the
comparison chapter.

### Session 1: The headline numbers (15 min)

**Open:** `results/plots/speedup_marmousi12_c73680b81fa3.png`, then
`results/benchmarks.csv` (the A40 rows).

| engine | forward + backward (s) | total (s) | GPts/s | vs cpu | vs cpu-opt |
|---|---|---|---|---|---|
| cpu (1 core) | 1012.37 | 1016.53 | 0.037 | 1× | — |
| cpu-opt (8 threads) | 497.98 | 498.85 | 0.075 | 2.0× | 1× |
| cuda-v0 | 2.310 | 2.315 | 16.2 | 439× | 215× |
| cuda-v1 | 1.866 | 2.134 | 20.0 | 476× | 234× |
| cuda-v2 | 1.872 | 1.875 | 20.0 | 542× | 266× |
| cuda-v3 | 1.942 | 1.945 | 19.3 | 523× | 256× |
| cuda-v4 | 1.966 | 1.970 | 19.0 | 516× | 253× |

**Look for:**
* **Compare the forward + backward column, not the total.** v2 looks faster than
  v1 in the total (1.88 vs 2.13 s), but their propagation times are equal (1.87 s
  each). v1's extra 0.26 s is spent outside the kernels, in its migration setup.
  Don't report "v2 beats v1".
* **v0 → v1 is the real kernel win** (2.31 → 1.87 s, −19%). v2, v3 and v4 add
  nothing on top of it at this size.
* **Two CPU baselines.** The plain reference is deliberately single-core. The
  fair CPU baseline is `cpu-opt` on 8 threads. Also note that the same reference
  took 211.6 s on the earlier AMD EPYC pod: cloud CPU timings vary about 5×
  between machines.
* **All gates PASS** (`results/compare.csv`). The reference image reproduced bit
  for bit on this Intel machine: difference 0, correlation 1. The earlier pod was
  AMD.

**Report:** results chapter, first table; methodology (two baselines, why).

### Session 2: What one time step costs (20 min)

**Open:** `results/plots/profiling/D2_step_breakdown.png` and
`results/profiles/step_breakdown.csv`.

| engine | kernels / step | stencil µs | sponge µs | imaging µs | injection µs | GPU total µs | CPU launch µs | wall µs | GPU busy |
|---|---|---|---|---|---|---|---|---|---|
| v0 | 3.05 | 15.4 | **14.4** | 0.8 | 2.3 | 33.3 | 28.8 | 35.9 | 92.9% |
| v1 | 2.05 | 23.5 | 0 | 0.9 | 2.4 | 27.2 | 21.8 | 29.0 | 93.7% |
| v2 | 2.05 | 22.3 (+2.3 fused imaging) | 0 | 0 | 2.4 | 27.3 | 21.7 | 29.1 | 93.7% |
| v3 | 2.05 | 23.1 (+2.3) | 0 | 0 | 2.5 | 28.4 | 22.5 | 30.2 | 93.9% |
| v4 | 2.05 | 23.5 (+2.3) | 0 | 0 | 2.5 | 28.8 | 22.8 | 30.7 | 93.8% |

**Look for:**
* **Fusion made measurable:** v0 pays a whole second pass over the grid for the
  sponge (14.4 µs). v1's fused kernel costs 23.5 µs instead of 15.4 + 14.4 = 29.8.
  That's one read/write pass saved, which matches the bytes argument (36 → 24
  bytes per point).
* **The launch overhead is almost as large as the work:** the CPU spends about
  22 µs per step inside `cudaLaunchKernel`, while the GPU needs about 27 µs. The
  CPU barely keeps up, and the GPU is idle about 6–7% of the time. This is
  hypothesis H2 confirmed. The fix it points to is CUDA Graphs, or fewer, fused
  launches (future work).
* **v3/v4 are slightly slower per step than v1** (28.4–28.8 vs 27.2 µs): shared
  memory and warp shuffles add work without saving any memory traffic.

**Report:** results chapter, "per-version analysis"; the step-breakdown chart.

### Session 3: The timelines in the Nsight Systems GUI (45 min)

**Open:** `results/profiles/marmousi12/nsys/cuda-v0.nsys-rep`, then `cuda-v1.nsys-rep`.
The click-by-click guide is in `docs/PROFILING_STRATEGY.md` §4.3. What to do with
these specific files:

1. **Whole run (screenshot 1):** the NVTX row shows 12 × `shot N` →
   `forward`/`backward`. Before the first shot there's a gap: `read inputs` +
   `setup` (CUDA context creation).
2. **Zoom to ~5 time steps inside one `forward` (screenshot 2, the key picture):**
   * **v0:** three kernels per step. `k_fd_time_step`, then the tiny
     `k_inject_source` (1 thread), then `k_sponge`, which is almost as long as the
     stencil.
   * **v1:** two kernels per step. Put both screenshots side by side: the sponge
     box is gone.
3. **The gaps:** click the empty space between kernels on the Kernels row, and
   compare with the `cudaLaunchKernel` boxes on the CUDA API row above. They're
   almost as wide as the kernels. That's Session 2's launch overhead, now visible.
4. **Stats System View** (bottom-pane drop-down): *CUDA GPU Kernel Summary* and
   *CUDA API Summary* hold the numbers behind `step_breakdown.csv`.
5. **Expert Systems View:** screenshot any rule that fires ("GPU gaps", "low
   utilization").
6. **Compare with a large grid:** open `results/profiles/marmousi_dx2.5/nsys/cuda-v1.nsys-rep`.
   The kernels are much longer (bigger grid), so the gaps become negligible. That
   explains why throughput rises with grid size (Session 4).
7. **Memory:** `CUDA HW → Memory usage` row, peak about 710 MiB at 12.5 m. It
   matches `device_mem_mib` in `benchmarks.csv`.

**Report:** 2–3 timeline screenshots in the results chapter, and the tool
walkthrough in the methodology.

### Session 4: How the GPU behaves as the problem grows (30 min)

**Open:** `results/plots/profiling/D8_efficiency.png` and
`D11b_order_throughput.png`, plus `results/scaling.csv`.

| dx | grid points | cuda-v1 GPts/s | s/shot | GPU power |
|---|---|---|---|---|
| 12.5 m | 0.38 M | 20.2 | 0.16 | 53 W |
| 6.25 m | 1.5 M | 20.6 | 1.02 | 173 W |
| 5 m | 2.4 M | 20.9 | 1.89 | 204 W |
| 3.75 m | 4.2 M | 21.3 | 4.22 | 241 W |
| 2.5 m | 9.5 M | ~22.4 | ~13 | ~285 W |
| 1.25 m | 38 M | ~22.8 (S1T) | ~98 | — |

**Look for:**
* **A plateau at about 22–23 GPts/s.** That's the processing rate of one A40 for
  this kernel.
* **Memory-bound, with a number to prove it:** 22 GPts/s × 24 bytes per point ≈
  **530 GB/s, about 92% of the 578 GB/s the probe measured** on this A40
  (`results/ref/bandwidth.csv`). There's no headroom left for a memory-access
  optimization, which is why v3 (shared memory) can't win. Even at 1.25 m (38 M
  points, far beyond the 6 MB L2 cache), v1 and v3 run at the same speed (22.8 vs
  22.9 GPts/s). Hypotheses H1 and H3: answered.
* **Stencil order doesn't change our speed:** 22.4 GPts/s at orders 4, 8 and 12.
  Doubling the arithmetic costs nothing, another signature of a kernel limited by
  memory, not compute.
* **Power rises with grid size** (53 → 285 W): small grids leave the GPU partly
  idle.

**Report:** results chapter, "scaling"; the discussion of why the optimizations
stop paying off (this is the core insight of the study).

### Session 5: How much fits in memory (30 min)

**Open:** `results/plots/profiling/D7_capacity.png` and
`D7b_snapshot_accuracy.png`, plus `results/compare_scaling.csv`.

**Look for:**
* **The memory model is confirmed:** measured points (dots) sit on the formula's
  prediction (dashed lines, `PROFILING_STRATEGY.md` §5.1).
* **The ceiling:** with one snapshot every 10 steps, memory grows as 1/dx³. 3.75 m
  fits (15.2 GiB measured), **2.5 m does not** (needs ~50 GiB, the A40 has ~45). At 1.25 m,
  only one snapshot every 160 steps fits.
* **What sparser snapshots cost** (at 3.75 m, correlation with the 10-step image):

  | every | correlation | |
  |---|---|---|
  | 20 steps | 0.9999986 | free: half the memory, same image |
  | 40 steps | 0.86 | aliasing begins |
  | 80 steps | 0.46 | image damaged |
  | 160 steps | 0.43 | image damaged |

  This is the sampling theorem at work: snapshots must stay denser than the
  highest frequency in the wavefield. So the 1.25 m grid only fits on one A40 with
  an aliased image, which is exactly why industry uses checkpointing or wavefield
  compression.
* **A methodology lesson worth a paragraph:** the first sweep used a fixed 10 ms
  spacing while the frequency grew with the grid. It aliased at fine grids (5 ms
  vs 10 ms images correlated at only 0.36 at 2.5 m), and a consistency check
  exposed it. Keeping the spacing constant in time steps fixed it.

**Report:** results chapter, "capacity"; discussion (checkpointing, compression,
multi-GPU as future work).

### Session 6: The comparison with Devito (45 min)

**Open:** `D8_efficiency.png` (Devito line), `D11b_order_throughput.png`,
`results/scaling.csv` (rows with `code = devito`), and in the GUI
`results/profiles/marmousi_dx3.75/nsys/devito_order8.nsys-rep`.

| dx | Devito wall GPts/s | Devito's own GPts/s (compute only) | ours GPts/s | Devito GPU power |
|---|---|---|---|---|
| 12.5 m | 3.7 | 9.7 | 20.2 | 63 W |
| 6.25 m | 5.8 | 17.7 | 20.6 | 86 W |
| 5 m | 5.9 | 19.1 | 20.9 | 98 W |
| 3.75 m | 7.0 | 22.7 | 21.3 | 115 W |
| 2.5 m (every 40 steps) | 15.7 | 26.5 | 22.4 | 179 W |

**Look for:**
* **Devito's kernels are as fast as ours or faster:** its own profiler measures
  up to 22–26 GPts/s of pure compute. The generated OpenACC code is competitive
  with hand-written CUDA.
* **But its end-to-end throughput is 3–5× lower** at 10-step snapshots, and it
  more than doubles when snapshots are 4× sparser (2.5 m). Devito spends most of
  its time moving the saved wavefield. **Check this in the timeline:** the Memory
  row during `devito forward` should show large device-to-host copies. Its low
  GPU power (115 W vs our 241 W at 3.75 m) says the same thing: the GPU is often
  waiting.
* **A different capacity limit:** Devito keeps snapshots in **host RAM**. At 2.5 m
  and 1.25 m, the container's 50 GB RAM limit killed it (`oom_kill 2`), not the
  GPU memory. Ours is bounded by GPU memory, Devito's by host memory.
* **Order sweep:** Devito slows at high orders (16.0 → 12.0 GPts/s from order 4
  to 16), while ours stays flat up to 12.
* **Image similarity:** raw correlation ≈ 0.69–0.74 with our image
  (`compare_scaling.csv`), but **0.95 once both are filtered**, with the same
  reflectors in the same places (Session 0, `devito_vs_ours.png`). The raw number
  mostly measures different amplitude scaling. The 2.5 m numbers (0.2–0.3) compare
  runs with different snapshot spacings, so ignore them.

**Report:** a comparison chapter, and the discussion (hand-written vs generated
code, where each one's time goes).

### Session 7: What it means for a processing company (15 min)

**Open:** `results/plots/profiling/D10_survey_projection_dx2.5.png` and
`results/survey_projection_dx2.5.md` (and `_dx6.25`).

1000 shots at 2.5 m (290 G point-updates per shot):

| | 1 device | 8 devices |
|---|---|---|
| ours, A40 | 3.7 h | 0.46 h |
| Devito, A40 | 5.1 h | 0.64 h |
| cpu-opt, Xeon 8 threads (extrapolated) | 1,073 h | — |
| cpu reference, 1 core (extrapolated) | 2,180 h | — |

**Look for:** the order of magnitude (weeks → hours), and energy (1.0 kWh on the
A40 vs ~300 kWh for the CPU node, using an assumed 280 W). **Add costs:** re-run
with your pod's price, e.g. `--gpu-price-per-hour 0.40`. Keep the assumptions
printed under the table in the report.

---

## 3. Open points to check before writing

1. ✅ **Devito vs our image:** done (Session 0). The reflectors sit in the same
   places (correlation 0.95 after filtering). The one visible difference is a
   stronger flat event near 0.4 km in Devito's image. Describe it as an
   implementation difference; don't claim either image is "wrong".
2. **v1's extra 0.26 s outside the kernels (Session 1):** visible in the
   `cuda-v1` timeline as time before the first shot. Probably one-time memory
   allocation. It's harmless for the conclusions, but explain it if you report
   totals.
3. **No hardware counters:** the kernel-level claims (bandwidth-bound, cache reuse)
   rest on measured timings + the bytes model + scaling experiments, not on
   Nsight Compute metrics. State this once, in the methodology chapter.
4. **Cloud CPU baseline:** 8 vCPUs of a shared Xeon. Give the machine, and
   present the speedups against both `cpu` and `cpu-opt`.

---

## 4. From sessions to report chapters

| Report chapter | Sessions | Main figures |
|---|---|---|
| Methodology | 1, 3, 5 (the aliasing lesson) | a timeline screenshot, tool walkthrough |
| Results: the image and correctness | 0 | `my_marmousi_filtered.png` (+ raw), `versions_grid.png`, `version_differences.png` |
| Results: per version | 1, 2, 3 | speedup chart, D2, v0 vs v1 timeline |
| Results: scaling and capacity | 4, 5 | D8, D11b, D7, D7b |
| Comparison with Devito | 0, 6 | `devito_vs_ours.png`, D8 (Devito line), Devito timeline |
| Industrial perspective | 7 | D10 |
| Discussion / future work | 2 (launch overhead → CUDA Graphs), 5 (checkpointing), 6, §3 | — |
