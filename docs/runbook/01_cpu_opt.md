# Runbook 01 — `cpu-opt`, the optimized CPU bar

Prerequisite: runbook 00 done (reference image exists on the pod).

What it is: `CPUReferenceRTM`'s algorithm with OpenMP over all cores, the
sponge fused into the stencil sweep (two full wavefield passes per step
removed), and the inner depth loop auto-vectorized (AVX2 on the pod's CPU).
Built so that its image is **bit-identical** to the reference — the gate should
print `L2rel = 0`, not just PASS. Source: `src/cpu_opt/`.

Expected time: build 1 min, run **2–5 min** (depends on the pod's core count).

## Commands

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && cmake --build build -j && ./build/rtm --list-engines"
#   expect: cpu cpu-opt cuda-v0
ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cpu-opt"
ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
scripts/local/sync_from_pod.sh pod
```

## What to expect

`bench.sh` ends with the summary table. The `cpu-opt` row should read
`vs cpu` ≈ 5–25× (scales with the pod's cores), `L2rel 0.000e+00`, `PASS`.

Optional, if you want the thread-scaling curve for the write-up (each run is
a new row, ~1–5 min):

```bash
ssh pod "cd /workspace/RTM && for t in 1 2 4 8; do OMP_NUM_THREADS=\$t scripts/pod/bench.sh cpu-opt; done"
```

## Send me

* the summary table printed by `bench.sh`,
* `results/benchmarks.csv`, `results/compare.csv` (come back with `sync_from_pod.sh`).

## If it fails

| symptom | do |
|---|---|
| `L2rel` is small but not 0 (e.g. `3e-07`) | still PASS; send me the number — it means the pod's compiler contracted an FMA somewhere and I'll pin it down. |
| `L2rel > 1e-5` / FAIL | send me the full `rtm_compare` block. |
| runs but not faster than `cpu` | `ssh pod nproc` and send me the number; also `ssh pod "cd /workspace/RTM && OMP_NUM_THREADS=1 scripts/pod/bench.sh cpu-opt"`. |
