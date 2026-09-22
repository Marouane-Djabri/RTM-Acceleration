# Runbook 12 — `cuda-v2`, imaging fused into the backward stencil

Prerequisite: runbook 11 done (`cuda-v1` already PASS on the pod, ideally
with `L2rel` confirmed to exactly match `cuda-v0`'s).

**Higher risk than the previous two rungs — read this before running it.**
`cuda-v1` was a straightforward reordering (same operations, different
kernel boundary). `cuda-v2` is structurally different: it splits imaging
into two kernels per stored step (`k_fd_time_step_v2_image` for every
interior point except this step's receiver locations, `k_image_unique_receivers`
for exactly those, after injection lands) to avoid re-reading the whole
backward wavefield. The full correctness argument — including why a naive
"same thread also does image += f*b" fusion is wrong, and why the receiver
correction has to be a from-scratch recompute rather than an additive
correction — is in the header comment of `src/cuda/kernels_v2_imaging.cu`.
This was reasoned through carefully but **could not be tested on real
hardware** while writing it (this dev machine has no GPU). Treat the first
run of this rung as a real test of that reasoning, not a formality.

What it is: identical to `cuda-v1`, except on stored backward steps the
standalone `k_imaging` kernel is replaced by `k_fd_time_step_v2_image` (does
the stencil+sponge+imaging in one pass, skipping receiver-touched points)
followed by `k_image_unique_receivers` (a tiny kernel, one thread per unique
receiver-touched interior point this shot, run right after injection).
Non-stored steps and the forward driver are untouched — literally the same
`k_fd_time_step_v1` call as `cuda-v1`.

Expected to be **bit-identical** to `cuda-v1` (and therefore to `cuda-v0`).

Expected time: build 1 min, run **close to `cuda-v1`**, with a bit more of
an edge at small `--store-interval` (more stored steps → more of the removed
read actually gets removed) and almost no difference at large `--store-interval`.

## Commands

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && cmake --build build -j && ./build/rtm --list-engines"
#   expect: cpu cpu-opt cuda-v0 cuda-v1 cuda-v2
ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cuda-v2"
ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
scripts/local/sync_from_pod.sh pod
```

## What to expect

`bench.sh`'s summary table: the `cuda-v2` row's `L2rel` should be the exact
same number as `cuda-v1`'s row (and `cuda-v0`'s). `vs cpu` should be equal to
or a little better than `cuda-v1`.

## Send me

* the summary table printed by `bench.sh`,
* `results/benchmarks.csv`, `results/compare.csv`.

## If it fails

| symptom | do |
|---|---|
| `L2rel` differs from `cuda-v1`'s | **expected failure mode to watch for.** Send me both numbers, plus `--order`/`--nb`/`--store-interval` used. If it's small but nonzero and scales with the number of receivers, the likely culprit is the deduplication in `map_geometry` (two receivers clamped to the same point) or an off-by-one in the interior-bounds check in `k_fd_time_step_v2_image`. |
| `illegal memory access` / crash | `compute-sanitizer --tool memcheck ./build/rtm --engine cuda-v2 ... --max-shots 1 --nt 50`; also try `--tool racecheck` — `d_image_`/`d_illum_` are now written by two different kernels per stored step (`k_fd_time_step_v2_image` and `k_image_unique_receivers`) and I want to be sure their index sets are genuinely disjoint on real data, not just in my reasoning. Send me the full output. |
| image visibly wrong specifically near receiver lines/surface | this is the signature of a receiver-imaging bug (the two kernels' point sets overlapping, or `k_image_unique_receivers` reading before injection actually completed) — send me a PNG (`scripts/raw_to_png.py`) of both `cuda-v1` and `cuda-v2` images side by side. |
| runs but not faster than `cuda-v1` | expected at large `--store-interval` (few stored steps ⇒ the removed read barely matters); send me `--store-interval` used and both `Total`/`imaging` times from the benchmark report. |
