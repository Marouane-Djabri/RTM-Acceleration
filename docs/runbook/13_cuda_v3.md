# Runbook 13 — `cuda-v3`, shared-memory stencil tile

Prerequisite: runbook 12 done (`cuda-v2` already PASS on the pod, ideally
with `L2rel` confirmed to exactly match `cuda-v1`/`cuda-v0`).

**Different risk profile than v1/v2 — read this first.** v1/v2 were about
*reordering* or *splitting* operations; this rung is about *where a value
physically comes from* (shared memory instead of global memory), which
should be the safer kind of change bit-identity-wise — but it introduces
`__syncthreads()`, and a boundary case that's easy to get wrong: threads
launched past the end of a grid dimension that doesn't divide evenly by the
block size can have coordinates genuinely outside the allocated array (not
just outside the region the stencil writes), and every thread in a block
must reach the same `__syncthreads()` call or the kernel hangs/misbehaves.
The full argument for how this is handled (every load clamped into the
array bounds, no thread returns early before the barrier) is in the header
comment of `src/cuda/kernels_v3_shared.cu`. This could not be exercised on
real hardware while writing it.

What it is: identical arithmetic to `cuda-v2` — same k-loop, same order of
adds/multiplies — but the stencil's Laplacian now reads `p_cur` from a
`(32+2·half) x (8+2·half)` shared-memory tile loaded once per block, instead
of re-reading it from global memory `1 + 2·order` times per point. Both
`cuda-v2` kernels get a shared-memory twin (`k_fd_time_step_v3` /
`k_fd_time_step_v3_image`); the receiver-skip-and-finish-later imaging split
from v2 is unchanged.

Expected to be **bit-identical** to `cuda-v2`.

Expected time: build 1 min, run **noticeably faster** than v2 on the
stencil specifically (the plan's estimate is 1.5–2.5x on the stencil alone,
less on the total since sponge/injection/imaging traffic doesn't change).

## Commands

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && cmake --build build -j && ./build/rtm --list-engines"
#   expect: cpu cpu-opt cuda-v0 cuda-v1 cuda-v2 cuda-v3 cuda-v4
ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cuda-v3"
ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
scripts/local/sync_from_pod.sh pod
```

**Before the full Marmousi run**, it's worth a quick smoke test on the
smallest possible grid — that's where a boundary/clamping bug in the tile
loader would most likely show up as a crash rather than a subtle wrong
pixel, and it's fast to iterate on:

```bash
ssh pod "cd /workspace/RTM && ./build/rtm_synth --outdir /tmp/tiny --nx 41 --nz 31 --dx 10 --dz 10 --nt 100 --dt 0.001 --f0 12 --nshots 1"
ssh pod "cd /workspace/RTM && compute-sanitizer --tool memcheck ./build/rtm --engine cuda-v3 --velocity /tmp/tiny/velocity.bin --shots /tmp/tiny/shots.bin --output /tmp/tiny_v3.bin --order 8 --nb 20 --f0 12 --max-shots 1"
```

## What to expect

`bench.sh`'s summary table: the `cuda-v3` row's `L2rel` should be the exact
same number as `cuda-v2`'s. `vs cpu` should be visibly better than `cuda-v2`.

## Send me

* the summary table printed by `bench.sh`,
* `results/benchmarks.csv`, `results/compare.csv`,
* the tiny-grid `compute-sanitizer` output either way (clean or not).

## If it fails

| symptom | do |
|---|---|
| hang / kernel never returns | classic `__syncthreads()` divergence symptom — send me the exact `--nx --nz --nb --order` used; I need to re-check the block/grid math for that specific combination. |
| `illegal memory access` | `compute-sanitizer --tool memcheck` on the tiny grid above; send me the full output, it will point at the exact line. |
| `L2rel` differs from `cuda-v2`'s | send me both numbers plus `--order`/`--nb`; also try `--tool racecheck` — a race between the tile load and a read of it would produce inconsistent-but-not-crashing results. |
| runs, gate passes, but not faster than v2 | possible on a card with a very large L2 cache where v2's redundant global reads were already mostly cache hits; send me the GPU name and both `Total` times. |
