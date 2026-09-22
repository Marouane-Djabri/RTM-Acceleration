# Runbook 14 — `cuda-v4`, warp-shuffle z-neighbours

Prerequisite: runbook 13 done (`cuda-v3` already PASS on the pod).

**Highest-risk rung so far — treat a PASS here as informative, not assumed.**
`docs/CUDA_PLAN.md` itself only says "try after V3, measure" for this rung,
because a register-queue in the classic 3D-finite-difference sense doesn't
map cleanly onto this 2D (x,z) problem — there's no natural third dimension
to sweep. `src/cuda/kernels_v4_regqueue.cu`'s header explains the specific,
verifiable reading this implementation takes instead: since the block is
32 threads along z (matching `warpSize` exactly) times 8 along x, each fixed
-x row of 32 threads *is* one warp, so a thread's z-neighbours are literally
register values held by other lanes of its own warp — fetched with
`__shfl_up/down_sync` instead of a shared-memory read. x-neighbours still
come from `cuda-v3`'s shared tile.

Two extra hazards on top of v3's (both handled — see the file header for
the argument, and note one was actually caught and fixed while writing this,
not just reasoned around):
1. `__shfl_*_sync` needs every lane in its mask to actually execute that
   call, so (like `__syncthreads()` in v3) no thread can return early before
   it — the validity guard is at the very end of the kernel here, after
   every shuffle, unlike v0-v3 where it's at the top.
2. Because of (1), `vdt2`/`sponge`/`p_prev` are read through the same
   clamped `safe_load()` v3's tile uses, for every thread — not through the
   raw flat index, which can be arbitrarily out of the allocated array for
   a boundary thread (this is exactly the hazard v3's header derives).

What it is: same arithmetic, same order of operations as v3 (this was
checked explicitly: an early draft accidentally split the x-term and
z-term into two separate `lap +=` statements instead of one combined
addition, which is *not* bit-identical to the rest of the ladder — float
addition isn't associative — and was caught and fixed before this ever
reached the build).

Expected to be **bit-identical** to `cuda-v3`.

Expected time: build 1 min, run **possibly a little faster than v3, possibly
about the same** — this is the "measure it" rung.

## Commands

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && cmake --build build -j && ./build/rtm --list-engines"
#   expect: cpu cpu-opt cuda-v0 cuda-v1 cuda-v2 cuda-v3 cuda-v4
ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cuda-v4"
ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
scripts/local/sync_from_pod.sh pod
```

Same tiny-grid smoke test as v3 first, strongly recommended here:

```bash
ssh pod "cd /workspace/RTM && ./build/rtm_synth --outdir /tmp/tiny --nx 41 --nz 31 --dx 10 --dz 10 --nt 100 --dt 0.001 --f0 12 --nshots 1"
ssh pod "cd /workspace/RTM && compute-sanitizer --tool memcheck ./build/rtm --engine cuda-v4 --velocity /tmp/tiny/velocity.bin --shots /tmp/tiny/shots.bin --output /tmp/tiny_v4.bin --order 8 --nb 20 --f0 12 --max-shots 1"
ssh pod "cd /workspace/RTM && compute-sanitizer --tool racecheck ./build/rtm --engine cuda-v4 --velocity /tmp/tiny/velocity.bin --shots /tmp/tiny/shots.bin --output /tmp/tiny_v4.bin --order 8 --nb 20 --f0 12 --max-shots 1"
```

## What to expect

`bench.sh`'s summary table: the `cuda-v4` row's `L2rel` should be the exact
same number as `cuda-v3`'s. `vs cpu` — genuinely uncertain, report it either
way (faster, slower, or a wash are all useful data points here).

## Send me

* the summary table printed by `bench.sh`,
* `results/benchmarks.csv`, `results/compare.csv`,
* both `compute-sanitizer` outputs from the tiny grid.

## If it fails

| symptom | do |
|---|---|
| hang / kernel never returns | a divergent `__shfl_*_sync` or `__syncthreads()` — send me `--nx --nz --nb --order` used. |
| `illegal memory access` | `compute-sanitizer --tool memcheck` on the tiny grid; send me the full output. |
| `L2rel` differs from `cuda-v3`'s | this is the one I'd genuinely want to dig into rather than guess about — send me both numbers, `--order`/`--nb`, and if possible dump a small `--nt 2 --order 2 --max-shots 1` case (the smallest one that still reproduces it, per `docs/CUDA_PLAN.md` §4.2) so I can hand-check it against v3. |
| runs but not faster than v3 (or slower) | this is a legitimate outcome the plan itself anticipated ("try... measure"). If so, `cuda-v3` is the rung to actually use going forward — send me both `Total` times and I'll note v4 as "measured, not adopted" in `docs/RESULTS.md` rather than push further on it. |
