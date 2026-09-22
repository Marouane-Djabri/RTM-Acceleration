# Runbook 11 — `cuda-v1`, sponge fused into the stencil kernel

Prerequisite: runbook 00 done (reference image frozen, `cuda-v0` already PASS
on the pod from step 4/6 of `setup.sh`).

What it is: identical to `cuda-v0` except the Cerjan sponge multiply
(`k_sponge`) is folded into the stencil kernel (`k_fd_time_step_v1`) instead
of being its own kernel launch — two full wavefield read+write passes per
time step removed. No change to the forward/backward driver structure, the
snapshot handling, or imaging; only the stencil+sponge kernel differs.
Source: `src/cuda/kernels_v1_fused.cu` (has the full bit-identity argument in
its header comment), dispatch in `src/cuda/propagation_cuda.cu`.

Expected to be **bit-identical** to `cuda-v0`, so its `L2rel` vs the CPU
reference should print the *exact same number* `cuda-v0` did in step 4/6 —
if `cuda-v0` had FMA-induced rounding (a small nonzero `L2rel`, still a
PASS), `cuda-v1` should show that identical number, not a different one.

Expected time: build 1 min, run **similar to or slightly faster than
`cuda-v0`** (removes 2 of the ~4-5 kernel launches per step, all still
memory-bound).

## Commands

```bash
scripts/local/sync_to_pod.sh pod
ssh pod "cd /workspace/RTM && cmake --build build -j && ./build/rtm --list-engines"
#   expect: cpu cpu-opt cuda-v0 cuda-v1
ssh pod "cd /workspace/RTM && scripts/pod/bench.sh cuda-v1"
ssh pod "cd /workspace/RTM && scripts/pod/plot.sh"
scripts/local/sync_from_pod.sh pod
```

## What to expect

`bench.sh` ends with the summary table. The `cuda-v1` row's `L2rel` should
match `cuda-v0`'s row exactly (same bit pattern in, same bit pattern out —
see the bit-identity argument in `kernels_v1_fused.cu`). Its `vs cpu` speedup
should be equal to or a bit higher than `cuda-v0`'s.

## Send me

* the summary table printed by `bench.sh`,
* `results/benchmarks.csv`, `results/compare.csv`.

## If it fails

| symptom | do |
|---|---|
| `L2rel` differs from `cuda-v0`'s (even slightly) | this means the bit-identity argument in `kernels_v1_fused.cu`'s header comment has a hole — send me both `L2rel` numbers and the exact `--order`/`--nb` used; do **not** treat a "close but different" number as a pass here, unlike `cpu-opt` this rung claims exact bit-identity, not just L2rel-close. |
| `illegal memory access` / crash | `compute-sanitizer --tool memcheck ./build/rtm --engine cuda-v1 ... --max-shots 1 --nt 50`, send me the output. |
| runs but not faster than `cuda-v0` | expected at high `store_interval` (sponge removal is a small fraction of total traffic there); send me both `Total` times and `--store-interval` used. |
