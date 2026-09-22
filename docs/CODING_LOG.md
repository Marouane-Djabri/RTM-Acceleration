# Coding Log

Running log of implementation work on this repo, newest entry first. Each
entry: what was done, why, what was verified locally (no GPU on this
machine — see `docs/CUDA_PLAN.md` §0), and what still needs a pod run.

---

## 2026-09-21 — `cuda-v3` and `cuda-v4`: shared-memory tiling and warp-shuffle z-neighbours (Phase 2, rungs G3-G4)

**Context.** Finishes the CUDA ladder's core rungs (`cuda-v1`..`cuda-v4`,
matching `src/rtm/factory.cpp`'s own "Phase 2" comment) at the user's
request ("implement the rest of the kernels"). These two are pure memory
-hierarchy optimizations — no change to *what* is computed, only *where
each value is read from* — which should in principle be the safer kind of
change to reason about than v1/v2's reordering. In practice, getting there
required catching real bugs along the way; recorded below because they're
the kind of mistake worth remembering the shape of.

**`cuda-v3` — shared-memory stencil tile.** Each block loads its
`(32+2·half) x (8+2·half)` tile of `p_cur` into `__shared__` once;
`k_fd_time_step_v3` / `k_fd_time_step_v3_image` are shared-memory twins of
v2's two kernels, same arithmetic, same operation order, just reading the
Laplacian's neighbours from the tile instead of re-fetching from global
memory `1+2·order` times per point.

**The boundary bug this rung has to avoid (worth writing down precisely,
since it's an easy one to miss):** v0-v2's kernels compute their flat index
and immediately `if (out of range) return;` before touching memory — safe,
because those kernels never call a block-wide collective operation, so
threads are free to take different paths. `cuda-v3` calls
`__syncthreads()` after loading the tile, and *every* thread in the block
must reach it. The subtlety is that "out of range" threads aren't just
outside the compute-valid region — when a grid dimension doesn't divide
evenly by the block size, a thread's `(ix,iz)` can be arbitrarily far past
the *end of the allocated array itself* (derived precisely in
`kernels_v3_shared.cu`'s header: the excess is bounded by the block size
minus one, not by `half`, so it's easily tens of elements past the buffer
for a small `half`). The fix: every thread participates in the tile load
unconditionally, but every global read goes through a `safe_load()` that
clamps to `[0,nxe_dev) x [0,nze_dev)` first — never reads outside the
buffer regardless of how far a thread's true coordinates have strayed —
and only *after* `__syncthreads()` does a thread check its own validity
and return. Clamped loads are never actually consumed by a valid thread's
computation (a valid thread's true neighbours are always themselves in
bounds), so clamping never substitutes a wrong value for a real one.

**`cuda-v4` — warp shuffle for z-neighbours.** Block layout is unchanged
from v3 (32 threads along z, 8 along x). Because `blockDim.x == 32 ==
warpSize`, each fixed-x row of 32 threads is exactly one warp — so a
thread's z-neighbours (needed at `iz±k`) are literally register values held
by other lanes of its own warp, fetchable with `__shfl_up/down_sync`
instead of a shared-memory read. x-neighbours still come from the v3 tile.
This is a deliberately narrow, verifiable reading of `docs/CUDA_PLAN.md`'s
terse "register queue along z, x-neighbours from shared memory" — a
faithful 3D-FD-style sweep doesn't have an obvious translation to a 2D
problem, and I'd rather implement something I can reason about rigorously
than guess at a bigger restructuring I can't verify at all.

**Two bugs caught while writing v4, before they ever reached a build —
worth recording since I can't rely on a GPU to catch them for me:**
1. First draft read `vdt2[i]` / `p_prev[i]` directly using the *unclamped*
   flat index inside the per-thread computation — exactly the hazard v3's
   `safe_load()` exists to prevent, reintroduced because v4's shuffle
   requirement forces the validity guard to the end of the kernel (shuffle,
   like `__syncthreads()`, requires every lane in its mask to actually
   reach the call — an early return before it is undefined behaviour, so
   the usual "check validity, return before touching memory" pattern
   doesn't work here). Fixed by routing every such read through
   `safe_load()`, mirroring v3.
2. First draft accumulated the Laplacian as two separate `lap += x_term;`
   then `lap += z_term;` statements. v0 through v3 all compute
   `lap += x_term + z_term;` as one grouped addition. Float addition isn't
   associative — `(lap+x)+z` is not guaranteed bit-identical to
   `lap+(x+z)` — so this would have silently broken bit-identity despite
   being "the same three numbers added together." Caught by re-deriving
   the exact op sequence against v3's code (not by running anything, since
   there's nothing to run here) and fixed to `lap += x_term + z_term;`,
   matching every other rung.

**What changed.**

* `include/rtm_cuda.hpp` — `CudaVariant::V3_Shared`, `V4_RegQueue`; new
  private helper `uses_receiver_marker()` (true for V2/V3/V4 — they all
  need the receiver-marker/dedup infrastructure v2 built) replacing three
  separate `variant_ == V2_Imaging` checks that would otherwise have needed
  updating by hand for each new rung.
* `src/cuda/kernels_v3_shared.cu`, `src/cuda/kernels_v4_regqueue.cu` (new)
  — the kernels described above, each with the correctness argument written
  into the file header next to the code it justifies (same convention as
  v1/v2's kernel files).
* `src/cuda/kernels_cuda.hpp` — declared the four new kernels.
* `src/cuda/rtm_cuda.cu` — `name()` reports v3/v4; the three places that
  used to check `variant_ == V2_Imaging` (buffer allocation, growth, rebuild
  in `setup()`/`map_geometry()`) now call `uses_receiver_marker()`.
* `src/cuda/propagation_cuda.cu` — replaced the growing if/else-if chain
  with two small local dispatch functions, `launch_plain_stencil` and
  `launch_imaging_stencil`, each a `switch` over `CudaVariant` — the
  forward and backward drivers just call one of these instead of branching
  inline, which is what made adding two more rungs to an already-nested
  conditional structure tractable to keep correct.
* `src/rtm/factory.cpp`, `CMakeLists.txt` — register `cuda-v3`, `cuda-v4`.
* `docs/runbook/13_cuda_v3.md`, `docs/runbook/14_cuda_v4.md` (new) — pod
  hand-offs, each including a tiny-grid `compute-sanitizer` smoke test
  *before* the full Marmousi run (boundary/shuffle bugs are far more likely
  to show up as a crash on a small grid than as a subtle wrong pixel on a
  big one), and v4's is explicit that "not faster than v3" is an
  acceptable, plan-anticipated outcome, not a failure to chase further.

**Verified locally (no GPU on this machine):**
- Clean `cmake --build build -j` succeeds for all five CUDA engines, no
  compiler warnings.
- `./build/rtm --list-engines` prints `cuda-v0` through `cuda-v4`; each of
  `cuda-v3`/`cuda-v4` fails cleanly with the same no-driver error as the
  others.
- `--engine cpu` still runs correctly.
- Manually re-derived the exact floating-point operation sequence in v3
  and v4 against v0-v2's, specifically checking associativity of the
  Laplacian's accumulation (this is what caught bug 2 above) and checked
  every shared-memory tile index and warp-shuffle lane-offset by hand
  against the block dimensions for both a small (`half=4`) and the largest
  supported (`half=6`, `MAX_HALF`) stencil order.

**Not yet verified (needs the pod, per the two new runbooks) — v4
especially should be treated as a real experiment, not a formality:**
- Bit-identity of v3 against v2, and v4 against v3, on real hardware.
- Whether v3's boundary-safety restructuring (clamped loads, deferred
  guard) actually behaves as reasoned once real out-of-range grid
  configurations exist — this is exactly the class of bug static reasoning
  is weakest against.
- Whether v4 is actually faster than v3 at all — the plan flags this as
  uncertain, and the runbook says explicitly to fall back to v3 as the
  adopted rung if it isn't, rather than iterating further on v4 blind.

---

## 2026-09-21 — `cuda-v2`: imaging fused into the backward stencil (Phase 2, rung G2)

**Context.** Follow-on to `cuda-v1` (previous entry below), continuing the
CUDA ladder one rung at a time at the user's direction ("implement the next
step"). This is the rung I'd flagged as genuinely tricky: naively fusing
`image += f*b` into the same thread that computes the stencil update breaks
bit-identity, because that thread runs *before* receiver injection, and
imaging needs the wavefield *after* injection.

**The design that makes it work.** Split imaging into two kernels on stored
backward steps only (non-stored steps and the whole forward driver are
untouched, still exactly `cuda-v1`):

1. `k_fd_time_step_v2_image` — stencil+sponge fused exactly like `cuda-v1`,
   and *also* does `image += f*b` / `illum += f*f` using its own
   just-computed value, but only for interior points that are **not** a
   receiver location this shot. Those points are never touched by injection
   (this step or any other), so the value the thread just computed already
   *is* the final post-injection value — reading it from a register is
   bit-identical to how the old standalone `k_imaging` read it from global
   memory one kernel later.
2. `k_inject_receivers` runs immediately after, unchanged.
3. `k_image_unique_receivers` (new, tiny) — one thread per **unique**
   receiver-touched interior point this shot (deduplicated on the host),
   run after injection + rotation, recomputing `image += f*b` from scratch
   with the true post-injection `b`.

Two failure modes I ruled out explicitly rather than by trial:
- **Additive correction instead of recompute.** I first considered letting
  step 1 image every point (including receivers) with the wrong
  pre-injection value, then having step 3 add a correction
  `f*injection_amount`. This is *not* bit-identical: `f*(a+b) != f*a + f*b`
  in IEEE-754 in general (different rounding sequence), so it would silently
  diverge from `k_imaging`'s single `f*b_final` multiply. Step 3 therefore
  always recomputes from scratch, never adds a delta.
- **Receiver-indexed instead of unique-point-indexed correction.** Two
  receivers can clamp to the same grid cell (the existing reason
  `k_inject_receivers` uses `atomicAdd`). `k_imaging` visits each *grid
  point* once regardless of how many receivers map to it, so a correction
  kernel indexed by receiver instead of by unique point would double-count
  the image contribution wherever two receivers coincide. `map_geometry`
  now sorts + deduplicates the extended-grid receiver indices per shot
  before uploading them, specifically to avoid this.

The full argument lives next to the code, in the header comment of
`src/cuda/kernels_v2_imaging.cu` — worth reading before touching that file.

**What changed.**

* `include/rtm_cuda.hpp` — added `CudaVariant::V2_Imaging`; new device state:
  `d_is_receiver_` (one byte per extended-grid point, rebuilt per shot),
  `d_unique_rec_ext_` + `nuniq_rec_` (deduplicated receiver indices, grown
  like `d_rec_index_`/`d_traces_`).
* `src/cuda/kernels_v2_imaging.cu` (new) — `k_mark_receivers`,
  `k_fd_time_step_v2_image`, `k_image_unique_receivers`, plus the full
  correctness argument in the file header.
* `src/cuda/kernels_cuda.hpp` — declared the three new kernels.
* `src/cuda/rtm_cuda.cu` — `name()` reports `cuda-v2`; `setup()` allocates
  `d_is_receiver_` (only for this variant); destructor frees the two new
  buffers; `map_geometry()` rebuilds the marker array and the deduplicated
  unique-receiver list every shot (only for this variant).
* `src/cuda/propagation_cuda.cu` — forward driver's variant check widened
  from `== V1_Fused` to `!= V0_Naive` (V2 uses the same plain fused stencil
  in forward, since forward never images). Backward driver: on a stored step
  with `V2_Imaging`, dispatches `k_fd_time_step_v2_image` instead of
  `k_fd_time_step_v1` and, after injection + rotation, launches
  `k_image_unique_receivers` instead of the standalone `k_imaging`; every
  other case (non-stored steps, other variants) is unchanged.
* `src/rtm/factory.cpp`, `CMakeLists.txt` — register `cuda-v2`.
* `docs/runbook/12_cuda_v2.md` (new) — pod hand-off, written with an
  explicit "this is higher-risk than the last rung" warning up front and a
  symptom table aimed at the specific failure modes this design could have
  (receiver double-imaging, reading before injection landed, index-math
  off-by-one).

**Verified locally (no GPU on this machine):**
- Clean `cmake --build build -j` succeeds, including device-link — the new
  file's `extern __constant__` references resolve, and `k_fd_time_step_v2_image`
  linking against kernels declared in a different translation unit works.
- `./build/rtm --list-engines` prints `cuda-v2`; `--engine cuda-v2` fails
  cleanly with the same no-driver error as the other CUDA engines.
- `--engine cpu` (via `rtm_synth`) still runs correctly; nothing CPU-side
  was touched.
- Manually traced the index math (`eidx`/`iidx` conventions, extended vs.
  interior bounds, the `ext / nze` / `ext % nze` decomposition in
  `k_image_unique_receivers`) against `include/rtm_types.hpp`'s actual
  formulas rather than assuming them.

**Not yet verified (needs the pod, per `docs/runbook/12_cuda_v2.md`) — and
this rung needs it more than the last one:**
- Bit-identity against `cuda-v1` on real data. This design has more moving
  parts than V1 (two kernels writing to `d_image_`/`d_illum_` per stored
  step instead of one, a host-side dedup step, a per-shot marker array) and
  I have no way to execute a single instruction of it here to check my own
  reasoning empirically. The runbook asks specifically for
  `compute-sanitizer --tool racecheck` on this rung, since the disjointness
  of the two kernels' write sets (receiver vs. non-receiver interior points)
  is exactly the kind of invariant that's easy to get subtly wrong and that
  racecheck would actually catch if violated.
- Whether the marker-array approach is actually worth its cost at realistic
  `nrec`/`nx*nz` ratios — the runbook says to expect a *smaller* win than
  V1 gave and to report the numbers either way.

---

## 2026-09-21 — `cuda-v1`: sponge fused into the stencil kernel (Phase 2, rung G1)

**Context.** Before this session, `docs/CUDA_PLAN.md` §2 (plumbing) and §3
(the `cuda-v0` naive-mirror engine) were already fully implemented and
committed — further along than the doc itself assumes, since it was written
to be followed step by step from an empty `src/cuda/`. This repo's own phase
numbering (`docs/OPTIMIZATION_PLAN.md` §4, and a pre-existing comment in
`src/rtm/factory.cpp`) calls the CUDA optimization ladder "Phase 2": a
sequence of `cuda-v1`..`cuda-v4` rungs, each gated bit-identical to the one
before it. This session implemented the first rung, `cuda-v1`, at the user's
direction (the full v1–v4 ladder was offered but the user asked for v1 only).

**What changed.**

* `include/rtm_cuda.hpp` — added `enum class CudaVariant { V0_Naive,
  V1_Fused }`; `CUDARTM` now takes a `CudaVariant` constructor argument
  (default `V0_Naive`, so nothing about `cuda-v0` changed) and `name()`
  reports the active variant.
* `src/cuda/kernels_v1_fused.cu` (new) — `k_fd_time_step_v1`: the same
  stencil computation as `k_fd_time_step`, but the thread that computes
  `p_next[i]` also does `p_cur[i] *= sponge[i]` and `p_next[i] *= sponge[i]`
  immediately, in registers, instead of a separate `k_sponge` kernel reading
  and writing the whole extended grid again. Removes two full read+write
  passes over the wavefield per time step, for both the forward and
  backward drivers.
* `src/cuda/propagation_cuda.cu` — `forward_propagation` / `backward_propagation`
  now branch on `variant_`: `V1_Fused` calls `k_fd_time_step_v1` and skips
  the separate `k_sponge` launch entirely; `V0_Naive` is untouched (same two
  kernels, same order as before).
* `src/cuda/kernels_cuda.hpp` — declared `k_fd_time_step_v1`.
* `src/cuda/rtm_cuda.cu` — added the `name()` definition (moved out of the
  header now that it depends on `variant_`).
* `src/rtm/factory.cpp` — registers `cuda-v1` (constructs `CUDARTM` with
  `CudaVariant::V1_Fused`); `--list-engines` now prints `cpu cpu-opt cuda-v0
  cuda-v1`.
* `CMakeLists.txt` — added `src/cuda/kernels_v1_fused.cu` to `rtm_core`'s
  CUDA sources.
* `docs/runbook/11_cuda_v1.md` (new) — the pod hand-off for this rung,
  matching the shape of `docs/runbook/01_cpu_opt.md`.

**Why this is bit-identical to `cuda-v0` (not just L2rel-close).** `cuda-v0`'s
per-step order is stencil → inject → sponge → rotate; the sponge kernel
damps every point of the extended grid, including the handful that
injection just touched. `cuda-v1` reorders this to stencil+sponge (fused) →
inject, i.e. damping now happens *before* injection instead of after. This
is safe only because every source/receiver index sits at extended-grid
distance exactly `nb` from the outer edge (`eidx(ix+nb, iz+nb)` with `ix` in
`[0,nx-1]`), which is precisely where `setup()`'s Cerjan taper evaluates to
`1.0f` exactly — and multiplying by `1.0f` in IEEE-754 is an exact no-op, so
"damp-then-add" and "add-then-damp" produce the identical bit pattern at
every point injection ever touches. At every other point, the single
multiply by `sponge[i]` is the same float operation regardless of which
kernel performs it. `cpu_opt.cpp` (Phase 1) relies on the same fact for its
own fused sponge (see its `fused_time_step_impl` comment) — this is not a
new trick, just its CUDA counterpart. The full argument, kept next to the
code it justifies, is in the header comment of `kernels_v1_fused.cu`.

I considered also fusing the imaging accumulation into the same kernel (the
plan's V2 rung) but did not — that fusion is **not** safe by the same
argument: imaging needs the wavefield *after* receiver injection, but the
stencil+sponge kernel runs *before* injection, so fusing imaging into it
would read pre-injection values at receiver locations and silently corrupt
the image there. `cpu_opt.cpp`'s backward driver hit exactly this and
deliberately kept imaging as a separate pass for that reason
(`src/cpu_opt/cpu_opt.cpp:254-258`); the same reasoning blocks a naive V2
fusion in CUDA. V2 was out of scope for this session.

**Verified locally (no GPU on this machine):**
- `cmake --build build -j` succeeds, including the CUDA device-link step —
  confirms the `extern __constant__` cross-translation-unit reference from
  `kernels_v1_fused.cu` into the symbols `propagation_cuda.cu` uploads
  resolves correctly under `CUDA_SEPARABLE_COMPILATION`.
- `./build/rtm --list-engines` prints `cuda-v1`.
- `./build/rtm --engine cuda-v1 ...` fails cleanly with the expected
  "CUDA driver version is insufficient" error (same as `cuda-v0`) — proves
  the new engine is wired into `setup()` correctly and doesn't crash before
  reaching the GPU check.
- `./build/rtm_synth --engine cpu ...` still runs correctly — the CPU
  reference path was not touched by any of this.

**Not yet verified (needs the pod, per `docs/runbook/11_cuda_v1.md`):**
- That `cuda-v1`'s image is actually bit-identical to `cuda-v0`'s (same
  `L2rel` vs the CPU reference) when run for real. The reasoning above is
  sound but the CUDA_PLAN.md validation ladder (§4) exists precisely
  because reasoning alone isn't the gate — an empirical PASS on hardware is.
- `results/ref/` (the frozen reference images from `CUDA_PLAN.md` §1) still
  doesn't exist in this repo; no rung, including the already-committed
  `cuda-v0`, has been gated on a GPU yet as far as this repo's history shows.
