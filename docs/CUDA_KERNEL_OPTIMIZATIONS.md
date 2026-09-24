# CUDA Kernel Optimizations, Explained for a CUDA Student

This walks through the five stencil-kernel versions in `src/cuda/` (`v0` through
`v4`) and explains **what changed and why it's faster**, using plain CUDA
vocabulary — kernel launches, global vs. shared memory, warps, coalescing,
occupancy. No domain knowledge about the actual numerical problem is required:
just treat it as "a grid of floats, and each point's new value depends on a
small neighborhood of points around it" (a classic stencil, the same shape of
problem as a heat-diffusion or Conway's-Game-of-Life kernel).

Every version below produces **bit-identical output** to the one before it —
these are pure performance optimizations, not algorithm changes. That
constraint is actually a nice teaching device: it forces every optimization to
be about *how* the work gets done (memory access pattern, kernel count,
where a value physically lives), never *what* gets computed.

---

## V0 — the naive baseline

File: `propagation_cuda.cu` (`k_fd_time_step`), `k_sponge` in the same file,
`k_imaging` in `imaging_cuda.cu`.

This is the "textbook" version: one thread per grid point, each kernel does
exactly one job, everything reads and writes straight to global memory.

```cpp
__global__ void k_fd_time_step(...) {
    ...
    float lap = c0_dev * p_cur[i];
    for (int k = 1; k <= half_dev; ++k) {
        lap += cx_dev[k] * (p_cur[i + kx] + p_cur[i - kx])
             + cz_dev[k] * (p_cur[i + k ] + p_cur[i - k ]);
    }
    p_next[i] = 2.0f * p_cur[i] - p_prev[i] + vdt2[i] * lap;
}
```

Per time step, the driver launches **three separate kernels**:

1. `k_fd_time_step` — compute the new grid values.
2. `k_sponge` — a second full pass over *every* grid point to damp values
   near the boundary (multiply by a weight `w`).
3. `k_imaging` — (backward pass only) a third full pass that reads the
   result and accumulates it into an output array.

One thing V0 already gets right, worth calling out because every later
version depends on it: **`threadIdx.x` maps to the fastest-varying array
index** (`iz`), so consecutive threads in a warp read consecutive addresses
in memory — a coalesced access pattern. Every later kernel keeps this
layout. The optimizations from here on are entirely about (a) how many
times the grid gets swept per time step, and (b) where each float is read
from.

**The inefficiency:** three kernel launches per step means the grid gets
read and written **three separate times**, each one bound mostly by
global-memory bandwidth, not compute. Kernel-launch overhead also adds up,
and each kernel re-reads data the previous kernel just touched, so there's
no reuse of anything still "hot" from the last pass.

---

## V1 — kernel fusion (fewer launches, fewer memory passes)

File: `kernels_v1_fused.cu` (`k_fd_time_step_v1`).

**The optimization: fold `k_sponge`'s work into `k_fd_time_step` itself.**

```cpp
const float w    = sponge[i];
const float pn_i = 2.0f * pc_i - p_prev[i] + vdt2[i] * lap;

p_cur[i]  = pc_i * w;   // used to be its own kernel launch
p_next[i] = pn_i * w;
```

This is the most basic and often highest-leverage GPU optimization: **if two
kernels touch the same data back to back, merge them into one kernel** so
the data only has to be loaded from and stored to global memory once instead
of twice. The stencil kernel already has `pc_i` (== `p_cur[i]`) sitting in a
register and is about to compute `pn_i`; multiplying both by the sponge
weight costs a couple of extra register-resident FLOPs, essentially free
compared to launching a whole second kernel that re-reads/re-writes the
entire array from global memory.

Net effect: **3 kernels → 2 kernels** per step, and one less full
global-memory round trip. No change in arithmetic, no change in the shared
constants, no change in the block/grid layout — this is purely "do the
follow-up work while the value is still in a register instead of writing it
out and reading it back in."

*Takeaway for the course:* kernel fusion trades launch overhead and
redundant memory traffic for a small amount of extra register pressure /
instructions per thread. It's almost always a win when the fused work is
cheap and touches data you already have on hand.

---

## V2 — fusing a third kernel (with a correctness subtlety)

File: `kernels_v2_imaging.cu` (`k_fd_time_step_v2_image`,
`k_image_unique_receivers`, `k_mark_receivers`).

**The optimization: fuse `k_imaging`'s accumulation into the stencil kernel
too** — but only for the backward pass, and only where it's safe to do so
without changing the answer.

```cpp
p_cur[i]  = pc_i * w;
p_next[i] = pn_dmp;

if (/* boundary padding region */) return;
if (is_receiver[i]) return;   // handled separately, see below

image[interior_idx] += f * pn_dmp;   // uses the value already in a register
illum[interior_idx] += f * f;
```

This is the same idea as V1 (avoid a third full read/write pass over the
grid) but it hits a real ordering hazard, which is a good lesson in why
fusion isn't always a free lunch:

Between the stencil kernel and the imaging kernel, a *different* small
kernel (`k_inject_receivers`) writes small corrections into a handful of
grid points — it must run **after** the stencil and **before** imaging
reads the final values. If the stencil kernel tried to do the imaging
accumulation itself for *every* point in one shot, it would be reading the
pre-injection value for the handful of points that are about to be
corrected — silently wrong for those specific cells.

The fix is a **split**:

- The fused stencil kernel does the accumulation immediately for every
  point that injection will *never* touch (checked against a small
  precomputed `is_receiver` flag array — `k_mark_receivers` builds this
  once per shot).
- A second, tiny kernel (`k_image_unique_receivers`) — launched with one
  thread per *unique* affected point, not one thread per point in the
  whole grid — finishes the handful of points that injection does touch,
  reading their now-final value after injection has run.

Because the number of receiver points is tiny compared to the whole grid
(a sparse set vs. the full 2D array), that second kernel is launched with a
much smaller grid than a full-array kernel would need — it's cheap.

*Takeaway for the course:* fusing kernel A's work into kernel B is only
valid if nothing *between* A and B (in the original ordering) can still
change the values A reads. When something does, you don't abandon the
fusion — you fuse the *safe* subset and handle the *unsafe* subset with a
small, targeted kernel instead of a full-grid one.

---

## V3 — shared memory tiling (cutting redundant global loads)

File: `kernels_v3_shared.cu` (`k_fd_time_step_v3`, `k_fd_time_step_v3_image`).

**The optimization: classic shared-memory stencil tiling.** No new fusion
here — same two kernels as V2, same arithmetic — but now each thread block
first cooperatively loads its tile of the grid (plus a halo/apron of
neighboring cells) into `__shared__` memory once, and every thread's
Laplacian computation reads its neighbors from that on-chip shared tile
instead of going back to global memory for each one.

```cpp
__shared__ float tile[kTileX][kTileZ];
load_tile(tile, p_cur, iz, ix, tz, tx);   // whole block loads together

// every thread in this block now reuses `tile` instead of hitting
// global memory again for each of its `half`-radius neighbors
for (int k = 1; k <= half_dev; ++k) {
    lap += cx_dev[k] * (tile[tx + k][tz] + tile[tx - k][tz])
         + cz_dev[k] * (tile[tx][tz + k] + tile[tx][tz - k]);
}
```

Why this matters: in the naive version, a stencil with radius `half` means
every interior grid point gets read from global memory roughly `2*half + 1`
times total across all the threads whose neighborhoods overlap it (once by
its "owner" thread, and once more by each nearby thread that also needs it
as a neighbor). Shared memory lets the whole thread block load each value
from global memory **exactly once**, then every thread that needs it reads
it from the much faster on-chip shared memory instead.

Two things worth calling out as general CUDA lessons baked into this file:

1. **The apron/halo load.** The tile is bigger than the block
   (`blockDim + 2*half` in each direction) because threads near the edge of
   the block need neighbors that belong to the *next* block's territory.
   Only the threads nearest each edge do the extra halo loads
   (`if (threadIdx.x < half_dev) { ... }`), so the extra work is proportional
   to the tile's perimeter, not its area.
2. **Why every thread must participate in the load, even "invalid" ones.**
   `__syncthreads()` is a collective operation — every thread in the block
   must reach it, or you get undefined behavior. So unlike V0-V2 (where an
   out-of-range thread could `return` immediately), this kernel has every
   thread do the tile load first (using a *clamped* index so out-of-range
   threads read some harmless in-bounds value instead of crashing), hit the
   barrier together, and only check "am I a real point?" and bail out
   *after* the barrier. This is a very common shared-memory-kernel bug
   pattern to internalize: **never let a thread return early before a
   `__syncthreads()` the rest of the block still needs to reach.**

*Takeaway for the course:* this is the standard "convert redundant global
loads into one shared load + many fast shared reads" tiling pattern you'd
apply to any stencil (image blur, heat equation, cellular automata, etc.).
The speedup comes from trading global-memory bandwidth for on-chip shared
-memory bandwidth, which is roughly an order of magnitude faster and much
lower latency.

---

## V4 — warp shuffles (skipping shared memory entirely for some neighbors)

File: `kernels_v4_regqueue.cu` (`k_fd_time_step_v4`, `k_fd_time_step_v4_image`).

**The optimization: replace some of V3's shared-memory reads with warp
shuffles** — register-to-register exchange between threads in the same
warp, which is even faster than shared memory because it never touches the
on-chip memory hierarchy at all.

The key observation: the thread block is shaped `32 x 8` (32 threads along
the fast-varying axis, 8 along the other). Since a warp is exactly 32
threads and CUDA groups threads into warps by their linear index, **each
row of 32 threads along the fast axis is exactly one warp.** That means a
thread's neighbors *along that axis* (the ones the stencil needs) aren't
just nearby in memory — they are literally other lanes of the thread's own
warp. Instead of going through shared memory to get a neighbor's value, the
thread can ask for it directly with `__shfl_up_sync` / `__shfl_down_sync`:

```cpp
const float shfl_minus = __shfl_up_sync(0xFFFFFFFFu, pc_i, k);
const float shfl_plus  = __shfl_down_sync(0xFFFFFFFFu, pc_i, k);
const float zminus = (lane >= k)      ? shfl_minus : tile[tx][tz - k];
const float zplus  = (lane < 32 - k)  ? shfl_plus  : tile[tx][tz + k];
```

Neighbors along the *other* axis (a different warp's data) still come from
the shared-memory tile, same as V3 — shuffles only work within a warp, so
they can't help there. Lanes close to the edge of the warp (within `half`
of lane 0 or lane 31) don't have an in-warp neighbor that far away, so those
specific lanes fall back to reading the shared tile instead, exactly like
V3 did for everyone.

Two subtleties worth internalizing, since they generalize to any
warp-shuffle kernel:

1. **Shuffles are collective, just like `__syncthreads()`.** Every lane
   named in the mask must execute the *same* shuffle call, or you get
   undefined behavior. So the code can't have some lanes skip the shuffle
   based on validity — instead **every** lane calls both shuffles
   unconditionally (using the full `0xFFFFFFFFu` mask), and only *after*
   getting the results does each thread decide (branch) whether to actually
   use the shuffled value or the tile value. Divergent branching on an
   already-computed value is fine; divergently skipping a collective call is
   not.
2. **Because of (1), the "bail out early if I'm out of range" guard has to
   move even later than in V3** — it can't happen before the shuffle calls
   either, since those are also collective. So every input this kernel reads
   before its final validity check (`vdt2`, `sponge`, `p_prev`) goes through
   the same clamped `safe_load()` helper as V3's tile load, and only the
   very last two writes (`p_cur[i]`, `p_next[i]`) — after the real
   validity check — use the raw, unclamped index.

*Takeaway for the course:* warp shuffles are the next rung down from shared
memory on the "how close is this data to the ALU" ladder — no shared-memory
bank access, no address computation, just a direct lane-to-lane register
exchange. They only apply when the data you need is guaranteed to live in
another thread of *your own warp*, which is why this version only replaces
the neighbor lookups along the axis whose block dimension happens to equal
`warpSize` — the axis that maps 1:1 onto a warp. The other axis still needs
shared memory because it spans multiple warps.

---

## Summary table

| Version | Kernels per step | Key idea | CUDA concept |
|---|---|---|---|
| V0 | 3 (stencil, sponge, imaging) | textbook, one job per kernel | coalesced global access (baseline) |
| V1 | 2 | fuse sponge into stencil | kernel fusion — reuse a value still in a register |
| V2 | 2 (+ tiny cleanup kernel on backward pass) | fuse imaging into stencil, split off the unsafe subset | fusion with a correctness-driven split; sparse follow-up kernel |
| V3 | 2 | load each neighborhood once into shared memory | shared-memory stencil tiling + halo loading |
| V4 | 2 | get same-warp neighbors via register shuffle instead of shared memory | `__shfl_up/down_sync`, warp-as-hardware-unit |

Each rung removes one more piece of redundant global-memory traffic (or, in
V4's case, redundant *shared*-memory traffic) without changing a single
computed value — the whole ladder is a case study in "same math, cheaper
memory path."
