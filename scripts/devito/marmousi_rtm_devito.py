#!/usr/bin/env python3
"""
2D acoustic RTM on the fixed Marmousi dataset, implemented with Devito
(github.com/devitocodes/devito) -- an independently-developed, widely used
finite-difference DSL from the seismic-imaging research community (Imperial
College London / SLIM). It is used here as an INDUSTRY REFERENCE
cross-check: a second, independently written code, migrating the exact same
input files, so that if its image agrees with this project's own cpu/cpu-opt/
cuda-* engines, that is strong evidence those engines are numerically correct
-- not just internally consistent with themselves.

It is not part of the CUDA optimization ladder (see docs/OPTIMIZATION_PLAN.md)
and does not write to results/benchmarks.csv / results/compare.csv: those
files are reserved for the project's own engines on the same host, "so one
chart, one honest 1x" (OPTIMIZATION_PLAN.md 0). This script writes its own
results/ref/marmousi_devito.bin and results/ref/devito_benchmark.csv instead.

Reads the SAME files as ./build/rtm and reproduces the SAME physics as
CPUReferenceRTM (src/rtm/rtm_cpu.cpp) as closely as Devito's symbolic PDE
API allows:
  * shot-profile acoustic RTM, zero-lag cross-correlation imaging condition
  * constant-density acoustic wave equation, 2nd-order leapfrog in time
  * spatial order matches --order (Devito's space_order)
  * extended grid: nb absorbing cells added on all four sides, velocity
    extended by edge replication (rtm_cpu.cpp setup(), step 1)
  * Cerjan (1985) exponential sponge w(i) = exp(-(alpha*(nb-d))^2), applied
    by POST-multiplying the wavefield every step -- not injected as an
    additive PDE damping term (rtm_cpu.cpp setup(), step 2)
  * Ricker wavelet, delayed by t0 = 1.2/f0 (rtm_cpu.cpp setup(), step 3)
  * direct-arrival mute on the recorded traces before backward propagation
    (src/io/seismic_io.cpp mute_direct_wave)

It does NOT reproduce: the illumination compensation or Laplacian filter
(separate post-processing in the C++ tool, out of scope here), and it does
NOT need to match bit-for-bit -- different discretization of the boundary,
different time-stepping code, different compiler. Compare with
rtm_compare's L2rel/correlation numbers as a similarity measure, not a gate.

Usage (see docs/runbook/15_devito_reference.md):
    python3 scripts/devito/marmousi_rtm_devito.py \\
        --velocity data/real/marmousi_vp_12.5m.bin \\
        --shots    data/real/marmousi_shots.bin \\
        --output   results/ref/marmousi_devito.bin \\
        --order 8 --nb 50 --f0 10 --store-interval 10 --mute-direct 1500
"""
import argparse
import csv
import os
import platform
import socket
import struct
import time as walltime

import numpy as np


# =============================================================================
# File readers -- same formats as README.md sections 4/5, ported from
# scripts/read_shots.py and segy_to_raw.py's sidecar header convention.
# =============================================================================

def read_velocity(path):
    hdr_path = path + ".hdr"
    if not os.path.exists(hdr_path):
        raise SystemExit(f"missing sidecar header: {hdr_path}")
    hdr = {}
    for line in open(hdr_path):
        line = line.split("#", 1)[0].strip()      # strip full-line and inline comments
        if not line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        hdr[key] = value
    nx, nz = int(hdr["nx"]), int(hdr["nz"])
    dx, dz = float(hdr["dx"]), float(hdr["dz"])
    ox, oz = float(hdr.get("ox", 0.0)), float(hdr.get("oz", 0.0))
    layout = hdr.get("layout", "zfast")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != nx * nz:
        raise SystemExit(f"{path}: expected {nx * nz} samples, got {raw.size}")
    if layout == "zfast":
        vp = raw.reshape(nx, nz)              # vp[ix, iz], already the layout Devito wants
    elif layout == "xfast":
        vp = raw.reshape(nz, nx).T.copy()
    else:
        raise SystemExit(f"{path}: unknown layout '{layout}'")
    return vp, nx, nz, dx, dz, ox, oz


RTMS_HEADER_BYTES = 32


def read_shots(path):
    with open(path, "rb") as f:
        raw = f.read(RTMS_HEADER_BYTES)
    if len(raw) < RTMS_HEADER_BYTES or raw[:4] != b"RTMS":
        raise SystemExit(f"{path}: not an RTMS file (bad magic)")
    version, nshots, nrec, nt = struct.unpack("<iiii", raw[4:20])
    dt, flags, reserved = struct.unpack("<fii", raw[20:32])
    if version != 1:
        raise SystemExit(f"{path}: unsupported RTMS version {version}")
    per_shot = 2 + 2 * nrec + nrec * nt
    expected = RTMS_HEADER_BYTES + 4 * nshots * per_shot
    actual = os.path.getsize(path)
    if actual != expected:
        raise SystemExit(f"{path}: size {actual} B, expected {expected} B -- truncated?")
    body = np.memmap(path, dtype=np.float32, mode="r",
                      offset=RTMS_HEADER_BYTES, shape=(nshots, per_shot))
    hdr = dict(nshots=nshots, nrec=nrec, nt=nt, dt=dt)
    sx = np.asarray(body[:, 0])
    sz = np.asarray(body[:, 1])
    rx = np.asarray(body[:, 2:2 + nrec])
    rz = np.asarray(body[:, 2 + nrec:2 + 2 * nrec])
    traces = body[:, 2 + 2 * nrec:].reshape(nshots, nrec, nt)   # memmap, per-shot copy on read
    return hdr, sx, sz, rx, rz, traces


def mute_direct_wave(traces, sx, sz, rx, rz, dt, v_surface, t_pad, t_taper):
    """Port of src/io/seismic_io.cpp:mute_direct_wave -- zero + raised-cosine
    taper of the direct arrival so it doesn't migrate as a false shallow
    reflector. traces: (nrec, nt) float32, muted in place."""
    if v_surface <= 0.0:
        return traces
    nrec, nt = traces.shape
    offset = np.sqrt((rx - sx) ** 2 + (rz - sz) ** 2)
    t0 = offset / v_surface + t_pad
    i0 = (t0 / dt).astype(np.int64)
    i1 = i0 + max(1, int(round(t_taper / dt)))
    for ir in range(nrec):
        lo, hi = int(i0[ir]), min(int(i1[ir]), nt)
        if lo >= nt:
            continue
        traces[ir, :min(lo + 1, nt)] = 0.0
        if hi > lo + 1:
            u = (np.arange(lo + 1, hi) - lo).astype(np.float32) / (hi - lo)
            traces[ir, lo + 1:hi] *= 0.5 * (1.0 - np.cos(np.pi * u))
    return traces


# =============================================================================
# Devito model: extended grid (nb absorbing cells on all sides, velocity
# extended by edge replication), Cerjan sponge, one compile reused across
# every shot (only source/receiver coordinates and receiver data change).
# =============================================================================

def build_devito_model(vp, nx, nz, dx, dz, ox, oz, nb, order, f0, dt, nt,
                        store_interval, sponge_alpha):
    from devito import (Grid, TimeFunction, Function, Eq, Inc, solve,
                         Operator, SparseTimeFunction, ConditionalDimension)

    nxe, nze = nx + 2 * nb, nz + 2 * nb
    origin = (ox - nb * dx, oz - nb * dz)
    extent = ((nxe - 1) * dx, (nze - 1) * dz)
    grid = Grid(shape=(nxe, nze), extent=extent, origin=origin, dtype=np.float32)
    time_dim = grid.time_dim

    vp_ext = np.pad(vp, ((nb, nb), (nb, nb)), mode="edge")
    m = Function(name="m", grid=grid, space_order=order)
    m.data[:] = 1.0 / (vp_ext.astype(np.float32) ** 2)

    def taper_1d(n):
        w = np.ones(n, dtype=np.float32)
        for i in range(n):
            d = min(i, n - 1 - i)
            if d < nb:
                a = sponge_alpha * (nb - d)
                w[i] = np.exp(-a * a)
        return w

    sponge = Function(name="sponge", grid=grid, space_order=order)
    sponge.data[:] = np.outer(taper_1d(nxe), taper_1d(nze))

    # ---- forward operator: propagate the source wavelet, save a subsampled
    # wavefield history (mirrors --store-interval's memory/accuracy trade-off).
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=order)
    stencil_fwd = Eq(u.forward, solve(m * u.dt2 - u.laplace, u.forward))
    sponge_fwd = Eq(u.forward, u.forward * sponge)

    nsave = nt // store_interval + 1
    t_sub = ConditionalDimension("t_sub", parent=time_dim, factor=store_interval)
    usave = TimeFunction(name="usave", grid=grid, time_order=2, space_order=order,
                          save=nsave, time_dim=t_sub)
    save_eq = Eq(usave, u.forward)

    src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=nt)
    t0 = 1.2 / f0
    tt = np.arange(nt, dtype=np.float64) * dt - t0
    a = (np.pi * f0 * tt) ** 2
    src.data[:, 0] = ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)
    src_term = src.inject(field=u.forward, expr=src * dt ** 2 / m)

    def make_fwd(nrec_):
        rec = SparseTimeFunction(name="rec", grid=grid, npoint=nrec_, nt=nt)
        rec_term = rec.interpolate(expr=u)
        op = Operator([stencil_fwd, sponge_fwd] + src_term + rec_term + [save_eq],
                       name="Fwd")
        return op, rec

    # ---- adjoint operator: back-propagate the (muted) receiver data,
    # correlate in-place against the saved forward wavefield every store step.
    v = TimeFunction(name="v", grid=grid, time_order=2, space_order=order)
    stencil_adj = Eq(v.backward, solve(m * v.dt2 - v.laplace, v.backward))
    sponge_adj = Eq(v.backward, v.backward * sponge)
    image = Function(name="image", grid=grid)
    image_update = Inc(image, usave * v)

    def make_adj(nrec_):
        rec_b = SparseTimeFunction(name="rec_b", grid=grid, npoint=nrec_, nt=nt)
        rec_inject = rec_b.inject(field=v.backward, expr=rec_b * dt ** 2 / m)
        op = Operator([stencil_adj, sponge_adj] + rec_inject + [image_update],
                       name="Adj")
        return op, rec_b

    return dict(grid=grid, u=u, v=v, usave=usave, image=image,
                src=src, make_fwd=make_fwd, make_adj=make_adj,
                nxe=nxe, nze=nze, nb=nb, dt=dt)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--velocity", required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--order", type=int, default=8, help="spatial FD order (Devito space_order)")
    p.add_argument("--nb", type=int, default=50, help="absorbing cells on each side")
    p.add_argument("--sponge-alpha", type=float, default=0.0053, help="Cerjan coefficient")
    p.add_argument("--f0", type=float, default=10.0, help="Ricker peak frequency, Hz")
    p.add_argument("--store-interval", type=int, default=10,
                    help="save the forward wavefield every N steps (memory/accuracy trade-off)")
    p.add_argument("--mute-direct", type=float, default=0.0,
                    help="direct-arrival mute velocity, m/s (0 = no mute)")
    p.add_argument("--max-shots", type=int, default=0, help="debug: migrate only the first N shots")
    p.add_argument("--benchmark-csv", default=None,
                    help="append one timing row (own schema, not results/benchmarks.csv)")
    p.add_argument("--dataset", default="marmousi12")
    a = p.parse_args()

    print(f"=== reading velocity: {a.velocity} ===")
    vp, nx, nz, dx, dz, ox, oz = read_velocity(a.velocity)
    print(f"  nx={nx} nz={nz} dx={dx} dz={dz}  vmin={vp.min():.1f} vmax={vp.max():.1f} m/s")

    print(f"=== reading shots: {a.shots} ===")
    hdr, sx, sz, rx, rz, traces = read_shots(a.shots)
    nshots, nrec, nt, dt = hdr["nshots"], hdr["nrec"], hdr["nt"], hdr["dt"]
    print(f"  nshots={nshots} nrec={nrec} nt={nt} dt={dt}  T={nt * dt:.3f} s")
    n_migrate = nshots if a.max_shots <= 0 else min(a.max_shots, nshots)
    if n_migrate != nshots:
        print(f"  --max-shots: migrating only {n_migrate}/{nshots} shots")

    print(f"=== building Devito operators (order={a.order}, nb={a.nb}, compiled once) ===")
    t_build0 = walltime.time()
    model = build_devito_model(vp, nx, nz, dx, dz, ox, oz, a.nb, a.order, a.f0,
                                dt, nt, a.store_interval, a.sponge_alpha)
    op_fwd, rec_fwd = model["make_fwd"](nrec)
    op_adj, rec_adj = model["make_adj"](nrec)
    print(f"  JIT compile: {walltime.time() - t_build0:.1f} s")

    image = model["image"]
    u, v, usave = model["u"], model["v"], model["usave"]

    t_total0 = walltime.time()
    for ishot in range(n_migrate):
        t_shot0 = walltime.time()
        shot_traces = np.array(traces[ishot], dtype=np.float32)   # (nrec, nt), copy off the memmap
        if a.mute_direct > 0.0:
            mute_direct_wave(shot_traces, float(sx[ishot]), float(sz[ishot]),
                              rx[ishot], rz[ishot], dt, a.mute_direct, 1.2 / a.f0, 1.2 / a.f0)

        model["src"].coordinates.data[0, :] = [sx[ishot], sz[ishot]]
        rec_fwd.coordinates.data[:, 0] = rx[ishot]
        rec_fwd.coordinates.data[:, 1] = rz[ishot]

        u.data[:] = 0.0
        op_fwd.apply(dt=dt)

        rec_adj.coordinates.data[:, 0] = rx[ishot]
        rec_adj.coordinates.data[:, 1] = rz[ishot]
        rec_adj.data[:] = shot_traces.T                          # (nt, nrec), reverse-time injection

        v.data[:] = 0.0
        op_adj.apply(dt=dt)

        print(f"  shot {ishot + 1}/{n_migrate}  sx={sx[ishot]:.0f} m  "
              f"{walltime.time() - t_shot0:.1f} s")

    t_migrate = walltime.time() - t_total0
    print(f"=== migrated {n_migrate} shots in {t_migrate:.1f} s "
          f"({t_migrate / n_migrate:.1f} s/shot) ===")

    img_interior = np.asarray(image.data)[a.nb:a.nb + nx, a.nb:a.nb + nz].astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    img_interior.tofile(a.output)
    print(f"wrote {a.output}  ({nx * nz * 4 / 1e6:.1f} MB, image[ix*nz+iz], nx={nx} nz={nz})")
    print(f"  max |I| = {np.abs(img_interior).max():.4g}")

    if a.benchmark_csv:
        need_header = not os.path.exists(a.benchmark_csv)
        with open(a.benchmark_csv, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["date", "host", "dataset", "nx", "nz", "nb", "order",
                            "nt", "nshots", "store_interval", "t_migrate_s",
                            "s_per_shot", "devito_version"])
            import devito
            w.writerow([walltime.strftime("%Y-%m-%d %H:%M:%S"), socket.gethostname(),
                        a.dataset, nx, nz, a.nb, a.order, nt, n_migrate,
                        a.store_interval, f"{t_migrate:.2f}", f"{t_migrate / n_migrate:.2f}",
                        devito.__version__])
        print(f"appended timing row to {a.benchmark_csv}")


if __name__ == "__main__":
    main()
