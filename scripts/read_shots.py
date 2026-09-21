#!/usr/bin/env python3
"""Read shot gathers in the RTMS format written by rtm_synth
(layout documented in src/io/seismic_io.cpp).

    read_shots.py data/synthetic/shots.bin                    # header + geometry
    read_shots.py data/real/marmousi_shots.bin --shot 0       # plot one gather
    read_shots.py data/real/marmousi_shots.bin --shot 0 --save g0.png

The file is memory-mapped, so a 900 MB Marmousi file opens instantly and only
the shot you touch is read. Importable too:

    from read_shots import read_shots
    hdr, s = read_shots("data/synthetic/shots.bin")
    s.traces[0]          # (nrec, nt) float32 view of shot 0, traces[ir, it]
"""
import argparse
import os
import struct
from types import SimpleNamespace

import numpy as np

HEADER_BYTES = 32


def read_shots(path):
    """Return (hdr, shots). hdr is a dict; shots has sx, sz (nshots,),
    rx, rz (nshots, nrec) and traces (nshots, nrec, nt) -- all views on a memmap."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES or raw[:4] != b"RTMS":
        raise SystemExit(f"{path}: not an RTMS file (bad magic)")
    version, nshots, nrec, nt = struct.unpack("<iiii", raw[4:20])
    dt, flags, reserved = struct.unpack("<fii", raw[20:32])
    if version != 1:
        raise SystemExit(f"{path}: unsupported RTMS version {version}")
    if nshots <= 0 or nrec <= 0 or nt <= 0 or not dt > 0:
        raise SystemExit(f"{path}: header contains invalid values")

    per_shot = 2 + 2 * nrec + nrec * nt            # floats per shot record
    expected = HEADER_BYTES + 4 * nshots * per_shot
    actual = os.path.getsize(path)
    if actual != expected:
        raise SystemExit(f"{path}: size {actual} B, expected {expected} B "
                         f"(nshots={nshots}, nrec={nrec}, nt={nt}) -- truncated?")

    body = np.memmap(path, dtype=np.float32, mode="r",
                     offset=HEADER_BYTES, shape=(nshots, per_shot))
    hdr = dict(version=version, nshots=nshots, nrec=nrec, nt=nt, dt=dt,
               flags=flags, reserved=reserved, bytes=actual)
    shots = SimpleNamespace(
        sx=body[:, 0],
        sz=body[:, 1],
        rx=body[:, 2:2 + nrec],
        rz=body[:, 2 + nrec:2 + 2 * nrec],
        traces=body[:, 2 + 2 * nrec:].reshape(nshots, nrec, nt),
    )
    return hdr, shots


def print_info(path, hdr, s):
    nshots, nrec, nt, dt = hdr["nshots"], hdr["nrec"], hdr["nt"], hdr["dt"]
    print(f"{path}")
    print(f"  RTMS v{hdr['version']}   {hdr['bytes'] / 1e6:.1f} MB")
    print(f"  nshots = {nshots}   nrec = {nrec}   nt = {nt}   dt = {dt:.6f} s"
          f"   T = {nt * dt:.3f} s")
    sx = np.asarray(s.sx)
    if nshots <= 8:
        sx_str = ", ".join(f"{x:.1f}" for x in sx)
    else:
        sx_str = (", ".join(f"{x:.1f}" for x in sx[:3]) + ", ..., "
                  + ", ".join(f"{x:.1f}" for x in sx[-2:]))
    print(f"  sx = [{sx_str}] m   sz = {float(s.sz[0]):.1f} m")
    rx0 = np.asarray(s.rx[0])
    drx = float(rx0[1] - rx0[0]) if nrec > 1 else 0.0
    print(f"  rx = {rx0[0]:.1f} .. {rx0[-1]:.1f} m every {drx:.1f} m"
          f"   rz = {float(s.rz[0, 0]):.1f} m")
    same = all(np.array_equal(s.rx[i], rx0) for i in range(min(nshots, 8)))
    print(f"  receiver line identical for every shot: {'yes' if same else 'NO'}")
    tr0 = np.asarray(s.traces[0])
    print(f"  shot 0 amplitude: max |a| = {np.abs(tr0).max():.4g}, "
          f"rms = {np.sqrt(np.mean(tr0 ** 2)):.4g}")


def plot_gather(path, hdr, s, ishot, perc, cmap, save):
    import matplotlib.pyplot as plt

    nt, dt = hdr["nt"], hdr["dt"]
    g = np.asarray(s.traces[ishot])                     # (nrec, nt)
    rx = np.asarray(s.rx[ishot])
    clip = np.percentile(np.abs(g), perc)
    clip = clip if clip > 0 else (np.abs(g).max() or 1.0)

    plt.figure(figsize=(10, 6))
    plt.imshow(g.T, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
               extent=[rx[0], rx[-1], nt * dt, 0])
    plt.axvline(float(s.sx[ishot]), color="r", lw=0.8, ls="--", label="source x")
    plt.colorbar(label="pressure")
    plt.xlabel("receiver x (m)"); plt.ylabel("time (s)")
    plt.title(f"{os.path.basename(path)}  shot {ishot}/{hdr['nshots']}  "
              f"sx = {float(s.sx[ishot]):.0f} m")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"wrote {save}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("path")
    p.add_argument("--shot", type=int, default=None, help="plot this shot gather")
    p.add_argument("--perc", type=float, default=98.0, help="clip percentile")
    p.add_argument("--cmap", default="gray")
    p.add_argument("--save", default=None, help="write PNG instead of showing")
    a = p.parse_args()

    hdr, shots = read_shots(a.path)
    print_info(a.path, hdr, shots)
    if a.shot is not None:
        if not 0 <= a.shot < hdr["nshots"]:
            raise SystemExit(f"--shot must be in [0, {hdr['nshots'] - 1}]")
        plot_gather(a.path, hdr, shots, a.shot, a.perc, a.cmap, a.save)
