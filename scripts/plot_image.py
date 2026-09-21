#!/usr/bin/env python3
"""Display an RTM image or velocity model stored as raw float32, v[ix*nz+iz]."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--nx", type=int, required=True)
p.add_argument("--nz", type=int, required=True)
p.add_argument("--dx", type=float, default=10.0)
p.add_argument("--dz", type=float, default=10.0)
p.add_argument("--perc", type=float, default=98.0, help="clip percentile")
p.add_argument("--cmap", default="gray")
p.add_argument("--save", default=None)
a = p.parse_args()

d = np.fromfile(a.path, dtype=np.float32)
assert d.size == a.nx * a.nz, f"expected {a.nx*a.nz} samples, got {d.size}"
img = d.reshape(a.nx, a.nz).T          # -> (nz, nx) for imshow

clip = np.percentile(np.abs(img), a.perc)
clip = clip if clip > 0 else np.abs(img).max()

plt.figure(figsize=(10, 6))
plt.imshow(img, cmap=a.cmap, vmin=-clip, vmax=clip, aspect="auto",
           extent=[0, a.nx * a.dx, a.nz * a.dz, 0])
plt.colorbar(label="amplitude")
plt.xlabel("x (m)"); plt.ylabel("depth (m)"); plt.title(a.path)
plt.tight_layout()
plt.savefig(a.save, dpi=150) if a.save else plt.show()