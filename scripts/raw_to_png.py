#!/usr/bin/env python3
"""Render a raw float32 array (v[ix*nz+iz]) to PNG. Standard library only --
the numpy/matplotlib path is scripts/plot_image.py."""
import argparse, array, os, struct, sys, zlib

p = argparse.ArgumentParser()
p.add_argument('path'); p.add_argument('--nx', type=int, required=True)
p.add_argument('--nz', type=int, required=True)
p.add_argument('--out', default=None)
p.add_argument('--perc', type=float, default=99.0, help='clip percentile for seismic mode')
p.add_argument('--mode', choices=['seismic', 'velocity'], default='seismic')
p.add_argument('--width', type=int, default=1400, help='max output width in pixels')
a = p.parse_args()

d = array.array('f'); d.frombytes(open(a.path, 'rb').read())
if len(d) != a.nx * a.nz:
    raise SystemExit('expected %d samples, got %d' % (a.nx * a.nz, len(d)))

if a.mode == 'seismic':
    mag = sorted(abs(v) for v in d)
    clip = mag[min(len(mag) - 1, int(a.perc / 100.0 * len(mag)))] or (mag[-1] or 1.0)
    lo, hi = -clip, clip
else:
    lo, hi = min(d), max(d)
    if hi <= lo: hi = lo + 1.0
print('%s  n=%d  min=%.6g max=%.6g  display [%.6g, %.6g]' % (a.path, len(d), min(d), max(d), lo, hi))

step = max(1, (a.nx + a.width - 1) // a.width)
W, H = (a.nx + step - 1) // step, (a.nz + step - 1) // step
scale = 255.0 / (hi - lo)
rows = []
for jz in range(H):
    iz = jz * step
    row = bytearray(b'\x00')                     # PNG filter type 0
    base = iz
    for jx in range(W):
        v = d[(jx * step) * a.nz + base]
        g = int((v - lo) * scale)
        row.append(0 if g < 0 else (255 if g > 255 else g))
    rows.append(bytes(row))

def chunk(tag, data):
    return (struct.pack('>I', len(data)) + tag + data +
            struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 0, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(b''.join(rows), 6))
       + chunk(b'IEND', b''))
out = a.out or os.path.splitext(a.path)[0] + '.png'
open(out, 'wb').write(png)
print('wrote %s  (%d x %d, downsampled x%d)' % (out, W, H, step))
