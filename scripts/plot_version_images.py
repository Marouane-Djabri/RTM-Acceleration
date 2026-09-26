#!/usr/bin/env python3
"""Report figures of the migrated Marmousi images: one per version, the
differences against the CPU reference, and Devito next to our engine.

    myVenv/bin/python scripts/plot_version_images.py

Reads (fixed 12.5 m dataset, image[ix*nz+iz], float32):
    results/ref/marmousi_ref.bin        CPU reference image
    results/images/cuda-v0..v4.bin      one image per GPU version  (scripts/pod/images_run.sh)
    results/images/devito.bin           Devito's image             (same script)
Writes to results/plots/images/:
    versions_grid.png        reference + the five GPU versions, same display
    version_differences.png  (version - reference), in units of 1e-5 of the reference's peak
    devito_vs_ours.png       cuda-v1 and Devito side by side, and their difference

Display: the same Laplacian filter the C++ engine applies (src/rtm/imaging.cpp)
to remove RTM's low-frequency noise, then each image normalized by its 99th
percentile so two codes with different amplitude scaling can be compared.
The difference maps use the RAW images (no filter) against the reference.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NX, NZ, DX, DZ = 1361, 281, 12.5, 12.5
REFERENCE = "results/ref/marmousi_ref.bin"
VERSIONS = ["cuda-v0", "cuda-v1", "cuda-v2", "cuda-v3", "cuda-v4"]
OUT = "results/plots/images"
EXTENT_KM = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]


def load(path):
    if not os.path.exists(path):
        return None
    image = np.fromfile(path, dtype=np.float32)
    if image.size != NX * NZ:
        raise SystemExit(f"{path}: {image.size} samples, expected {NX * NZ}")
    return image.reshape(NX, NZ).T            # -> [depth, x] for plotting


def laplacian(image):
    """Same 5-point Laplacian as the engine's --filter laplacian."""
    out = np.zeros_like(image)
    out[1:-1, 1:-1] = ((image[1:-1, 2:] - 2 * image[1:-1, 1:-1] + image[1:-1, :-2]) / DX ** 2
                       + (image[2:, 1:-1] - 2 * image[1:-1, 1:-1] + image[:-2, 1:-1]) / DZ ** 2)
    return out


def for_display(image):
    filtered = laplacian(image)
    return filtered / (np.percentile(np.abs(filtered), 99) or 1.0)


def correlation(a, b):
    a = a.ravel().astype(np.float64)     # float64: float32 sums round past 1.0
    b = b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def show(ax, image, title, cmap="gray", limit=0.6):
    im = ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit, extent=EXTENT_KM, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("depth (km)")
    return im


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/{name}")


def main():
    reference = load(REFERENCE)
    if reference is None:
        raise SystemExit(f"missing {REFERENCE}")
    versions = {v: load(f"results/images/{v}.bin") for v in VERSIONS}
    versions = {v: img for v, img in versions.items() if img is not None}
    devito = load("results/images/devito.bin")
    if not versions and devito is None:
        raise SystemExit("no images in results/images/: run scripts/pod/images_run.sh on a pod, then sync")

    # 1. every version, displayed identically
    if versions:
        panels = [("CPU reference", reference)] + list(versions.items())
        fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(13, 2.6 * ((len(panels) + 1) // 2)))
        for ax, (name, image) in zip(axes.flat, panels):
            label = name if name == "CPU reference" else f"{name}   (corr. with reference {correlation(image, reference):.7f})"
            show(ax, for_display(image), label)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle("Migrated Marmousi image, every version (Laplacian-filtered, same display)")
        save(fig, "versions_grid.png")

        # 2. what actually differs: raw difference, in units of 1e-5 of the reference peak
        peak = float(np.abs(reference).max())
        fig, axes = plt.subplots(len(versions), 1, figsize=(10, 2.4 * len(versions)), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, (name, image) in zip(axes, versions.items()):
            difference = (image - reference) / peak * 1e5
            im = show(ax, difference, f"{name} - reference   (max {np.abs(difference).max():.2f})",
                      cmap="seismic", limit=2.0)
        fig.colorbar(im, ax=axes, label="difference, units of 1e-5 of the reference peak")
        fig.suptitle("Differences from the CPU reference (floating-point rounding)")
        save(fig, "version_differences.png")

    # 3. Devito next to our engine
    if devito is not None:
        ours = versions.get("cuda-v1", reference)
        ours_name = "cuda-v1" if "cuda-v1" in versions else "CPU reference"
        ours_display, devito_display = for_display(ours), for_display(devito)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        show(axes[0], ours_display, f"ours ({ours_name})")
        show(axes[1], devito_display, "Devito")
        show(axes[2], devito_display - ours_display, "Devito - ours (both normalized)", cmap="seismic")
        fig.suptitle(f"Devito vs our engine: correlation {correlation(devito, ours):.3f} raw, "
                     f"{correlation(devito_display, ours_display):.3f} after the display filter")
        save(fig, "devito_vs_ours.png")


if __name__ == "__main__":
    main()
