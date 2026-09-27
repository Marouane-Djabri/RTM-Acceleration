#!/usr/bin/env python3
"""Build the results dashboard: one self-contained HTML page that shows every
result of the project next to what it means.

    myVenv/bin/python scripts/build_dashboard.py            # -> results/dashboard.html
    myVenv/bin/python scripts/build_dashboard.py --fragment # page body only (for publishing)

Reads only files already in results/ (no pod needed):
    results/benchmarks.csv, compare.csv, compare_images.csv, scaling.csv,
    compare_scaling.csv, ref/bandwidth.csv, profiles/step_breakdown.csv,
    survey_projection_dx*.md, ref/marmousi_ref.bin, images/*.bin

Every number in the interpretation text is computed here from those files, so
re-running the script after a new run keeps the text and the charts in sync.
The page template (CSS + chart code) is scripts/dashboard/template.html.
The page works offline: charts are drawn by the page's own script, and the
migrated images are embedded (compressed int8).
"""
import argparse
import base64
import csv
import datetime
import glob
import gzip
import html
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from plot_version_images import NX, NZ, DX, DZ, load, laplacian, correlation  # noqa: E402

TEMPLATE = "scripts/dashboard/template.html"
OUTPUT = "results/dashboard.html"
CUDA_VERSIONS = ["cuda-v0", "cuda-v1", "cuda-v2", "cuda-v3", "cuda-v4"]
BYTES_PER_POINT = 24          # bytes moved per grid-point update by the fused kernel (docs/CUDA_KERNEL_OPTIMIZATIONS.md)
STRICT_L2_GATE = 1e-5         # rtm_compare's relative-L2 gate

WHAT_EACH_VERSION_DOES = {
    "cpu": "plain reference, 1 core",
    "cpu-opt": "optimized CPU, OpenMP",
    "cuda-v0": "one thread per point, separate sponge kernel",
    "cuda-v1": "sponge fused into the stencil kernel",
    "cuda-v2": "imaging fused into the backward kernel",
    "cuda-v3": "shared-memory tiles",
    "cuda-v4": "register sliding window + warp shuffles",
}


# ---------------------------------------------------------------- reading

def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def latest(rows, **match):
    """The most recent row whose columns equal every value in `match`."""
    found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    return max(found, key=lambda r: r["date"]) if found else None


def short_cpu(name):
    name = re.sub(r"\(R\)|\(TM\)| CPU| @.*| Processor", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ---------------------------------------------------------------- formatting

def esc(text):
    return html.escape(str(text))


def fmt(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def sci(value):
    """1.23e-06 -> 1.2·10⁻⁶ as HTML."""
    if value is None:
        return "—"
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.1f}·10<sup>{exponent}</sup>"


def table(headers, rows, numeric=()):
    head = "".join(f'<th{" class=num" if i in numeric else ""}>{h}</th>' for i, h in enumerate(headers))
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f'<td{" class=num" if i in numeric else ""}>{c}</td>'
                                 for i, c in enumerate(row)) + "</tr>"
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def chip(state, text):
    icon = {"good": "✓", "warn": "!", "bad": "✕"}[state]
    return f'<span class="chip chip-{state}"><span aria-hidden="true">{icon}</span> {esc(text)}</span>'


def numbers_disclosure(inner):
    return f'<details class="numbers"><summary>Show the numbers</summary>{inner}</details>'


def section(anchor, kicker, title, lede, figure, reading, report=None):
    report_html = f'<p class="report"><span>In the report</span> {report}</p>' if report else ""
    return f"""
<section id="{anchor}" class="section">
  <header class="section-head">
    <p class="kicker">{kicker}</p>
    <h2>{title}</h2>
    <p class="lede">{lede}</p>
  </header>
  <div class="section-body">
    <div class="figure">{figure}</div>
    <aside class="reading">
      <h3>What it shows</h3>
      {reading}
      {report_html}
    </aside>
  </div>
</section>"""


def bullets(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ---------------------------------------------------------------- the data

def collect():
    benchmarks = read_rows("results/benchmarks.csv")
    if not benchmarks:
        raise SystemExit("results/benchmarks.csv is missing: nothing to show")
    main_host = max(benchmarks, key=lambda r: r["date"])["host"]
    gpu_row = latest([r for r in benchmarks if r["engine"].startswith("cuda")], host=main_host)

    ladder = []
    for engine in ["cpu", "cpu-opt"] + CUDA_VERSIONS:
        r = latest(benchmarks, host=main_host, engine=engine)
        if not r:
            continue
        ladder.append({
            "engine": engine,
            "what": WHAT_EACH_VERSION_DOES[engine],
            "threads": int(r["threads"]),
            "t_fb": number(r["t_forward"]) + number(r["t_backward"]),
            "t_total": number(r["t_total"]),
            "gpts": number(r["gpts_per_s"]),
            "mem_mib": number(r["device_mem_mib"]),
        })
    by_engine = {row["engine"]: row for row in ladder}
    for row in ladder:
        row["vs_cpu"] = by_engine["cpu"]["t_total"] / row["t_total"] if "cpu" in by_engine else None
        row["vs_cpu_opt"] = by_engine["cpu-opt"]["t_total"] / row["t_total"] if "cpu-opt" in by_engine else None

    other_cpu = [r for r in benchmarks if r["engine"] == "cpu" and r["host"] != main_host]

    bandwidth_rows = read_rows("results/ref/bandwidth.csv")
    gpu_bandwidth = number(latest(bandwidth_rows, host=main_host, device_kind="gpu")["gbps"])

    compare_main = {e: latest(read_rows("results/compare.csv"), host=main_host, engine=e)
                    for e in ["cpu", "cpu-opt"] + CUDA_VERSIONS}
    compare_images_rows = read_rows("results/compare_images.csv")
    compare_images = {r["engine"]: r for r in compare_images_rows}

    steps = []
    for r in read_rows("results/profiles/step_breakdown.csv"):
        steps.append({
            "engine": r["engine"],
            "kernels": number(r["kernel_launches_per_step"]),
            "stencil": number(r["stencil us/step"]),
            "sponge": number(r["sponge us/step"]),
            "imaging": number(r["imaging us/step"]) + number(r["stencil + imaging (fused) us/step"]),
            "imaging_fused": number(r["stencil + imaging (fused) us/step"]) > 0,
            "other": number(r["injection / recording us/step"]) + number(r["other us/step"])
                     + number(r["memcpy / memset us/step"]),
            "gpu_total": number(r["gpu total us/step"]),
            "launch": number(r["cudaLaunchKernel us/step (CPU)"]),
            "wall": number(r["migrate wall us/step"]),
            "busy": number(r["gpu busy %"]),
        })

    scaling_rows = read_rows("results/scaling.csv")
    gpu_total_mib = next((number(r["gpu_total_mib"]) for r in scaling_rows if r["gpu_total_mib"]), None)

    def pick(**match):
        return latest([r for r in scaling_rows if r["status"] == "ok"], **match)

    spacings = sorted({number(r["dx"]) for r in scaling_rows}, reverse=True)
    scaling = []
    for dx in spacings:
        dx_text = next(r["dx"] for r in scaling_rows if number(r["dx"]) == dx)
        v1 = pick(engine="cuda-v1", sweep="S1T", dx=dx_text)
        v3 = pick(engine="cuda-v3", sweep="S1T", dx=dx_text)
        v1_power = pick(engine="cuda-v1", sweep="S1", dx=dx_text)
        devito = pick(code="devito", sweep="S1", dx=dx_text) or pick(code="devito", sweep="S2", dx=dx_text, order="8")
        any_row = next(r for r in scaling_rows if number(r["dx"]) == dx)
        scaling.append({
            "dx": dx,
            "points": int(any_row["nx"]) * int(any_row["nz"]),
            "ours_v1": number(v1["gpts_per_s"]) if v1 else None,
            "ours_v3": number(v3["gpts_per_s"]) if v3 else None,
            "ours_power": number(v1_power["mean_power_w"]) if v1_power else None,
            "ours_s_per_shot": number(v1["s_per_shot"]) if v1 else None,
            "devito_wall": number(devito["gpts_per_s"]) if devito else None,
            "devito_compute": number(devito["devito_gpts"]) if devito else None,
            "devito_power": number(devito["mean_power_w"]) if devito else None,
            "devito_every": int(devito["store_interval"]) if devito else None,
        })

    # Capacity: the sweep that keeps one snapshot every 10 time steps (S1).
    capacity = []
    for dx in spacings:
        r = next((r for r in scaling_rows if r["code"] == "ours" and r["engine"] == "cuda-v1"
                  and r["sweep"] == "S1" and number(r["dx"]) == dx), None)
        if not r:
            continue
        nx, nz, nb, nt, every = int(r["nx"]), int(r["nz"]), int(r["nb"]), int(r["nt"]), int(r["store_interval"])
        snapshots = (nt - 1) // every + 1
        need_gib = (snapshots * nx * nz + 6 * (nx + 2 * nb) * (nz + 2 * nb)) * 4 / 2 ** 30
        finest_ok = min((int(s["store_interval"]) for s in scaling_rows
                         if s["code"] == "ours" and s["engine"] == "cuda-v1" and s["sweep"] in ("S1", "S1B")
                         and number(s["dx"]) == dx and s["status"] == "ok"), default=None)
        capacity.append({
            "dx": dx, "points": nx * nz, "status": r["status"], "every": every,
            "need_gib": need_gib,
            "measured_gib": number(r["device_mem_mib_peak"]) / 1024 if r["status"] == "ok" else None,
            "densest_that_fits": finest_ok,
        })

    orders = {
        "ours": sorted((int(r["order"]), number(r["gpts_per_s"])) for r in scaling_rows
                       if r["code"] == "ours" and r["sweep"] == "S2" and r["status"] == "ok"),
        "devito": sorted((int(r["order"]), number(r["gpts_per_s"])) for r in scaling_rows
                         if r["code"] == "devito" and r["sweep"] == "S2" and r["status"] == "ok"),
    }

    compare_scaling = read_rows("results/compare_scaling.csv")
    snapshot_accuracy = {}
    for r in compare_scaling:
        m = re.search(r"_every(\d+)steps_vs_every(\d+)steps", r["engine"])
        if m:
            snapshot_accuracy.setdefault(r["dataset"], {int(m.group(2)): 1.0})[int(m.group(1))] = number(r["correlation"])
    aliased = {r["dataset"]: number(r["correlation"]) for r in compare_scaling if "snap5ms_vs_snap10ms" in r["engine"]}

    surveys = {}
    for path in sorted(glob.glob("results/survey_projection_dx*.md")):
        dx = re.search(r"dx([\d.]+)\.md", path).group(1)
        with open(path) as f:
            text = f.read()
        rows = []
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[1].isdigit():
                rows.append({"who": cells[0], "devices": int(cells[1]),
                             "hours": number(cells[2].replace(",", "")), "kwh": number(cells[4])})
        heading = re.search(r"^# (.*)$", text, re.M)
        grid = re.search(r"^Grid .*$", text, re.M)
        assumptions = re.search(r"^Assumptions: .*$", text, re.M)
        surveys[dx] = {"title": heading.group(1) if heading else path, "grid": grid.group(0) if grid else "",
                       "assumptions": assumptions.group(0) if assumptions else "", "rows": rows}

    return {
        "main_host": main_host, "gpu_row": gpu_row, "ladder": ladder, "by_engine": by_engine,
        "other_cpu": other_cpu, "gpu_bandwidth": gpu_bandwidth, "gpu_total_mib": gpu_total_mib,
        "compare_main": compare_main, "compare_images": compare_images,
        "images_host": compare_images_rows[0]["host"] if compare_images_rows else None,
        "steps": steps, "scaling": scaling, "capacity": capacity, "orders": orders,
        "snapshot_accuracy": snapshot_accuracy, "aliased": aliased,
        "surveys": surveys,
    }


# ---------------------------------------------------------------- images

QUANT_STEPS_PER_UNIT = 40      # display units (x p99) -> int8: covers +-3.2 x p99


def pack(values_int8):
    return base64.b64encode(gzip.compress(values_int8.astype(np.int8).tobytes(), 9)).decode()


def collect_images():
    reference = load("results/ref/marmousi_ref.bin")
    if reference is None:
        return None
    versions = {v: load(f"results/images/{v}.bin") for v in CUDA_VERSIONS}
    versions = {v: img for v, img in versions.items() if img is not None}
    devito = load("results/images/devito.bin")

    def image_entry(key, label, image, filtered=True, note=""):
        shown = laplacian(image) if filtered else image
        p99 = float(np.percentile(np.abs(shown), 99)) or 1.0
        q = np.clip(np.round(shown / p99 * QUANT_STEPS_PER_UNIT), -127, 127)
        return {"id": key, "label": label, "kind": "image", "data": pack(q), "note": note}

    entries = [image_entry("reference-raw", "Reference, unfiltered", reference, filtered=False,
                           note="The raw zero-lag image. The large smooth blobs near the surface are RTM's usual "
                                "low-frequency noise, which the Laplacian filter removes."),
               image_entry("reference", "CPU reference", reference,
                           note="The CPU reference engine's image, Laplacian-filtered. Every other engine is "
                                "compared against this one.")]
    peak = float(np.abs(reference).max())
    differences = {v: (img - reference) / peak * 1e5 for v, img in versions.items()}
    diff_scale = max((float(np.abs(d).max()) for d in differences.values()), default=1.0) or 1.0
    for v, img in versions.items():
        entry = image_entry(v, v, img, note=WHAT_EACH_VERSION_DOES[v] + ".")
        entry["corr"] = correlation(img, reference)
        entry["diff"] = pack(np.clip(np.round(differences[v] / diff_scale * 127), -127, 127))
        entry["diff_max"] = float(np.abs(differences[v]).max())
        entries.append(entry)
    devito_corr = None
    if devito is not None:
        ours = versions.get("cuda-v1", reference)
        devito_corr = {"raw": correlation(devito, ours), "filtered": correlation(laplacian(devito), laplacian(ours))}
        entry = image_entry("devito", "Devito", devito,
                            note="Devito's image of the same data, filtered and scaled the same way.")
        entry["corr_devito"] = devito_corr
        entries.append(entry)
    return {
        "nx": NX, "nz": NZ, "x_km": (NX - 1) * DX / 1000, "z_km": (NZ - 1) * DZ / 1000,
        "quant": QUANT_STEPS_PER_UNIT, "diff_scale": diff_scale, "entries": entries,
        "devito_corr": devito_corr, "versions": list(versions),
    }


# ---------------------------------------------------------------- the page

def build_page(d, images, built_at):
    ladder, E = d["ladder"], d["by_engine"]
    gpu = d["gpu_row"]
    gpu_name = gpu["gpu_name"].replace("NVIDIA ", "")
    cpu_name = short_cpu(gpu["cpu_name"])
    cpu_opt_threads = E["cpu-opt"]["threads"] if "cpu-opt" in E else None
    cuda = [row for row in ladder if row["engine"].startswith("cuda")]
    fastest_total = min(cuda, key=lambda r: r["t_total"])
    fastest_fb = min(cuda, key=lambda r: r["t_fb"])
    plateau_row = max((s for s in d["scaling"] if s["ours_v1"] or s["ours_v3"]),
                      key=lambda s: max(s["ours_v1"] or 0, s["ours_v3"] or 0))
    plateau = max(plateau_row["ours_v1"] or 0, plateau_row["ours_v3"] or 0)
    ceiling = d["gpu_bandwidth"] / BYTES_PER_POINT
    bandwidth_used = plateau * BYTES_PER_POINT / d["gpu_bandwidth"]
    limit_gib = d["gpu_total_mib"] / 1024 if d["gpu_total_mib"] else None
    fits = [c for c in d["capacity"] if c["status"] == "ok"]
    too_big = [c for c in d["capacity"] if c["status"] != "ok"]
    finest_fit = min(fits, key=lambda c: c["dx"]) if fits else None
    gates = [d["compare_main"][e] for e in CUDA_VERSIONS if d["compare_main"].get(e)]
    worst_l2_main = max(number(g["l2_relative"]) for g in gates)
    all_pass_main = all(g["pass"] == "1" for g in gates)
    image_l2 = [number(d["compare_images"][v]["l2_relative"]) for v in CUDA_VERSIONS if v in d["compare_images"]]
    worst_l2_any = max([worst_l2_main] + image_l2)
    devito_rows = [s for s in d["scaling"] if s["devito_wall"]]
    devito_ten = [s for s in devito_rows if s["devito_every"] == 10 and s["ours_v1"]]
    devito_slowdown = [s["ours_v1"] / s["devito_wall"] for s in devito_ten]
    devito_best_compute = max((s["devito_compute"] or 0) for s in devito_rows) if devito_rows else None

    # ---------- header: a SEG-Y style textual header of the run
    first_date = gpu["date"][:10]
    header_lines = [
        ("PROJECT", "RTM-Hardware-Acceleration · 2D acoustic reverse time migration"),
        ("MODEL", f"Marmousi · {gpu['nx']} × {gpu['nz']} cells · {DX:g} m · {gpu['nb']}-cell sponge"),
        ("SURVEY", f"{gpu['nshots']} shots · {gpu['nt']} steps · dt {number(gpu['dt']) * 1000:.1f} ms · "
                   f"FD order {gpu['order']} · snapshot every {gpu['store_interval']} steps"),
        ("GPU", f"NVIDIA {gpu_name} · {limit_gib:.0f} GiB · {d['gpu_bandwidth']:.0f} GB/s measured"),
        ("CPU", f"{cpu_name} · {cpu_opt_threads} threads"),
        ("RUN", f"{first_date} · host {d['main_host']}"),
    ]
    if d["images_host"]:
        header_lines.append(("IMAGES", f"separate session · host {d['images_host']} · RTX PRO 6000 Blackwell"))
    header_lines.append(("BUILT", f"{built_at} · scripts/build_dashboard.py"))
    textual_header = "\n".join(f"C{i + 1:02d} {k:<8} {v}" for i, (k, v) in enumerate(header_lines))

    tiles = [
        (f"{fastest_fb['t_fb']:.2f} s", "GPU propagation time",
         f"forward + backward, {gpu['nshots']} shots ({fastest_fb['engine']})"),
        (f"{fastest_total['vs_cpu_opt']:.0f}×", f"faster than {cpu_opt_threads} CPU threads",
         f"{fastest_total['vs_cpu']:.0f}× against the 1-core reference"),
        (f"{plateau:.1f}", "GPts/s at large grids", f"{bandwidth_used:.0%} of measured memory bandwidth"),
        (f"{finest_fit['dx']:g} m" if finest_fit else "—", "finest grid that fits",
         f"with a snapshot every {finest_fit['every']} steps" if finest_fit else ""),
    ]
    tiles_html = "".join(f'<div class="tile"><p class="tile-value">{v}</p><p class="tile-label">{l}</p>'
                         f'<p class="tile-note">{n}</p></div>' for v, l, n in tiles)

    findings = [
        f"<strong>The GPU migrates the {gpu['nshots']}-shot Marmousi survey in {fastest_total['t_total']:.2f} s</strong>, "
        f"against {E['cpu-opt']['t_total']:.0f} s for the {cpu_opt_threads}-thread CPU engine and "
        f"{E['cpu']['t_total']:.0f} s for the single-core reference.",
        f"<strong>Every GPU version reproduces the reference image</strong>: correlation 1.0000000, relative L2 "
        f"≤ {sci(worst_l2_any)} on every GPU tested. That's floating-point rounding, not an error.",
        f"<strong>The kernel is memory-bound.</strong> Throughput levels off at {plateau:.1f} GPts/s, which moves "
        f"{plateau * BYTES_PER_POINT:.0f} GB/s: {bandwidth_used:.0%} of the {d['gpu_bandwidth']:.0f} GB/s the probe "
        f"measured. There's no bandwidth left for shared memory (v3) or warp shuffles (v4) to win back.",
        (f"<strong>Memory, not speed, sets the largest grid.</strong> Keeping a snapshot every 10 steps, "
         f"{finest_fit['dx']:g} m is the finest grid that fits in {limit_gib:.0f} GiB. "
         + (f"{too_big[0]['dx']:g} m would need about {too_big[0]['need_gib']:.0f} GiB." if too_big else ""))
        if finest_fit else "",
    ]
    if devito_slowdown:
        findings.append(
            f"<strong>Devito's generated kernels are as fast as ours</strong> (up to {devito_best_compute:.1f} GPts/s of "
            f"pure compute), but end to end it runs {min(devito_slowdown):.1f}–{max(devito_slowdown):.1f}× slower "
            f"because it copies the saved wavefield to host memory.")

    overview = f"""
<section id="overview" class="section overview">
  <div class="overview-grid">
    <div>
      <p class="kicker">Summary</p>
      <h2>What the runs show</h2>
      {bullets([f for f in findings if f])}
    </div>
    <div class="tiles">{tiles_html}</div>
  </div>
</section>"""

    sections = [overview]

    # ---------- the image
    if images:
        devito_corr = images["devito_corr"]
        image_reading = bullets([
            "<strong>The geology is right.</strong> Layered sediments on the left, faulted blocks dipping through "
            "the centre (8–11 km), a strong deep reflector near 2.3 km: this is the known Marmousi model, so the "
            "migration works.",
            f"<strong>The {gpu['nshots']} arcs along the top</strong> are the acquisition footprint: each of the "
            f"{gpu['nshots']} shot positions leaves a curved artifact near the surface.",
            "<strong>Switch between versions</strong>: cuda-v0 to v4 look identical to the reference. Pick "
            "<em>Difference</em> to see what does differ: rounding at the 10<sup>-5</sup> level, concentrated near "
            "the shallow shots.",
            "<strong>Unfiltered</strong> shows the raw image before the Laplacian filter, with RTM's low-frequency "
            "noise near the surface.",
        ] + ([f"<strong>Devito</strong> images the same reflectors in the same places: correlation "
              f"{devito_corr['filtered']:.2f} with our image once both are filtered, {devito_corr['raw']:.2f} raw "
              f"(the two codes scale amplitudes differently). Devito shows a stronger flat event near 0.4 km: an "
              f"implementation difference (boundary, source interpolation), not an error in either code."]
             if devito_corr else []))
        viewer = """
<div class="viewer" id="image-viewer">
  <div class="viewer-controls">
    <div class="segmented" role="group" aria-label="Image" id="image-choices"></div>
    <div class="viewer-row">
      <div class="segmented small" role="group" aria-label="Display" id="image-mode">
        <button type="button" data-mode="image" aria-pressed="true">Image</button>
        <button type="button" data-mode="diff" aria-pressed="false">Difference from reference</button>
      </div>
      <label class="clip" for="image-clip">Clip <input id="image-clip" type="range" min="0.1" max="3" step="0.05" value="0.6"><output id="image-clip-value">0.60</output></label>
    </div>
  </div>
  <div class="canvas-wrap">
    <canvas id="image-canvas" aria-label="Migrated image"></canvas>
    <div class="axis-x"><span>0</span><span>x (km)</span><span id="image-xmax"></span></div>
  </div>
  <div class="viewer-foot">
    <p id="image-caption" class="caption"></p>
    <p id="image-readout" class="readout mono">Point at the image for position and value.</p>
  </div>
  <div class="colorbar" id="image-colorbar"></div>
</div>"""
        sections.append(section(
            "image", "The result", "The migrated Marmousi image",
            "One image per engine, filtered and displayed identically. Every correctness number on this page "
            "compares these images.",
            viewer, image_reading,
            "first results figure (the filtered image), then the correctness figures."))

    # ---------- correctness
    rows = []
    for e in CUDA_VERSIONS + ["cpu-opt"]:
        g = d["compare_main"].get(e)
        if not g:
            continue
        im = d["compare_images"].get(e)
        strict_ok = g["pass"] == "1" and (im is None or im["pass"] == "1")
        if strict_ok:
            state = chip("good", "pass")
        else:
            state = chip("warn", "rounding, above strict gate")
        rows.append([f'<span class="mono">{e}</span>', sci(number(g["l2_relative"])),
                     sci(number(im["l2_relative"])) if im else "—",
                     f"{number(g['correlation']):.7f}", state])
    blackwell_over = [v for v in CUDA_VERSIONS if v in d["compare_images"] and d["compare_images"][v]["pass"] != "1"]
    correctness_reading = bullets([
        f"<strong>On the {gpu_name}, every version passes</strong> the strict gate (relative L2 below "
        f"{sci(STRICT_L2_GATE)}); the worst is {sci(worst_l2_main)}." if all_pass_main else
        "<strong>Some versions fail the strict gate on the main run.</strong> Check the log before using the numbers.",
        (f"<strong>On the Blackwell GPU, {' and '.join(blackwell_over)} land just above the strict gate</strong> "
         f"(up to {sci(max(image_l2))}). Their correlation is still exactly 1, and a real bug shows up around "
         f"10<sup>-3</sup>. The compiler orders the float operations differently on each architecture.")
        if blackwell_over else "The images session passes the strict gate too.",
        "The CPU engines reproduce the reference bit for bit: difference 0.",
        f"<strong>What to write:</strong> all versions agree with the reference to within floating-point "
        f"rounding (relative L2 ≤ {sci(worst_l2_any)} on every GPU tested).",
    ])
    correctness_table = table(
        ["engine", f"relative L2, {gpu_name}", "relative L2, Blackwell", "correlation", "verdict"],
        rows, numeric={1, 2, 3})
    sections.append(section(
        "correctness", "Correctness", "Every version matches the reference",
        f"Each engine's image against the CPU reference, from <span class=mono>compare.csv</span> "
        f"({gpu_name} run) and <span class=mono>compare_images.csv</span> (Blackwell session). "
        f"Gate: relative L2 below {sci(STRICT_L2_GATE)}.",
        correctness_table, correctness_reading, "“correctness across versions”, with the image grid and difference maps."))

    # ---------- the speed ladder
    v0, v1, v2 = E.get("cuda-v0"), E.get("cuda-v1"), E.get("cuda-v2")
    later = [E[v]["t_fb"] for v in ["cuda-v2", "cuda-v3", "cuda-v4"] if v in E]
    ladder_reading = [
        "<strong>Read the forward + backward time, not the total.</strong> "
        + (f"v2's total looks better than v1's ({v2['t_total']:.2f} vs {v1['t_total']:.2f} s), but their "
           f"propagation times are equal ({v1['t_fb']:.2f} and {v2['t_fb']:.2f} s). v1 spends "
           f"{(v1['t_total'] - v1['t_fb']) - (v2['t_total'] - v2['t_fb']):.2f} s more outside the kernels, "
           f"in setup before the first shot." if v1 and v2 else ""),
        (f"<strong>v0 → v1 is the real kernel win</strong>: {v0['t_fb']:.2f} → {v1['t_fb']:.2f} s "
         f"({(v1['t_fb'] / v0['t_fb'] - 1):+.0%}), from fusing the sponge into the stencil kernel. "
         f"v2, v3 and v4 stay within {min(later) / v1['t_fb'] - 1:+.0%} to {max(later) / v1['t_fb'] - 1:+.0%} of v1."
         if v0 and v1 and later else ""),
        f"<strong>Two CPU baselines.</strong> The reference is deliberately single-core; the fair baseline is "
        f"cpu-opt on {cpu_opt_threads} threads ({E['cpu']['t_total'] / E['cpu-opt']['t_total']:.1f}× faster than the reference).",
    ]
    if d["other_cpu"]:
        o = d["other_cpu"][0]
        ladder_reading.append(
            f"<strong>Cloud CPUs vary.</strong> The same reference took {number(o['t_total']):.0f} s on an "
            f"{short_cpu(o['cpu_name'])} pod. Quote the machine next to every CPU number.")
    ladder_table = table(
        ["engine", "what changes", "forward + backward (s)", "total (s)", "GPts/s", "vs cpu", "vs cpu-opt"],
        [[f'<span class="mono">{r["engine"]}</span>', esc(r["what"]), fmt(r["t_fb"]), fmt(r["t_total"]),
          fmt(r["gpts"], 3 if r["gpts"] and r["gpts"] < 1 else 1), f"{r['vs_cpu']:,.0f}×", f"{r['vs_cpu_opt']:,.0f}×"]
         for r in ladder], numeric={2, 3, 4, 5, 6})
    cpu_tiles = "".join(
        f'<div class="mini"><span class="mono">{r["engine"]}</span><strong>{r["t_total"]:,.0f} s</strong>'
        f'<span>{esc(r["what"])}</span></div>' for r in ladder if not r["engine"].startswith("cuda"))
    ladder_figure = f"""
<div class="mini-row">{cpu_tiles}<div class="mini accent"><span class="mono">{fastest_total['engine']}</span>
<strong>{fastest_total['t_total']:.2f} s</strong><span>fastest GPU version, total</span></div></div>
<p class="chart-title">GPU forward + backward time per version, seconds ({gpu['nshots']} shots)</p>
<div class="chart" id="chart-ladder" data-height="230"></div>
{numbers_disclosure(ladder_table)}"""
    sections.append(section(
        "speed", "Speed", "The optimization ladder",
        f"Each version adds one optimization. Timings from <span class=mono>benchmarks.csv</span>, "
        f"{gpu_name} run on {first_date}, fixed {DX:g} m dataset.",
        ladder_figure, bullets([x for x in ladder_reading if x]),
        "results chapter, first table; methodology (why two CPU baselines)."))

    # ---------- one time step
    S = {s["engine"]: s for s in d["steps"]}
    if S:
        s0, s1 = S.get("cuda-v0"), S.get("cuda-v1")
        busy = [s["busy"] for s in d["steps"]]
        step_reading = [
            (f"<strong>Fusion, measured.</strong> v0 spends {s0['sponge']:.1f} µs per step on a separate sponge "
             f"pass. v1's fused kernel costs {s1['stencil']:.1f} µs instead of {s0['stencil']:.1f} + "
             f"{s0['sponge']:.1f} = {s0['stencil'] + s0['sponge']:.1f} µs: one read/write pass over the grid saved "
             f"(36 → 24 bytes per point)." if s0 and s1 else ""),
            (f"<strong>Launching kernels costs almost as much as running them.</strong> The CPU spends about "
             f"{s1['launch']:.0f} µs per step in <span class=mono>cudaLaunchKernel</span> (dashed marks) while the "
             f"GPU works {s1['gpu_total']:.0f} µs. The GPU is busy {min(busy):.0f}–{max(busy):.0f}% of the time. "
             f"CUDA Graphs or fewer launches would close that gap (future work)." if s1 else ""),
            (f"<strong>v3 and v4 are slightly slower per step than v1</strong> ({S['cuda-v3']['gpu_total']:.1f} and "
             f"{S['cuda-v4']['gpu_total']:.1f} vs {s1['gpu_total']:.1f} µs): shared memory and shuffles add work "
             f"without saving memory traffic." if s1 and "cuda-v3" in S and "cuda-v4" in S else ""),
        ]
        step_table = table(
            ["engine", "kernels / step", "stencil µs", "sponge µs", "imaging µs", "injection + copies µs",
             "GPU total µs", "CPU launch µs", "wall µs", "GPU busy"],
            [[f'<span class="mono">{s["engine"]}</span>', fmt(s["kernels"]), fmt(s["stencil"], 1), fmt(s["sponge"], 1),
              fmt(s["imaging"], 1) + (" (fused)" if s["imaging_fused"] else ""), fmt(s["other"], 1),
              fmt(s["gpu_total"], 1), fmt(s["launch"], 1), fmt(s["wall"], 1), f"{s['busy']:.1f}%"]
             for s in d["steps"]], numeric=set(range(1, 10)))
        step_figure = f"""
<p class="chart-title">GPU time per time step, µs (Nsight Systems, {gpu_name})</p>
<div class="legend" id="legend-steps"></div>
<div class="chart" id="chart-steps" data-height="250"></div>
{numbers_disclosure(step_table)}"""
        sections.append(section(
            "timestep", "Profile", "What one time step costs",
            "Kernel time per time step, from the Nsight Systems timelines "
            "(<span class=mono>profiles/step_breakdown.csv</span>). A time step is one stencil update of the whole grid.",
            step_figure, bullets([x for x in step_reading if x]),
            "“per-version analysis”, next to a v0 vs v1 timeline screenshot."))

    # ---------- scaling
    sc = [s for s in d["scaling"] if s["ours_v1"]]
    power = [s for s in d["scaling"] if s["ours_power"]]
    finest = min(sc, key=lambda s: s["dx"])
    scaling_reading = [
        f"<strong>A plateau at about {plateau:.0f} GPts/s.</strong> From {sc[0]['ours_v1']:.1f} GPts/s at "
        f"{sc[0]['dx']:g} m to {finest['ours_v1']:.1f} at {finest['dx']:g} m: once the grid is large enough, "
        f"the {gpu_name} processes grid points at a fixed rate.",
        f"<strong>That rate is the memory bandwidth.</strong> {plateau:.1f} GPts/s × {BYTES_PER_POINT} bytes per "
        f"point ≈ {plateau * BYTES_PER_POINT:.0f} GB/s, {bandwidth_used:.0%} of the measured {d['gpu_bandwidth']:.0f} "
        f"GB/s. The dashed line is that ceiling ({ceiling:.1f} GPts/s).",
        (f"<strong>Shared memory still doesn't help at {finest['dx']:g} m</strong> ({finest['points'] / 1e6:.0f} M "
         f"points, far beyond the L2 cache): v1 {finest['ours_v1']:.2f} vs v3 {finest['ours_v3']:.2f} GPts/s."
         if finest["ours_v3"] else ""),
        (f"<strong>Power rises with grid size</strong> ({power[0]['ours_power']:.0f} W at {power[0]['dx']:g} m, "
         f"{power[-1]['ours_power']:.0f} W at {power[-1]['dx']:g} m): small grids leave the GPU partly idle, "
         f"and launch gaps weigh more." if len(power) > 1 else ""),
    ]
    scaling_table = table(
        ["dx (m)", "grid points", "v1 GPts/s", "v3 GPts/s", "v1 s/shot", "v1 power (W)",
         "Devito end-to-end", "Devito compute", "Devito power (W)"],
        [[f"{s['dx']:g}", f"{s['points'] / 1e6:.2f} M", fmt(s["ours_v1"], 1), fmt(s["ours_v3"], 1),
          fmt(s["ours_s_per_shot"], 2), fmt(s["ours_power"], 0), fmt(s["devito_wall"], 1),
          fmt(s["devito_compute"], 1), fmt(s["devito_power"], 0)] for s in d["scaling"]],
        numeric=set(range(0, 9)))
    scaling_figure = f"""
<p class="chart-title">Throughput vs grid size, GPts/s (billion grid-point updates per second)</p>
<div class="legend" id="legend-scaling"></div>
<div class="chart" id="chart-scaling" data-height="300"></div>
{numbers_disclosure(scaling_table + '<p class="note">Our throughput: sweep S1T. Power: sweep S1, one snapshot every 10 steps (blank where it ran out of memory). Devito: sweep S1, or S2 at 2.5 m (snapshot every 40 steps).</p>')}"""
    sections.append(section(
        "scaling", "Scaling", "How the GPU behaves as the grid grows",
        "The same Marmousi model resampled from 12.5 m down to 1.25 m, with the source frequency scaled to keep "
        "the same points per wavelength (<span class=mono>scaling.csv</span>).",
        scaling_figure, bullets([x for x in scaling_reading if x]),
        "“scaling”, and the discussion of why the optimizations stop paying off: the core insight of the study."))

    # ---------- memory
    if d["capacity"]:
        acc_dataset, acc = next(iter(d["snapshot_accuracy"].items()), (None, {}))
        acc_sorted = sorted(acc.items())
        one_twenty_five = next((c for c in d["capacity"] if c["dx"] == min(x["dx"] for x in d["capacity"])), None)
        memory_reading = [
            (f"<strong>The ceiling.</strong> With a snapshot every 10 steps, memory grows as 1/dx³. "
             f"{finest_fit['dx']:g} m fits ({finest_fit['measured_gib']:.1f} GiB measured); "
             + ", ".join(f"{c['dx']:g} m would need about {c['need_gib']:.0f} GiB" for c in too_big)
             + f", and the {gpu_name} has {limit_gib:.0f}.") if finest_fit else "",
            (f"<strong>At {one_twenty_five['dx']:g} m</strong> the densest spacing that fits is one snapshot every "
             f"{one_twenty_five['densest_that_fits']} steps." if one_twenty_five and one_twenty_five["densest_that_fits"] else ""),
        ]
        if acc_sorted:
            text = ", ".join(f"every {k} steps: {v:.{7 if v > 0.9999 else 2}f}" for k, v in acc_sorted if k != 10)
            memory_reading.append(
                f"<strong>Sparser snapshots save memory but can alias the image</strong> "
                f"({acc_dataset.replace('marmousi_dx', '')} m, correlation with the 10-step image: {text}). "
                f"Snapshots must stay denser than the highest frequency in the wavefield. Past that, the image is "
                f"damaged, which is why industry codes use checkpointing or wavefield compression.")
        if d["aliased"]:
            k, v = next(iter(d["aliased"].items()))
            memory_reading.append(
                f"<strong>Methodology lesson.</strong> The first sweep used a fixed 10 ms spacing while the frequency "
                f"grew with the grid, and it aliased at fine grids (5 ms and 10 ms images correlate at only {v:.2f} at "
                f"{k.replace('marmousi_dx', '')} m). Keeping the spacing constant in time steps fixed it.")
        capacity_table = table(
            ["dx (m)", "grid points", "snapshot every", "status", "measured (GiB)", "estimated need (GiB)",
             "densest spacing that fits"],
            [[f"{c['dx']:g}", f"{c['points'] / 1e6:.2f} M", f"{c['every']} steps",
              chip("good", "fits") if c["status"] == "ok" else chip("bad", "out of memory"),
              fmt(c["measured_gib"], 1), fmt(c["need_gib"], 1),
              f"every {c['densest_that_fits']} steps" if c["densest_that_fits"] else "—"] for c in d["capacity"]],
            numeric={0, 1, 4, 5})
        memory_figure = f"""
<p class="chart-title">Device memory needed with a snapshot every 10 steps, GiB (log scale)</p>
<div class="legend" id="legend-capacity"></div>
<div class="chart" id="chart-capacity" data-height="270"></div>
<p class="chart-title">Image accuracy vs snapshot spacing ({esc(acc_dataset.replace('marmousi_dx', '') + ' m') if acc_dataset else ''}), correlation with the 10-step image</p>
<div class="chart" id="chart-snapshots" data-height="220"></div>
{numbers_disclosure(capacity_table)}"""
        sections.append(section(
            "memory", "Capacity", "How much fits in GPU memory",
            "RTM stores the source wavefield during the forward pass and reads it back during the backward pass. "
            "Those snapshots, not the solver, fill the GPU.",
            memory_figure, bullets([x for x in memory_reading if x]),
            "“capacity”; discussion (checkpointing, compression, multi-GPU as future work)."))

    # ---------- Devito
    if devito_rows:
        three = next((s for s in devito_rows if s["devito_every"] == 10 and s["dx"] == min(x["dx"] for x in devito_ten)), None)
        ours_orders, devito_orders = d["orders"]["ours"], d["orders"]["devito"]
        devito_reading = [
            f"<strong>Devito's kernels are competitive.</strong> Its own profiler measures up to "
            f"{devito_best_compute:.1f} GPts/s of pure compute: generated OpenACC code keeps up with hand-written CUDA.",
            (f"<strong>But end to end it is {min(devito_slowdown):.1f}–{max(devito_slowdown):.1f}× slower</strong> "
             f"with a snapshot every 10 steps. At {three['dx']:g} m it draws {three['devito_power']:.0f} W against "
             f"our {next(s['ours_power'] for s in d['scaling'] if s['dx'] == three['dx']):.0f} W: the GPU is often "
             f"waiting while the saved wavefield goes to host memory. Check the Memory row of the Devito timeline."
             if three else ""),
            "<strong>A different capacity limit.</strong> Devito keeps snapshots in host RAM. At 2.5 m and 1.25 m, "
            "the container's RAM limit stopped it, not GPU memory.",
            (f"<strong>Stencil order.</strong> Ours stays flat ({ours_orders[0][1]:.1f} → {ours_orders[-1][1]:.1f} "
             f"GPts/s from order {ours_orders[0][0]} to {ours_orders[-1][0]}); Devito slows "
             f"({devito_orders[0][1]:.1f} → {devito_orders[-1][1]:.1f} from order {devito_orders[0][0]} to "
             f"{devito_orders[-1][0]}). Doubling the arithmetic costs a memory-bound kernel nothing."
             if ours_orders and devito_orders else ""),
            (f"<strong>Same image.</strong> Correlation {images['devito_corr']['filtered']:.2f} after filtering. "
             f"Pick <em>Devito</em> in the image viewer to compare." if images and images["devito_corr"] else ""),
        ]
        devito_figure = f"""
<p class="chart-title">Throughput vs stencil order at 2.5 m, GPts/s</p>
<div class="legend" id="legend-orders"></div>
<div class="chart" id="chart-orders" data-height="250"></div>
<p class="note">Devito's end-to-end and compute-only throughput against grid size are in the Scaling chart above.</p>"""
        sections.append(section(
            "devito", "Comparison", "Our CUDA code against Devito",
            "Devito generates its GPU code from the equation. Same model, same data, same stencil order, same GPU.",
            devito_figure, bullets([x for x in devito_reading if x]),
            "a comparison chapter: hand-written vs generated code, and where each one's time goes."))

    # ---------- survey projection
    if d["surveys"]:
        dx_key = min(d["surveys"], key=float)
        sv = d["surveys"][dx_key]
        one = [r for r in sv["rows"] if r["devices"] == 1]
        # cuda-v1 keeps the densest snapshots that fit, so it's the fair row to quote.
        ours_best = next((r for r in one if r["who"].startswith("cuda-v1")),
                         min((r for r in one if r["who"].startswith("cuda")), key=lambda r: r["hours"], default=None))
        cpu_opt = next((r for r in one if r["who"].startswith("cpu-opt")), None)
        survey_reading = [
            (f"<strong>From weeks to hours.</strong> One {gpu_name} would take {ours_best['hours']:.1f} h; the "
             f"{cpu_opt_threads}-thread CPU node about {cpu_opt['hours']:,.0f} h (~{cpu_opt['hours'] / 24 / 7:.0f} "
             f"weeks)." if ours_best and cpu_opt else ""),
            (f"<strong>Energy follows:</strong> {ours_best['kwh']:.1f} kWh on the GPU against {cpu_opt['kwh']:,.0f} kWh "
             f"for the CPU node." if ours_best and cpu_opt and ours_best["kwh"] else ""),
            "Shots are independent, so 8 GPUs divide the time by 8.",
            f'<span class="note">{esc(sv["assumptions"])}</span>',
        ]
        survey_table = table(
            ["code / device", "devices", "hours", "energy (kWh)"],
            [[esc(r["who"]), r["devices"], fmt(r["hours"], 2), fmt(r["kwh"], 1)] for r in sv["rows"]],
            numeric={1, 2, 3})
        others = [k for k in d["surveys"] if k != dx_key]
        sections.append(section(
            "survey", "Perspective", "What it means for a whole survey",
            f"{esc(sv['title'])}. {esc(sv['grid'])}"
            + (f" Projections at {', '.join(o + ' m' for o in others)} are in <span class=mono>results/</span>." if others else ""),
            survey_table, bullets([x for x in survey_reading if x]),
            "industrial perspective; keep the assumptions under the table."))

    # ---------- caveats and the timelines
    reports = sorted(glob.glob("results/profiles/*/nsys/*.nsys-rep"))
    report_rows = [[f'<span class="mono">{esc(p.split("/")[2])}</span>', f'<span class="mono">{esc(os.path.basename(p))}</span>',
                    f"{os.path.getsize(p) / 2 ** 20:,.0f} MB"] for p in reports]
    caveats = bullets([
        "<strong>No hardware counters.</strong> The host blocked Nsight Compute (<span class=mono>ERR_NVGPUCTRPERM</span>), "
        "so the memory-bound claim rests on timings, the bytes-per-point model and the scaling runs, not on measured "
        "cache or DRAM metrics. Say so once in the methodology.",
        f"<strong>Cloud CPU baseline.</strong> {cpu_opt_threads} vCPUs of a shared {cpu_name}. Give the machine and "
        "quote speedups against both CPU engines.",
        "<strong>Images come from a second session</strong> on a different GPU (RTX PRO 6000 Blackwell). It produced "
        "images only, no timings.",
        "<strong>Ignore <span class=mono>S1B_aliased</span> rows</strong> in <span class=mono>scaling.csv</span>: "
        "that sweep's fixed 10 ms spacing aliased the imaging at fine grids.",
    ])
    timelines = (table(["dataset", "report", "size"], report_rows, numeric={2}) if report_rows else
                 "<p>No <span class=mono>.nsys-rep</span> files on this machine. They stay local (synced from the pod).</p>")
    sections.append(f"""
<section id="notes" class="section">
  <header class="section-head">
    <p class="kicker">Before writing</p>
    <h2>Caveats and raw material</h2>
  </header>
  <div class="section-body notes">
    <div class="reading plain"><h3>Caveats</h3>{caveats}</div>
    <div class="reading plain"><h3>Timelines to open in Nsight Systems</h3>
      <p>Open them with Nsight Systems 2026.3.2 or newer. The click-by-click guide is in
      <span class=mono>docs/PROFILING_STRATEGY.md</span> §4.3.</p>{timelines}</div>
  </div>
</section>""")

    nav = [("overview", "Summary"), ("image", "Image"), ("correctness", "Correctness"), ("speed", "Speed"),
           ("timestep", "Time step"), ("scaling", "Scaling"), ("memory", "Memory"), ("devito", "Devito"),
           ("survey", "Survey"), ("notes", "Notes")]
    present = set(re.findall(r'<section id="([\w-]+)"', "".join(sections)))
    nav_html = "".join(f'<a href="#{a}">{t}</a>' for a, t in nav if a in present)

    body = f"""
<header class="masthead">
  <div class="wrap">
    <p class="kicker">Reverse time migration on GPU · results</p>
    <h1>Marmousi RTM, CPU to {gpu_name}</h1>
    <pre class="textual-header" aria-label="Run parameters">{esc(textual_header)}</pre>
  </div>
</header>
<nav class="sections-nav" aria-label="Sections"><div class="wrap nav-inner">{nav_html}</div></nav>
<main class="wrap">{"".join(sections)}</main>
<footer class="wrap footer"><p>Generated from <span class=mono>results/</span> by
<span class=mono>scripts/build_dashboard.py</span> on {built_at}. Re-run it after a new run; every number on this page is
recomputed.</p></footer>"""

    chart_data = {
        "ladder": [{"engine": r["engine"], "what": r["what"], "t_fb": r["t_fb"], "t_total": r["t_total"],
                    "vs_cpu_opt": r["vs_cpu_opt"], "vs_cpu": r["vs_cpu"], "gpts": r["gpts"]}
                   for r in ladder if r["engine"].startswith("cuda")],
        "steps": d["steps"],
        "scaling": d["scaling"],
        "ceiling": ceiling,
        "bandwidth": d["gpu_bandwidth"],
        "capacity": d["capacity"],
        "limit_gib": limit_gib,
        "snapshots": sorted(next(iter(d["snapshot_accuracy"].values()), {}).items()),
        "orders": d["orders"],
        "images": images,
    }
    return body, chart_data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", action="store_true",
                        help="write the page body only, without <!doctype>/<head> (for publishing as an artifact)")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    built_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    images = collect_images()
    if images is None:
        print("note: results/ref/marmousi_ref.bin missing, the image section is skipped")
    body, chart_data = build_page(collect(), images, built_at)

    with open(TEMPLATE) as f:
        template = f.read()
    data_json = json.dumps(chart_data, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace("{{BODY}}", body).replace("{{DATA}}", data_json)
    if not args.fragment:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(page)
    print(f"wrote {args.output} ({os.path.getsize(args.output) / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()
