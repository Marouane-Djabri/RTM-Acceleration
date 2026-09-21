#!/usr/bin/env python3
"""Charts + docs/RESULTS.md from the benchmark CSVs (docs/OPTIMIZATION_PLAN.md §2.7).

    plot_benchmark.py --benchmarks results/benchmarks.csv --compare results/compare.csv \
                      --bandwidth results/ref/bandwidth.csv --out results/plots \
                      --results-md docs/RESULTS.md

Charts written to --out, one set per (host, dataset):
    speedup_<dataset>_<host>.png     speedup vs `cpu` per engine (log), dashed line at cpu-opt
    stages_<dataset>_<host>.png      stacked forward / backward / imaging / H2D / D2H
    bandwidth_<dataset>_<host>.png   achieved stencil GB/s as % of the host's measured peak

For every engine the LAST row in the CSV wins (the most recent run).
"""
import argparse
import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bytes the stencil must move per grid point per time step at minimum:
# read p_prev, p_cur, vdt2 and write p_next (src/rtm/forward.cpp header).
# Fused engines are judged against this same floor, so "% of peak" says how
# close they come to the roofline for the useful traffic.
STENCIL_BYTES_PER_POINT_STEP = 16.0

# Ladder order for the x axis; unknown engines go after these, alphabetically.
ENGINE_ORDER = ["cpu", "cpu-opt", "cuda-v0", "cuda-v1", "cuda-v2", "cuda-v3",
                "cuda-v4", "cuda-v5", "cuda-v6", "cuda-multi"]


def read_latest(path, key_fields):
    rows = OrderedDict()
    if not path or not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[tuple(row[k] for k in key_fields)] = row
    return rows


def engine_sort_key(name):
    base = name.split("_gpus")[0]
    return (ENGINE_ORDER.index(base) if base in ENGINE_ORDER else len(ENGINE_ORDER), name)


def engine_label(row):
    gpus = int(row.get("gpus", "0") or 0)
    return row["engine"] if gpus <= 1 else f"{row['engine']} x{gpus}"


def is_gpu_engine(row):
    return int(row.get("gpus", "0") or 0) >= 1


def achieved_gbps(row):
    points = (float(row["nx"]) + 2 * float(row["nb"])) * (float(row["nz"]) + 2 * float(row["nb"]))
    steps = 2.0 * float(row["nt"]) * float(row["nshots"])          # forward + backward
    seconds = float(row["t_forward"]) + float(row["t_backward"])
    if seconds <= 0:
        return 0.0
    return STENCIL_BYTES_PER_POINT_STEP * points * steps / seconds / 1e9


def peak_gbps_for(row, bandwidth_rows):
    kind = "gpu" if is_gpu_engine(row) else "cpu"
    for (host, device_kind), b in bandwidth_rows.items():
        if host == row["host"] and device_kind == kind:
            return float(b["gbps"]), b["device_name"]
    return None, None


def plot_speedup(rows, gates, out_path, title):
    labels, speedups, colors, hatches = [], [], [], []
    cpu_total = float(rows["cpu"]["t_total"]) if "cpu" in rows else None
    cpu_opt_total = float(rows["cpu-opt"]["t_total"]) if "cpu-opt" in rows else None
    for name in sorted(rows, key=engine_sort_key):
        row = rows[name]
        total = float(row["t_total"])
        if cpu_total is None or total <= 0:
            continue
        gate = gates.get(name)
        passed = gate is None or gate["pass"] == "1"
        labels.append(engine_label(row))
        speedups.append(cpu_total / total)
        colors.append("#c0504d" if not passed else ("#4f81bd" if is_gpu_engine(row) else "#9bbb59"))
        hatches.append("//" if not passed else "")
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.5))
    bars = ax.bar(labels, speedups, color=colors)
    for bar, hatch, value in zip(bars, hatches, speedups):
        bar.set_hatch(hatch)
        ax.annotate(f"{value:.1f}x", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs cpu (reference, log scale)")
    ax.axhline(1.0, color="gray", lw=0.8)
    if cpu_opt_total:
        ax.axhline(cpu_total / cpu_opt_total, color="#9bbb59", ls="--", lw=1,
                   label="best CPU (cpu-opt)")
        ax.legend(loc="upper left")
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_stages(rows, out_path, title):
    stages = [("t_forward", "forward"), ("t_backward", "backward (incl. fused imaging)"),
              ("t_imaging", "imaging"), ("t_h2d", "H2D"), ("t_d2h", "D2H")]
    names = sorted(rows, key=engine_sort_key)
    if not names:
        return False
    labels = [engine_label(rows[n]) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.5))
    bottom = [0.0] * len(names)
    for field, stage_label in stages:
        values = [float(rows[n][field]) for n in names]
        ax.bar(labels, values, bottom=bottom, label=stage_label)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_yscale("log")
    ax.set_ylabel("seconds (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_bandwidth(rows, bandwidth_rows, out_path, title):
    labels, percents, texts = [], [], []
    for name in sorted(rows, key=engine_sort_key):
        row = rows[name]
        peak, device = peak_gbps_for(row, bandwidth_rows)
        if not peak:
            continue
        achieved = achieved_gbps(row)
        labels.append(engine_label(row))
        percents.append(100.0 * achieved / peak)
        texts.append(f"{achieved:.0f} / {peak:.0f} GB/s")
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.5))
    bars = ax.bar(labels, percents, color="#8064a2")
    for bar, text, value in zip(bars, texts, percents):
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(100, color="gray", ls="--", lw=0.8, label="measured peak (probe)")
    ax.set_ylabel("stencil traffic as % of measured peak bandwidth")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def write_results_md(groups, gates, bandwidth_rows, plots_dir, out_path):
    lines = ["# Results", "",
             "Generated by `scripts/plot_benchmark.py` from `results/benchmarks.csv`, "
             "`results/compare.csv` and `results/ref/bandwidth.csv`. "
             "Speedups are against the CPU reference (`cpu`) measured on the same host.", ""]
    for (host, dataset), rows in groups.items():
        cpu_total = float(rows["cpu"]["t_total"]) if "cpu" in rows else None
        cpu_opt_total = float(rows["cpu-opt"]["t_total"]) if "cpu-opt" in rows else None
        sample = next(iter(rows.values()))
        lines += [f"## {dataset} on `{host}`", "",
                  f"CPU: {sample['cpu_name']} — GPU: "
                  f"{next((r['gpu_name'] for r in rows.values() if is_gpu_engine(r)), 'none')}", "",
                  "| engine | threads/GPUs | total s | vs cpu | vs cpu-opt | GFLOP/s | GB/s | % peak | L2rel | gate |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name in sorted(rows, key=engine_sort_key):
            row = rows[name]
            total = float(row["t_total"])
            vs_cpu = f"{cpu_total / total:.1f}x" if cpu_total and total > 0 else "—"
            vs_opt = f"{cpu_opt_total / total:.2f}x" if cpu_opt_total and total > 0 else "—"
            peak, _ = peak_gbps_for(row, bandwidth_rows)
            gbps = achieved_gbps(row)
            pct = f"{100 * gbps / peak:.0f}%" if peak else "—"
            gate = gates.get(name)
            l2 = f"{float(gate['l2_relative']):.2e}" if gate else "—"
            verdict = ("PASS" if gate["pass"] == "1" else "**FAIL**") if gate else "not gated"
            units = f"{row['threads']} thr" if not is_gpu_engine(row) else f"{row['gpus']} GPU"
            lines.append(f"| {engine_label(row)} | {units} | {total:.2f} | {vs_cpu} | {vs_opt} | "
                         f"{float(row['stencil_gflops']):.1f} | {gbps:.0f} | {pct} | {l2} | {verdict} |")
        lines += ["", f"![speedup]({os.path.relpath(os.path.join(plots_dir, f'speedup_{dataset}_{host}.png'), os.path.dirname(out_path))})",
                  f"![stages]({os.path.relpath(os.path.join(plots_dir, f'stages_{dataset}_{host}.png'), os.path.dirname(out_path))})",
                  f"![bandwidth]({os.path.relpath(os.path.join(plots_dir, f'bandwidth_{dataset}_{host}.png'), os.path.dirname(out_path))})", ""]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--benchmarks", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--bandwidth", default=None)
    p.add_argument("--out", default="results/plots")
    p.add_argument("--results-md", default=None)
    a = p.parse_args()

    bench = read_latest(a.benchmarks, ("host", "engine", "dataset", "gpus"))
    compare = read_latest(a.compare, ("host", "engine", "dataset"))
    bandwidth = read_latest(a.bandwidth, ("host", "device_kind"))
    if not bench:
        raise SystemExit(f"no rows in {a.benchmarks}")
    os.makedirs(a.out, exist_ok=True)

    # group by (host, dataset); key inside a group = engine name (+ _gpusN)
    groups = OrderedDict()
    for (host, engine, dataset, gpus), row in bench.items():
        name = engine if int(gpus or 0) <= 1 else f"{engine}_gpus{gpus}"
        groups.setdefault((host, dataset), OrderedDict())[name] = row
    gates_by_group = {}
    for (host, engine, dataset), row in compare.items():
        gates_by_group.setdefault((host, dataset), {})[engine] = row

    for (host, dataset), rows in groups.items():
        gates = gates_by_group.get((host, dataset), {})
        suffix = f"{dataset}_{host}"
        title = f"{dataset} on {host}"
        if plot_speedup(rows, gates, os.path.join(a.out, f"speedup_{suffix}.png"), title):
            print(f"wrote {a.out}/speedup_{suffix}.png")
        if plot_stages(rows, os.path.join(a.out, f"stages_{suffix}.png"), title):
            print(f"wrote {a.out}/stages_{suffix}.png")
        if plot_bandwidth(rows, bandwidth, os.path.join(a.out, f"bandwidth_{suffix}.png"), title):
            print(f"wrote {a.out}/bandwidth_{suffix}.png")

    if a.results_md:
        all_gates = {}
        for g in gates_by_group.values():
            all_gates.update(g)
        write_results_md(groups, all_gates, bandwidth, a.out, a.results_md)
        print(f"wrote {a.results_md}")


if __name__ == "__main__":
    main()
