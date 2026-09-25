#!/usr/bin/env python3
"""Turn the profiler exports into the report's tables and charts
(docs/PROFILING_STRATEGY.md change M4). Runs on the pod or locally after
scripts/local/sync_from_pod.sh; only reads CSV/text exports, never the GUI files.

    python3 scripts/profile_extract.py

Reads
    results/profiles/<dataset>/ncu/*_raw.csv + *.meta     (scripts/pod/profile_ncu.sh)
    results/profiles/<dataset>/nsys/*_cuda_*_sum.csv       (scripts/pod/profile_nsys.sh)
    results/scaling.csv, results/compare_scaling.csv       (sweeps S1, S1B, S2)
    results/ref/bandwidth.csv                              (measured GPU bandwidth)
Writes
    results/profiles/kernel_metrics.csv     D3: the same metrics for every kernel
    results/profiles/step_breakdown.csv     D2: GPU time per time step, by kernel family
    results/profiles/SUMMARY.md             both tables in Markdown, for the report
    results/plots/profiling/*.png           D2, D3, D4 (ladder), D7, D8, D11 (sweeps)
Every chart is skipped silently when its data does not exist yet.
"""
import csv
import glob
import os
import re
from collections import OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILES = "results/profiles"
PLOTS = "results/plots/profiling"
LADDER_DATASET = "marmousi12"
LADDER_ENGINES = ["cuda-v0", "cuda-v1", "cuda-v2", "cuda-v3", "cuda-v4"]

# Metric name in ncu's raw page -> our column name (docs/PROFILING_STRATEGY.md §4.6).
NCU_METRICS = OrderedDict([
    ("duration_ns", "gpu__time_duration.sum"),
    ("dram_pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram_read_bytes", "dram__bytes_read.sum"),
    ("dram_write_bytes", "dram__bytes_write.sum"),
    ("memory_pct", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("achieved_occupancy_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("theoretical_occupancy_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("registers_per_thread", "launch__registers_per_thread"),
    ("shared_mem_per_block_bytes", "launch__shared_mem_per_block_static"),
    ("waves_per_sm", "launch__waves_per_multiprocessor"),
    ("l1_hit_pct", "l1tex__t_sector_hit_rate.pct"),
    ("l2_hit_pct", "lts__t_sector_hit_rate.pct"),
    ("global_load_sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
    ("global_load_requests", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"),
    ("shared_bank_conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
    ("threads_per_instruction", "smsp__thread_inst_executed_per_inst_executed.ratio"),
    ("branch_uniform_pct", "smsp__sass_average_branch_targets_threads_uniform.pct"),
    ("warp_instructions", "smsp__inst_executed.sum"),
    ("fadd", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"),
    ("fmul", "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"),
    ("ffma", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"),
    ("sm_count", "device__attribute_multiprocessor_count"),
    ("sm_clock_hz", "sm__cycles_elapsed.avg.per_second"),
    ("cc_major", "device__attribute_compute_capability_major"),
    ("cc_minor", "device__attribute_compute_capability_minor"),
    ("dram_peak_bytes_per_cycle", "dram__bytes.sum.peak_sustained"),
    ("dram_clock_hz", "dram__cycles_elapsed.avg.per_second"),
])
STALL_PATTERN = re.compile(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")

# FP32 lanes per SM, by compute capability (for the FP32 roof of the roofline).
FP32_LANES_PER_SM = {(7, 0): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128, (8, 7): 128,
                     (8, 9): 128, (9, 0): 128, (10, 0): 128, (12, 0): 128}

KERNEL_METRIC_COLUMNS = [
    "dataset", "engine", "capture", "order", "cache", "kernel", "launches", "grid_points",
    "duration_us", "dram_pct", "effective_bandwidth_gbs", "dram_bytes_per_point",
    "memory_pct", "sm_pct", "achieved_occupancy_pct", "theoretical_occupancy_pct",
    "registers_per_thread", "shared_mem_per_block_bytes", "waves_per_sm", "l1_hit_pct",
    "l2_hit_pct", "sectors_per_global_load", "shared_bank_conflicts",
    "warp_execution_efficiency_pct", "branch_uniform_pct", "thread_instructions_per_point",
    "gflops", "arithmetic_intensity", "gpoints_per_s", "top_stalls",
    "peak_dram_gbs", "peak_fp32_gflops"]


# =============================================================================
# Small readers
# =============================================================================
def read_key_values(path):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                values[key] = value
    return values


def to_float(text):
    if text is None:
        return None
    text = str(text).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grid_of(velocity_path, nb):
    """(interior points, extended points) of a velocity model, from its .hdr."""
    header = read_key_values(velocity_path + ".hdr")
    if "nx" not in header:
        return None, None
    nx, nz, nb = int(header["nx"]), int(header["nz"]), int(nb)
    return nx * nz, (nx + 2 * nb) * (nz + 2 * nb)


def measured_gpu_bandwidth_gbs():
    """The most recent GPU row: the probe appends one per pod, the latest is this run's."""
    gpu_rows = [r for r in read_csv_rows("results/ref/bandwidth.csv") if r.get("device_kind") == "gpu"]
    return to_float(gpu_rows[-1]["gbps"]) if gpu_rows else None


# =============================================================================
# Nsight Compute: one row per report
# =============================================================================
def read_ncu_raw(path):
    """Launch rows of `ncu --page raw --csv --print-units base` (row 2 = units)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    rows = [r for r in rows if r and not r[0].startswith("==")]
    if len(rows) < 3:
        return []
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[2:]]


def summarize_ncu_report(raw_path):
    meta = read_key_values(raw_path.replace("_raw.csv", ".meta"))
    launches = read_ncu_raw(raw_path)
    if not meta or not launches:
        return None

    def mean_of(metric):
        values = [to_float(l.get(metric)) for l in launches]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    m = {name: mean_of(metric) for name, metric in NCU_METRICS.items()}
    interior_points, extended_points = grid_of(meta.get("velocity", ""), meta.get("nb", 0))
    points = extended_points

    row = {c: "" for c in KERNEL_METRIC_COLUMNS}
    row.update(dataset=meta.get("dataset", ""), engine=meta.get("engine", ""),
               capture=meta.get("capture", ""), order=meta.get("order", ""),
               cache=meta.get("cache", ""), kernel=launches[0].get("Kernel Name", ""),
               launches=len(launches), grid_points=points or "")

    def put(column, value, digits=3):
        if value is not None:
            row[column] = round(value, digits)

    duration_s = m["duration_ns"] * 1e-9 if m["duration_ns"] else None
    put("duration_us", m["duration_ns"] / 1e3 if m["duration_ns"] else None)
    for name in ("dram_pct", "memory_pct", "sm_pct", "achieved_occupancy_pct",
                 "theoretical_occupancy_pct", "registers_per_thread", "shared_mem_per_block_bytes",
                 "waves_per_sm", "l1_hit_pct", "l2_hit_pct", "shared_bank_conflicts",
                 "branch_uniform_pct"):
        put(name, m[name])

    dram_bytes = None
    if m["dram_read_bytes"] is not None and m["dram_write_bytes"] is not None:
        dram_bytes = m["dram_read_bytes"] + m["dram_write_bytes"]
    if dram_bytes is not None and duration_s:
        put("effective_bandwidth_gbs", dram_bytes / duration_s / 1e9)
    if dram_bytes is not None and points:
        put("dram_bytes_per_point", dram_bytes / points)
    if m["global_load_sectors"] and m["global_load_requests"]:
        put("sectors_per_global_load", m["global_load_sectors"] / m["global_load_requests"])
    if m["threads_per_instruction"] is not None:
        put("warp_execution_efficiency_pct", 100.0 * m["threads_per_instruction"] / 32.0)
    if m["warp_instructions"] and m["threads_per_instruction"] and points:
        put("thread_instructions_per_point", m["warp_instructions"] * m["threads_per_instruction"] / points)

    flops = None
    if m["fadd"] is not None and m["fmul"] is not None and m["ffma"] is not None:
        flops = m["fadd"] + m["fmul"] + 2.0 * m["ffma"]
    if flops is not None and duration_s:
        put("gflops", flops / duration_s / 1e9)
    if flops is not None and dram_bytes:
        put("arithmetic_intensity", flops / dram_bytes, 4)
    if points and duration_s:
        put("gpoints_per_s", points / duration_s / 1e9)

    stalls = []
    for column, value in launches[0].items():
        match = STALL_PATTERN.search(column)
        if match:
            values = [to_float(l.get(column)) for l in launches]
            values = [v for v in values if v is not None]
            if values:
                stalls.append((sum(values) / len(values), match.group(1)))
    stalls.sort(reverse=True)
    row["top_stalls"] = "; ".join(f"{name} {value:.2f}" for value, name in stalls[:3])

    # Roofs of this GPU. The memory roof is the bandwidth our probe MEASURED
    # (achievable), falling back to the theoretical peak ncu reports.
    peak_dram = measured_gpu_bandwidth_gbs()
    if peak_dram is None and m["dram_peak_bytes_per_cycle"] and m["dram_clock_hz"]:
        clock = m["dram_clock_hz"] * (1e9 if m["dram_clock_hz"] < 1e6 else 1.0)
        peak_dram = m["dram_peak_bytes_per_cycle"] * clock / 1e9
    put("peak_dram_gbs", peak_dram, 1)
    if m["sm_count"] and m["sm_clock_hz"] and m["cc_major"] is not None:
        clock = m["sm_clock_hz"] * (1e9 if m["sm_clock_hz"] < 1e6 else 1.0)
        lanes = FP32_LANES_PER_SM.get((int(m["cc_major"]), int(m["cc_minor"] or 0)), 128)
        put("peak_fp32_gflops", m["sm_count"] * clock * 2 * lanes / 1e9, 1)
    return row


def collect_kernel_metrics():
    rows = []
    for raw_path in sorted(glob.glob(f"{PROFILES}/*/ncu/*_raw.csv")):
        row = summarize_ncu_report(raw_path)
        if row:
            rows.append(row)
    if rows:
        with open(f"{PROFILES}/kernel_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=KERNEL_METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {PROFILES}/kernel_metrics.csv ({len(rows)} kernel reports)")
    return rows


# =============================================================================
# Nsight Systems: GPU time per time step, by kernel family (D2)
# =============================================================================
def kernel_family(name):
    if "_image" in name:
        return "stencil + imaging (fused)"
    if "k_fd_time_step" in name:
        return "stencil"
    if "k_sponge" in name:
        return "sponge"
    if "k_imaging" in name or "k_image_unique" in name:
        return "imaging"
    if "k_inject" in name or "k_record" in name or "k_mark" in name:
        return "injection / recording"
    return "other"


FAMILIES = ["stencil", "stencil + imaging (fused)", "sponge", "imaging",
            "injection / recording", "other", "memcpy / memset"]


def column_containing(row, text):
    return next((k for k in row if text in k), None)


def collect_step_breakdown(dataset=LADDER_DATASET):
    nsys_dir = f"{PROFILES}/{dataset}/nsys"
    runs = read_csv_rows(f"{nsys_dir}/runs_under_nsys.csv")
    rows = []
    for engine in LADDER_ENGINES:
        kernels = read_csv_rows(f"{nsys_dir}/{engine}_cuda_gpu_kern_sum.csv")
        if not kernels:
            continue
        run = next((r for r in reversed(runs) if r["engine"] == engine), None)
        if not run:
            continue
        steps = int(run["nt"]) * 2 * int(run["nshots"])
        time_key = column_containing(kernels[0], "Total Time")
        by_family = defaultdict(float)
        launches = 0
        count_key = column_containing(kernels[0], "Instances")
        for k in kernels:
            by_family[kernel_family(k["Name"])] += float(k[time_key]) * 1e-9
            launches += int(float(k[count_key])) if count_key else 0
        memops = read_csv_rows(f"{nsys_dir}/{engine}_cuda_gpu_mem_time_sum.csv")
        if memops:
            mem_key = column_containing(memops[0], "Total Time")
            by_family["memcpy / memset"] = sum(float(r[mem_key]) for r in memops) * 1e-9

        api = read_csv_rows(f"{nsys_dir}/{engine}_cuda_api_sum.csv")
        launch_seconds = 0.0
        if api:
            api_key = column_containing(api[0], "Total Time")
            launch_seconds = sum(float(r[api_key]) for r in api if "LaunchKernel" in r["Name"]) * 1e-9
        nvtx = read_csv_rows(f"{nsys_dir}/{engine}_nvtx_sum.csv")
        migrate_seconds = None
        if nvtx:
            nvtx_key = column_containing(nvtx[0], "Total Time")
            name_key = "Range" if "Range" in nvtx[0] else column_containing(nvtx[0], "Name")
            for r in nvtx:
                if r[name_key].strip(":").strip() == "migrate":
                    migrate_seconds = float(r[nvtx_key]) * 1e-9

        gpu_seconds = sum(by_family.values())
        row = {"engine": engine, "steps": steps, "kernel_launches_per_step": round(launches / steps, 2)}
        for family in FAMILIES:
            row[f"{family} us/step"] = round(by_family.get(family, 0.0) / steps * 1e6, 3)
        row["gpu total us/step"] = round(gpu_seconds / steps * 1e6, 3)
        row["cudaLaunchKernel us/step (CPU)"] = round(launch_seconds / steps * 1e6, 3)
        row["migrate wall us/step"] = round(migrate_seconds / steps * 1e6, 3) if migrate_seconds else ""
        row["gpu busy %"] = round(100 * gpu_seconds / migrate_seconds, 1) if migrate_seconds else ""
        rows.append(row)
    if rows:
        with open(f"{PROFILES}/step_breakdown.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {PROFILES}/step_breakdown.csv ({len(rows)} engines)")
    return rows


# =============================================================================
# Charts
# =============================================================================
def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/{name}", dpi=150)
    plt.close(fig)
    print(f"wrote {PLOTS}/{name}")


def plot_step_breakdown(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r["engine"] for r in rows]
    bottom = [0.0] * len(rows)
    for family in FAMILIES:
        values = [r[f"{family} us/step"] for r in rows]
        if any(values):
            ax.bar(labels, values, bottom=bottom, label=family)
            bottom = [b + v for b, v in zip(bottom, values)]
    walls = [r["migrate wall us/step"] for r in rows]
    if all(walls):
        ax.plot(labels, walls, "k_", markersize=30, mew=2, label="wall time per step")
    ax.set_ylabel("µs per time step")
    ax.set_title(f"GPU time per time step ({LADDER_DATASET}, Nsight Systems)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "D2_step_breakdown.png")


def ladder_rows(kernel_rows, capture, cache="all"):
    by_engine = {r["engine"]: r for r in kernel_rows
                 if r["dataset"] == LADDER_DATASET and r["capture"] == capture and r["cache"] == cache}
    return [by_engine[e] for e in LADDER_ENGINES if e in by_engine]


def plot_kernel_metrics(kernel_rows):
    rows = ladder_rows(kernel_rows, "stencil")
    if not rows:
        return
    panels = [("duration_us", "duration (µs, base clock)"), ("dram_pct", "DRAM throughput (% peak)"),
              ("dram_bytes_per_point", "DRAM bytes per grid point"), ("l2_hit_pct", "L2 hit rate (%)"),
              ("achieved_occupancy_pct", "achieved occupancy (%)"),
              ("thread_instructions_per_point", "instructions per grid point")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    labels = [r["engine"] for r in rows]
    for ax, (column, title) in zip(axes.flat, panels):
        values = [float(r[column]) if r[column] != "" else 0.0 for r in rows]
        ax.bar(labels, values, color="#4f81bd")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Main stencil kernel per version ({LADDER_DATASET}, Nsight Compute, cold caches)")
    save(fig, "D3_kernel_metrics.png")


def plot_roofline(rows, title, name, label_of):
    rows = [r for r in rows if r["arithmetic_intensity"] != "" and r["gflops"] != ""]
    if not rows:
        return
    peak_dram = next((float(r["peak_dram_gbs"]) for r in rows if r["peak_dram_gbs"] != ""), None)
    peak_fp32 = next((float(r["peak_fp32_gflops"]) for r in rows if r["peak_fp32_gflops"] != ""), None)
    fig, ax = plt.subplots(figsize=(7, 5))
    intensities = [10 ** (i / 20) for i in range(-60, 61)]    # 0.001 .. 1000 FLOP/byte
    if peak_dram:
        roof = [min(peak_dram * x, peak_fp32 or float("inf")) for x in intensities]
        ax.plot(intensities, roof, "k-", lw=1.5,
                label=f"roof: {peak_dram:.0f} GB/s (measured)" + (f", {peak_fp32 / 1000:.1f} TFLOP/s FP32" if peak_fp32 else ""))
    for index, r in enumerate(rows):
        x, y = float(r["arithmetic_intensity"]), float(r["gflops"])
        ax.plot(x, y, "o", markersize=8)
        # Stagger labels: ladder points often sit almost on top of each other.
        ax.annotate(label_of(r), (x, y), textcoords="offset points", xytext=(8, 12 * index - 20),
                    fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FP32 FLOP / DRAM byte)")
    ax.set_ylabel("achieved GFLOP/s (base clock)")
    ax.set_title(title)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, name)


def plot_capacity(scaling):
    rows = [r for r in scaling if r["code"] == "ours" and r["sweep"] in ("S1", "S1B")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    total_gib = next((float(r["gpu_total_mib"]) / 1024 for r in rows if r["gpu_total_mib"]), None)
    # Estimate from the allocation formula (§5.1), so OOM points still show how much was needed.
    for snapshot_ms in sorted({int(r["snapshot_ms"]) for r in rows}):
        points, estimated, measured_x, measured_y, oom_x, oom_y = [], [], [], [], [], []
        for r in sorted((r for r in rows if int(r["snapshot_ms"]) == snapshot_ms),
                        key=lambda r: int(r["nx"]) * int(r["nz"])):
            nx, nz, nb, nt = int(r["nx"]), int(r["nz"]), int(r["nb"]), int(r["nt"])
            nsnap = (nt - 1) // int(r["store_interval"]) + 1
            need_gib = (nsnap * nx * nz + 6 * (nx + 2 * nb) * (nz + 2 * nb)) * 4 / 2 ** 30
            points.append(nx * nz)
            estimated.append(need_gib)
            if r["status"] == "ok" and r["device_mem_mib_peak"]:
                measured_x.append(nx * nz)
                measured_y.append(float(r["device_mem_mib_peak"]) / 1024)
            if r["status"] == "oom":
                oom_x.append(nx * nz)
                oom_y.append(need_gib)
        line, = ax.plot(points, estimated, "--", alpha=0.6)
        ax.plot(measured_x, measured_y, "o", color=line.get_color(),
                label=f"snapshot every {snapshot_ms} ms (measured; dashed = estimate)")
        ax.plot(oom_x, oom_y, "x", color=line.get_color(), markersize=10, mew=2)
    # Devito: measured peak only (its memory use has no formula here).
    devito = sorted((r for r in scaling if r["code"] == "devito" and r["sweep"] == "S1"),
                    key=lambda r: int(r["nx"]) * int(r["nz"]))
    ok = [r for r in devito if r["status"] == "ok" and r["device_mem_mib_peak"]]
    if ok:
        ax.plot([int(r["nx"]) * int(r["nz"]) for r in ok],
                [float(r["device_mem_mib_peak"]) / 1024 for r in ok], "D-", color="#8064a2",
                label="Devito GPU, snapshot every 10 ms (measured)")
    failed = [r for r in devito if r["status"] in ("oom", "fail")]
    if failed and total_gib:
        ax.plot([int(r["nx"]) * int(r["nz"]) for r in failed], [total_gib] * len(failed), "D",
                color="#8064a2", markersize=10, mfc="none", label="Devito GPU: did not run (oom/fail)")
    if total_gib:
        ax.axhline(total_gib, color="red", lw=1.5, label=f"GPU memory ({total_gib:.0f} GiB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid points (interior)")
    ax.set_ylabel("device memory (GiB)")
    ax.set_title("D7 capacity: device memory vs grid size (x = out of memory)")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "D7_capacity.png")


def plot_snapshot_accuracy():
    rows = [r for r in read_csv_rows("results/compare_scaling.csv") if "_vs_snap10ms" in r["engine"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    by_dataset = defaultdict(list)
    for r in rows:
        match = re.search(r"_snap(\d+)ms_vs", r["engine"])
        if match:
            by_dataset[r["dataset"]].append((int(match.group(1)), float(r["l2_relative"])))
    for dataset, values in by_dataset.items():
        values.sort()
        ax.plot([v[0] for v in values], [v[1] for v in values], "o-", label=dataset)
    ax.set_xlabel("snapshot spacing (ms)")
    ax.set_ylabel("relative L2 error vs 10 ms")
    ax.set_yscale("log")
    ax.set_title("D7b: accuracy cost of storing fewer snapshots")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "D7b_snapshot_accuracy.png")


def plot_efficiency(scaling, kernel_rows):
    rows = [r for r in scaling if r["sweep"] == "S1" and r["status"] == "ok" and r["gpts_per_s"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax_dram = ax.twinx()
    series = sorted({(r["code"], r["engine"]) for r in rows})
    for code, engine in series:
        points = sorted(((int(r["nx"]) * int(r["nz"]), float(r["gpts_per_s"]), r["dataset"])
                         for r in rows if r["engine"] == engine), key=lambda p: p[0])
        line, = ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=f"{engine} GPts/s")
        ncu_engine = "devito" if code == "devito" else engine
        dram = []
        for x, _, dataset in points:
            match = next((k for k in kernel_rows if k["dataset"] == dataset and k["engine"] == ncu_engine
                          and k["capture"] == "stencil" and k["dram_pct"] != ""
                          and str(k["order"]) == "8"), None)
            if match:
                dram.append((x, float(match["dram_pct"])))
        if dram:
            ax_dram.plot([d[0] for d in dram], [d[1] for d in dram], "s:", color=line.get_color(),
                         alpha=0.7, label=f"{engine} DRAM % (ncu)")
    ax.set_xscale("log")
    ax.set_xlabel("grid points (interior)")
    ax.set_ylabel("throughput (GPts/s, solid)")
    ax_dram.set_ylabel("stencil DRAM throughput, % of peak (dotted)")
    ax_dram.set_ylim(0, 100)
    ax.set_title("D8 efficiency: throughput and bandwidth use vs grid size")
    ax.grid(which="both", alpha=0.3)
    lines = ax.get_legend_handles_labels()
    lines_dram = ax_dram.get_legend_handles_labels()
    ax.legend(lines[0] + lines_dram[0], lines[1] + lines_dram[1], fontsize=8, loc="lower right")
    save(fig, "D8_efficiency.png")


def plot_orders(scaling, kernel_rows):
    # One point per (engine, order); the S1 report of the same engine at order 8 is a duplicate.
    unique = {}
    for k in kernel_rows:
        if k["dataset"] == "marmousi_dx2.5" and k["capture"] == "stencil" and k["engine"] in ("cuda-v1", "devito"):
            unique[(k["engine"], k["order"])] = k
    order_reports = list(unique.values())
    plot_roofline(order_reports, "D11: roofline vs stencil order (dx = 2.5 m)", "D11_order_roofline.png",
                  lambda r: f"{'devito' if r['engine'] == 'devito' else r['engine']} o{r['order']}")
    rows = [r for r in scaling if r["sweep"] == "S2" and r["status"] == "ok" and r["gpts_per_s"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for engine in sorted({r["engine"] for r in rows}):
        values = sorted((int(r["order"]), float(r["gpts_per_s"])) for r in rows if r["engine"] == engine)
        ax.plot([v[0] for v in values], [v[1] for v in values], "o-", label=engine)
    ax.set_xlabel("stencil order")
    ax.set_ylabel("GPts/s")
    ax.set_title("D11b: throughput vs stencil order (dx = 2.5 m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "D11b_order_throughput.png")


# =============================================================================
# Markdown summary for the report
# =============================================================================
def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_summary(kernel_rows, step_rows, scaling):
    parts = ["# Profiling summary (generated by scripts/profile_extract.py)\n"]
    ladder = [r for r in kernel_rows if r["dataset"] == LADDER_DATASET]
    if ladder:
        parts.append("## D3: kernel metrics, fixed dataset\n")
        parts.append(markdown_table(ladder, ["engine", "capture", "cache", "duration_us", "dram_pct",
                                             "effective_bandwidth_gbs", "dram_bytes_per_point", "l2_hit_pct",
                                             "achieved_occupancy_pct", "registers_per_thread",
                                             "shared_mem_per_block_bytes", "waves_per_sm",
                                             "warp_execution_efficiency_pct", "thread_instructions_per_point",
                                             "arithmetic_intensity", "top_stalls"]))
    if step_rows:
        parts.append("\n## D2: GPU time per time step (Nsight Systems)\n")
        parts.append(markdown_table(step_rows, list(step_rows[0].keys())))
    if scaling:
        parts.append("\n## Scaling runs (results/scaling.csv)\n")
        parts.append(markdown_table(scaling, ["code", "engine", "sweep", "dataset", "f0", "order",
                                              "snapshot_ms", "status", "s_per_shot", "gpts_per_s",
                                              "device_mem_mib_peak", "mean_power_w"]))
    if len(parts) > 1:
        with open(f"{PROFILES}/SUMMARY.md", "w") as f:
            f.write("\n".join(parts) + "\n")
        print(f"wrote {PROFILES}/SUMMARY.md")


def main():
    if not os.path.isdir(PROFILES):
        print(f"no {PROFILES}/ yet: run the pod scripts and sync first")
        return
    kernel_rows = collect_kernel_metrics()
    step_rows = collect_step_breakdown()
    scaling = read_csv_rows("results/scaling.csv")

    plot_step_breakdown(step_rows)
    plot_kernel_metrics(kernel_rows)
    plot_roofline([r for r in kernel_rows if r["dataset"] == LADDER_DATASET and r["cache"] == "all"],
                  f"D4: roofline, CUDA ladder ({LADDER_DATASET})", "D4_roofline_ladder.png",
                  lambda r: f"{r['engine']} {r['capture']}")
    plot_capacity(scaling)
    plot_snapshot_accuracy()
    plot_efficiency(scaling, kernel_rows)
    plot_orders(scaling, kernel_rows)
    write_summary(kernel_rows, step_rows, scaling)


if __name__ == "__main__":
    main()
