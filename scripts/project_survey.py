#!/usr/bin/env python3
"""Project measured throughput to a whole seismic survey (docs/PROFILING_STRATEGY.md §7, M11).

    python3 scripts/project_survey.py --survey-shots 1000 --dx 6.25 \
        [--gpu-price-per-hour 0.44] [--cpu-node-price-per-hour 1.0] [--cpu-node-power-w 280] \
        [--gpus 1 8]

For every code measured at that grid spacing (results/scaling.csv, sweep S1,
the most detailed snapshot policy that fitted), and for the CPU engines
(results/benchmarks.csv at 12.5 m, extrapolated by work):

    work per shot  W = extended points x nt x 2            (point-updates)
    time per shot  t = measured s/shot  (CPU: W / measured CPU GPts/s)
    survey time    T = survey_shots x t / devices          (shots are independent)
    cost           C = T x price per device-hour           (if a price is given)
    energy         E = T x mean power x devices            (GPU: nvidia-smi; CPU: --cpu-node-power-w)

Assumptions written into the output: linear scaling with the number of
devices (shots are independent), identical parameters for every shot, no
I/O bottleneck, CPU throughput independent of grid size.

Writes results/survey_projection_dx<dx>.md and results/plots/profiling/D10_survey_projection_dx<dx>.png.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cpu_throughput_gpts():
    """GPts/s of cpu and cpu-opt from the fixed-dataset benchmark rows (latest row wins)."""
    rates = {}
    for r in read_csv_rows("results/benchmarks.csv"):
        if r["engine"] not in ("cpu", "cpu-opt"):
            continue
        points = (int(r["nx"]) + 2 * int(r["nb"])) * (int(r["nz"]) + 2 * int(r["nb"]))
        seconds = float(r["t_forward"]) + float(r["t_backward"])
        if seconds > 0:
            rates[r["engine"]] = (points * int(r["nt"]) * 2 * int(r["nshots"]) / seconds / 1e9,
                                  r["cpu_name"], int(r["threads"]))
    return rates


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--survey-shots", type=int, default=1000)
    p.add_argument("--dx", default="6.25", help="grid spacing of the scaled dataset to project")
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 8])
    p.add_argument("--gpu-price-per-hour", type=float, default=None, help="$ per GPU-hour (pod page)")
    p.add_argument("--cpu-node-price-per-hour", type=float, default=None, help="$ per CPU-node-hour")
    p.add_argument("--cpu-node-power-w", type=float, default=280.0,
                   help="CPU node power for the energy estimate (EPYC 7H12 TDP = 280 W)")
    a = p.parse_args()

    scaling = [r for r in read_csv_rows("results/scaling.csv")
               if r["dataset"] == f"marmousi_dx{a.dx}" and r["sweep"] in ("S1", "S1B", "S1T", "S2")
               and r["order"] == "8" and r["status"] == "ok"]
    if not scaling:
        raise SystemExit(f"no successful S1 rows for marmousi_dx{a.dx} in results/scaling.csv")
    reference = scaling[0]
    nx, nz, nb, nt = (int(reference[k]) for k in ("nx", "nz", "nb", "nt"))
    work_per_shot = (nx + 2 * nb) * (nz + 2 * nb) * nt * 2

    # One line per (code, engine): the densest snapshot spacing that fitted.
    best = {}
    for r in scaling:
        key = r["engine"]
        if key not in best or int(r["store_interval"]) < int(best[key]["store_interval"]):
            best[key] = r

    lines = []   # (label, devices, hours, cost, kwh)
    for engine, r in sorted(best.items()):
        seconds_per_shot = float(r["s_per_shot"])
        power = float(r["mean_power_w"]) if r["mean_power_w"] else None
        for devices in a.gpus:
            hours = a.survey_shots * seconds_per_shot / devices / 3600
            cost = hours * devices * a.gpu_price_per_hour if a.gpu_price_per_hour else None
            kwh = hours * devices * power / 1000 if power else None
            lines.append((f"{engine} ({r['gpu_name']}), snapshot every {r['store_interval']} steps", devices, hours, cost, kwh))
    for engine, (gpts, cpu_name, threads) in sorted(cpu_throughput_gpts().items()):
        hours = a.survey_shots * work_per_shot / (gpts * 1e9) / 3600
        cost = hours * a.cpu_node_price_per_hour if a.cpu_node_price_per_hour else None
        lines.append((f"{engine} ({cpu_name}, {threads} threads), extrapolated", 1, hours, cost,
                      hours * a.cpu_node_power_w / 1000))

    fmt = lambda v, d=1: "" if v is None else f"{v:,.{d}f}"
    md = [f"# Survey projection: {a.survey_shots} shots at dx = {a.dx} m\n",
          f"Grid {nx} x {nz} (+{nb} sponge cells per side), {nt} time steps, "
          f"{work_per_shot / 1e9:,.0f} G point-updates per shot.\n",
          "| code / device | devices | hours | cost ($) | energy (kWh) |", "|---|---|---|---|---|"]
    for label, devices, hours, cost, kwh in lines:
        md.append(f"| {label} | {devices} | {fmt(hours, 2)} | {fmt(cost, 2)} | {fmt(kwh, 1)} |")
    md.append("\nAssumptions: shots are independent so time divides linearly by the number of devices; "
              "every shot has the same parameters; I/O is not a bottleneck; CPU throughput measured at "
              "12.5 m is assumed to hold at this grid size; CPU energy uses a fixed node power of "
              f"{a.cpu_node_power_w:.0f} W; GPU energy uses the mean nvidia-smi power of the measured run.")
    os.makedirs("results", exist_ok=True)
    md_path = f"results/survey_projection_dx{a.dx}.md"
    png_path = f"results/plots/profiling/D10_survey_projection_dx{a.dx}.png"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(lines) + 1.5))
    labels = [f"{label} x{devices}" for label, devices, *_ in lines]
    hours = [line[2] for line in lines]
    ax.barh(labels, hours, color=["#9bbb59" if "cpu" in l.split()[0] else "#4f81bd" for l in labels])
    for y, h in enumerate(hours):
        ax.annotate(f"{h:,.1f} h", (h, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("hours for the whole survey (log)")
    ax.set_title(f"D10: {a.survey_shots} shots at dx = {a.dx} m")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs("results/plots/profiling", exist_ok=True)
    fig.savefig(png_path, dpi=150)
    print(f"wrote {md_path}, {png_path}")


if __name__ == "__main__":
    main()
