#!/usr/bin/env python3
"""Append one row to results/scaling.csv: one run of one code at one sweep point
(docs/PROFILING_STRATEGY.md §5-§6). Our engine and Devito write the same columns,
so the capacity / efficiency / order charts put both codes on one plot.

Our engine:  --bench-csv  the temporary CSV that `rtm --benchmark-csv` wrote
Devito:      --devito-json  the JSON that marmousi_rtm_devito.py --run-json wrote
Both:        --monitor  the nvidia-smi log (timestamp, memory.used, memory.total, power.draw)

Standard library only.
"""
import argparse
import csv
import json
import os
import socket
import time

COLUMNS = ["date", "host", "gpu_name", "gpu_total_mib", "code", "engine", "sweep", "dataset",
           "dx", "f0", "order", "nx", "nz", "nb", "nt", "nshots", "store_interval",
           "snapshot_ms", "status", "t_migrate_s", "s_per_shot", "gpts_per_s",
           "device_mem_mib_engine", "device_mem_mib_peak", "mean_power_w", "energy_j",
           "devito_gpts", "devito_gflops", "devito_oi", "jit_s", "image"]


def read_key_values(path):
    """KEY=VALUE or 'key = value' lines, '#' comments ignored (.env and .hdr files)."""
    values = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                values[key] = value
    return values


def read_monitor(path):
    """Peak memory used, total memory and mean power from the nvidia-smi log."""
    peak_used, total, powers = 0.0, "", []
    if not path or not os.path.exists(path):
        return "", "", ""
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                peak_used = max(peak_used, float(parts[1]))
                total = parts[2]
                powers.append(float(parts[3]))
            except ValueError:
                continue
    mean_power = sum(powers) / len(powers) if powers else ""
    return (peak_used if peak_used > 0 else ""), total, mean_power


def gpu_name():
    try:
        return os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip().splitlines()[0]
    except (IndexError, OSError):
        return "unknown-gpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scaling.csv")
    p.add_argument("--code", required=True, choices=["ours", "devito"])
    p.add_argument("--engine", required=True)
    p.add_argument("--sweep", required=True, help="S1 (grid), S1B (snapshot policy B), S2 (order)")
    p.add_argument("--env", required=True, help="dataset env file")
    p.add_argument("--order", required=True)
    p.add_argument("--store-interval", required=True)
    p.add_argument("--snapshot-ms", required=True)
    p.add_argument("--status", required=True, help="ok | oom | fail")
    p.add_argument("--bench-csv")
    p.add_argument("--devito-json")
    p.add_argument("--monitor")
    p.add_argument("--image", default="")
    a = p.parse_args()

    env = read_key_values(a.env)
    header = read_key_values(env["MARMOUSI_VEL"] + ".hdr")
    peak_mib, total_mib, mean_power = read_monitor(a.monitor)

    row = {c: "" for c in COLUMNS}
    row.update(date=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(),
               gpu_name=gpu_name(), gpu_total_mib=total_mib, code=a.code, engine=a.engine,
               sweep=a.sweep, dataset=env["DATASET_NAME"], dx=env.get("DX", header["dx"]),
               f0=env["F0"], order=a.order, nx=header["nx"], nz=header["nz"], nb=env["NB"],
               nt=env["NT"], store_interval=a.store_interval, snapshot_ms=a.snapshot_ms,
               status=a.status, device_mem_mib_peak=peak_mib, mean_power_w=mean_power,
               image=a.image)

    if a.status == "ok" and a.code == "ours" and a.bench_csv and os.path.exists(a.bench_csv):
        with open(a.bench_csv, newline="") as f:
            bench = list(csv.DictReader(f))[-1]
        nshots = int(bench["nshots"])
        total = float(bench["t_total"])
        propagation = float(bench["t_forward"]) + float(bench["t_backward"])
        points = (int(bench["nx"]) + 2 * int(bench["nb"])) * (int(bench["nz"]) + 2 * int(bench["nb"]))
        row.update(nshots=nshots, t_migrate_s=f"{total:.4f}", s_per_shot=f"{total / nshots:.4f}",
                   gpts_per_s=f"{points * int(bench['nt']) * 2 * nshots / propagation / 1e9:.4f}",
                   device_mem_mib_engine=bench.get("device_mem_mib", ""))
    if a.status == "ok" and a.code == "devito" and a.devito_json and os.path.exists(a.devito_json):
        with open(a.devito_json) as f:
            run = json.load(f)
        row.update(nshots=run["nshots"], t_migrate_s=f"{run['t_migrate_s']:.4f}",
                   s_per_shot=f"{run['s_per_shot']:.4f}", gpts_per_s=f"{run['gpts_per_s']:.4f}",
                   devito_gpts=run.get("devito_gpts", ""), devito_gflops=run.get("devito_gflops", ""),
                   devito_oi=run.get("devito_oi", ""), jit_s=f"{run['jit_s']:.2f}")
    if row["t_migrate_s"] and mean_power != "":
        row["energy_j"] = f"{float(mean_power) * float(row['t_migrate_s']):.1f}"

    need_header = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"    scaling row: {a.code} {a.engine} {row['dataset']} order {a.order} "
          f"snap {a.snapshot_ms} ms -> {a.status}"
          + (f", {row['gpts_per_s']} GPts/s, {row['s_per_shot']} s/shot" if row["gpts_per_s"] else ""))


if __name__ == "__main__":
    main()
