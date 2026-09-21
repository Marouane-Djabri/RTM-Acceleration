#!/usr/bin/env python3
"""Print one line per engine from results/benchmarks.csv + results/compare.csv.
Standard library only, so it runs on any pod without pip."""
import csv, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BENCH = os.path.join(ROOT, "results", "benchmarks.csv")
COMPARE = os.path.join(ROOT, "results", "compare.csv")


def latest_rows(path, key_fields):
    """Last row per key (the most recent run of each engine wins)."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[tuple(row[k] for k in key_fields)] = row
    return rows


def main():
    bench = latest_rows(BENCH, ("host", "engine", "dataset", "gpus"))
    gate = latest_rows(COMPARE, ("host", "engine", "dataset"))
    if not bench:
        print("no benchmark rows yet")
        return
    print(f"{'engine':<16}{'dataset':<20}{'gpus':>5}{'total s':>11}{'vs cpu':>9}"
          f"{'vs cpu-opt':>12}{'GFLOP/s':>10}{'L2rel':>12}  gate")
    for (host, engine, dataset, gpus), row in sorted(bench.items()):
        cpu = bench.get((host, "cpu", dataset, "0"))
        cpu_opt = bench.get((host, "cpu-opt", dataset, "0"))
        total = float(row["t_total"])
        speed = f"{float(cpu['t_total']) / total:8.1f}x" if cpu and total > 0 else "      --"
        speed_opt = f"{float(cpu_opt['t_total']) / total:10.2f}x" if cpu_opt and total > 0 else "         --"
        gate_engine = engine if gpus in ("0", "1") else f"{engine}_gpus{gpus}"
        g = gate.get((host, gate_engine, dataset))
        l2 = f"{float(g['l2_relative']):12.3e}" if g else "          --"
        verdict = ("PASS" if g["pass"] == "1" else "FAIL") if g else "not gated"
        print(f"{engine:<16}{dataset:<20}{gpus:>5}{total:11.2f}{speed:>9}{speed_opt:>12}"
              f"{float(row['stencil_gflops']):10.1f}{l2}  {verdict}")


if __name__ == "__main__":
    sys.exit(main())
