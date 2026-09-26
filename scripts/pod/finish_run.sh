#!/usr/bin/env bash
# =============================================================================
# Fix-ups after the 2026-09-26 A40 run of run_all.sh. Run ONCE, after run_all.sh
# has finished and the corrected scripts have been synced:
#
#     ssh pod "cd /workspace/RTM && nohup scripts/pod/finish_run.sh > finish_run.out 2>&1 < /dev/null &"
#
#   1. keep the first capacity sweep's rows but relabel them: its snapshots were
#      every 10 ms, which aliases the imaging at fine grids. Its throughput stays
#      valid (sweep S1 -> S1T); its snapshot-policy rows do not (S1B -> S1B_aliased).
#   2. rewrite the scaled datasets' env files (snapshot every 10 STEPS)
#   3. corrected capacity sweep, cuda-v1 only                        (~15-20 min)
#   4. Nsight Systems ladder (nsys was missing during phase 4)        (~3 min)
#   5. this machine's bandwidth probe + CPU reference timing (both were
#      overwritten by a sync from the laptop)                         (~20 min)
#   6. clean cpu-opt timing (the first one was starved of CPU)        (~2 min)
#   7. Devito: HPC SDK install (failed in the main run on a script bug) and both
#      sweeps, after step 3 so its images compare with the corrected ones (~1-1.5 h)
#   8. benchmark plots + the full report (tables, charts, projections)
# A failed step is recorded and the script moves on. Summary at the end.
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/common.sh"

if pgrep -f "scripts/pod/run_all.sh" >/dev/null; then
    echo "run_all.sh is still running: start this script after it has finished."
    exit 1
fi

declare -a SUMMARY
step() {   # step NAME COMMAND...
    local name="$1"; shift
    log_step "$name"
    local start=$SECONDS
    if "$@"; then SUMMARY+=("OK      $(( (SECONDS - start) / 60 )) min  $name")
    else          SUMMARY+=("FAILED  $(( (SECONDS - start) / 60 )) min  $name")
    fi
}

relabel_aliased_rows() {
    local marker="$PROFILES_DIR/.phases/relabel_aliased.done"
    [ -f "$marker" ] && { echo "already relabelled"; return 0; }
    python3 - <<'EOF'
import csv
path = "results/scaling.csv"
rows = list(csv.DictReader(open(path, newline="")))
fields = list(rows[0].keys())
relabelled = 0
for r in rows:
    if r["code"] == "ours" and r["sweep"] in ("S1", "S1B"):
        r["sweep"] = "S1T" if r["sweep"] == "S1" else "S1B_aliased"
        relabelled += 1
with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"relabelled {relabelled} rows of the first capacity sweep")
EOF
    touch "$marker"
}

rewrite_dataset_env_files() {
    local failed=0
    for decimation in $SWEEP_DECIMATIONS; do
        scripts/pod/make_scaled_dataset.sh "$decimation" | grep -E "===|STORE|skipping" || failed=1
    done
    grep -h "STORE_INTERVAL" data/scaled/*/dataset.env | sort | uniq -c
    return $failed
}

step "relabel the first capacity sweep"          relabel_aliased_rows
step "dataset env files: snapshot every 10 steps" rewrite_dataset_env_files
step "corrected capacity sweep (cuda-v1)"         scripts/pod/capacity_sweep.sh cuda-v1
step "Nsight Systems ladder"                      scripts/pod/run_all.sh --only nsys_ladder
step "bandwidth probe + CPU reference timing"     scripts/pod/run_all.sh --only setup
step "clean cpu-opt timing"                       scripts/pod/bench.sh cpu-opt
step "Devito: NVIDIA HPC SDK + GPU venv"          scripts/pod/setup_devito_gpu.sh
step "Devito: grid sweep S1"                      scripts/pod/devito_sweep.sh grid
step "Devito: order sweep S2"                     scripts/pod/devito_sweep.sh order
step "benchmark plots"                            scripts/pod/plot.sh
step "report: tables, charts, projections"        scripts/pod/run_all.sh --only report

echo
echo "================================ FINISH SUMMARY ================================"
printf '  %s\n' "${SUMMARY[@]}"
echo "================================================================================"
echo "Report files: results/profiles/SUMMARY.md, results/plots/profiling/, results/survey_projection_dx*.md"
