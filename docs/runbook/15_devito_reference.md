# Runbook 15 — Devito, the industry-reference cross-check

Prerequisite: runbook 00 done (`results/ref/marmousi_ref.bin` exists on the
pod — that's what Devito's image gets compared against).

What it is: a **second, independently written** acoustic RTM code
([Devito](https://www.devitoproject.org/), a finite-difference DSL from the
seismic-imaging research community: Imperial College London / SLIM, used in
many published RTM/FWI papers) migrating the exact same
`data/real/marmousi_vp_12.5m.bin` + `data/real/marmousi_shots.bin` files.
It is **not** a rung in the CUDA ladder — it doesn't compile against this
repo's C++ at all, runs entirely in its own Python venv, and never writes to
`results/benchmarks.csv` / `results/compare.csv` (those stay reserved for the
project's own engines on one host, per `docs/OPTIMIZATION_PLAN.md` §0).

Why it matters: `cpu`, `cpu-opt` and every `cuda-*` engine are all graded
against `CPUReferenceRTM` — if that reference itself had a bug, every engine
would "pass" while being wrong together. Devito is written by a different
team, with a different absorbing boundary implementation and a different
time-stepping code path. If its image lands on the same reflectors as this
project's reference image, that is external evidence the physics is right,
not just that the engines agree with each other.

Script: `scripts/devito/marmousi_rtm_devito.py`. It reads/writes the same
file formats as `./build/rtm` (raw float32 velocity + `.hdr` sidecar, RTMS
shot gathers, raw float32 `image[ix*nz+iz]` output — see README.md §4–6), so
the project's own `rtm_compare` and `scripts/plot_image.py` work on its
output unmodified. What it reproduces from `CPUReferenceRTM`
(`src/rtm/rtm_cpu.cpp`) and what it doesn't is documented in the script's
docstring — read it before running.

Expected time: ~5 min setup, ~1–2 min JIT compile, then roughly
**shot-count × (grid size / your reference run's throughput)** — expect the
same order of magnitude as the CPU reference run in runbook 00 (15–30 min for
12 shots), since Devito is also a plain OpenMP CPU stencil code here.

---

## 1. Install Devito on the pod

```bash
ssh pod "cd /workspace/RTM && scripts/pod/setup_devito.sh"
```

This creates `venv_devito/`, `pip install devito`, and runs a smoke test on
the small synthetic model (skipped with a note if `data/synthetic/*.bin`
don't exist yet — that's fine, it's optional).

Expect: `devito <version>` printed, then either the smoke-test image path or
the "skip" note.

## 2. Sanity-check on the synthetic model (optional but recommended)

Confirms the coordinate mapping and file formats line up before spending
20+ minutes on the full Marmousi run.

```bash
ssh pod "cd /workspace/RTM && ./build/rtm --engine cpu \
    --velocity data/synthetic/velocity.bin --shots data/synthetic/shots.bin \
    --output results/ref/smoke_cpu.bin \
    --order 8 --nb 60 --f0 12 --store-interval 2 --mute-direct 1500 --quiet"
ssh pod "cd /workspace/RTM && ./build/rtm_compare results/ref/smoke_cpu.bin results/ref/smoke_devito.bin --nx 81 --nz 61"
```

(`--nx --nz` here match whatever `data/synthetic/velocity.bin.hdr` says — check
it with `cat data/synthetic/velocity.bin.hdr` if it's not the 81×61 default.)

**Do not expect `PASS`** against the default `L2rel < 1e-5` gate — that gate is
for two builds of *the same* code. Two independently discretized absorbing
boundaries and time-steppers routinely differ by a constant amplitude scale
and a few percent RMS. What to actually look for:

* `normalized correlation` comfortably above ~0.85 for a single shot (it
  climbs with more shots — the real 12-shot Marmousi stack should do better),
* the reflector positions visually match when you plot both (§4),
* no `nan`/`inf`, no wildly different structure (that would mean a real bug,
  e.g. a coordinate-frame mismatch).

## 3. Run on the fixed Marmousi dataset

```bash
ssh pod "cd /workspace/RTM && source scripts/dataset_marmousi.env && source venv_devito/bin/activate && \
    DEVITO_LANGUAGE=openmp OMP_NUM_THREADS=\$(nproc) \
    python3 scripts/devito/marmousi_rtm_devito.py \
        --velocity \$MARMOUSI_VEL --shots \$MARMOUSI_SHOTS \
        --output results/ref/marmousi_devito.bin \
        --order \$ORDER --nb \$NB --f0 \$F0 --store-interval \$STORE_INTERVAL \
        --mute-direct \$MUTE_VELOCITY --dataset \$DATASET_NAME \
        --benchmark-csv results/ref/devito_benchmark.csv"
```

Sourcing `scripts/dataset_marmousi.env` keeps every parameter identical to
what `scripts/pod/bench.sh` used for `cpu` / `cpu-opt` / `cuda-*` — this is
the same fixed dataset, same order/nb/f0/mute, so the comparison in §4 is
apples-to-apples. It prints one line per shot with elapsed time; if it's
going to run long, wrap it in `nohup ... &` the same way as runbook 00's
"if something fails" table suggests for `setup.sh`.

Running out of memory: the forward wavefield is stored every
`--store-interval` steps at extended-grid resolution — for the fixed dataset
that's roughly `(nt/store_interval) × (nx+2·nb) × (nz+2·nb) × 4 bytes` ≈
`281 × 1461 × 381 × 4` ≈ 625 MB, held once per shot (freed before the next
one starts). If a pod is unusually memory-constrained, raise
`--store-interval` (halves memory roughly per doubling) or test first with
`--max-shots 1`.

## 4. Compare against the CPU reference

```bash
ssh pod "cd /workspace/RTM && ./build/rtm_compare results/ref/marmousi_ref.bin results/ref/marmousi_devito.bin --nx 1361 --nz 281"
ssh pod "cd /workspace/RTM && source venv_devito/bin/activate && python3 scripts/plot_image.py results/ref/marmousi_devito.bin --nx 1361 --nz 281 --dx 12.5 --dz 12.5 --save results/plots/marmousi_devito.png"
ssh pod "cd /workspace/RTM && source venv_devito/bin/activate && python3 scripts/plot_image.py results/ref/marmousi_ref.bin --nx 1361 --nz 281 --dx 12.5 --dz 12.5 --save results/plots/marmousi_ref_forcompare.png"
```

Read the `rtm_compare` numbers as a similarity score, not a pass/fail gate
(see §2). Put both PNGs side by side — the AGL Marmousi2 reflectors (the
dipping layers, the anticline, the faulted blocks) should appear in the same
place in both images.

## 5. Bring the results back

```bash
scripts/local/sync_from_pod.sh pod
rsync -az pod:/workspace/RTM/results/ref/marmousi_devito.bin results/ref/ 2>/dev/null
rsync -az pod:/workspace/RTM/results/ref/devito_benchmark.csv results/ref/ 2>/dev/null
rsync -az pod:/workspace/RTM/results/plots/marmousi_devito.png results/plots/ 2>/dev/null
```

(`sync_from_pod.sh` only pulls `*.csv`/`*.txt`/`results/ref/***`/`results/plots/***`
by its include list — `marmousi_devito.bin` is inside `results/ref/` so it's
actually already covered by the first command; the extra `rsync` lines are
here only in case you ran it into a different output path.)

## 6. Send me

* the `rtm_compare` block from §4,
* `results/ref/marmousi_devito.bin` and `results/plots/marmousi_devito.png`,
* `results/ref/devito_benchmark.csv` (just for the record — it is a timing
  curiosity, not a ladder bar; Devito's own JIT-compiled OpenMP loop is not
  wall-clock comparable to a hand-tuned engine without controlling for far
  more variables than this quick script does).

---

## Optional: Devito on the GPU

Devito can offload to the GPU via OpenACC, but that needs the **NVIDIA HPC
SDK** (`nvc`/`nvc++`, not `nvcc`) installed and `DEVITO_PLATFORM=nvidiaX
DEVITO_ARCH=nvc`, which most RunPod CUDA-devel images don't ship and which
can be fiddly to match against the pod's driver. Since Devito's role here is
a **correctness** cross-check (§0), not a speed bar — that story is already
told by `cuda-v0`..`cuda-v4` against `cpu`/`cpu-opt` — this is not worth
chasing unless you specifically want it. If you do:

```bash
ssh pod "curl -fsSL https://developer.download.nvidia.com/hpc-sdk/... -o nvhpc.tar.gz"  # find current URL on developer.nvidia.com
# install, then:
ssh pod "cd /workspace/RTM && source venv_devito/bin/activate && \
    DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=nvc DEVITO_LANGUAGE=openacc \
    python3 scripts/devito/marmousi_rtm_devito.py --max-shots 1 ..."
```

Start with `--max-shots 1` — GPU-offload JIT failures in Devito are common on
a first try and much cheaper to debug on one shot.

---

## If something fails

| symptom | do |
|---|---|
| `pip install devito` fails to build a dependency | `apt-get install -y build-essential python3-dev`, retry. |
| `ImportError: cannot import name ...` from devito | version drift; pin `pip install devito==4.8.23` (the version this runbook was written against — check with `python3 -c "import devito; print(devito.__version__)"` locally). |
| very slow (` > 5 min/shot`) | check `DEVITO_LANGUAGE=openmp` and `OMP_NUM_THREADS` are actually set in the shell that runs the script — Devito defaults to sequential C if `DEVITO_LANGUAGE` isn't exported. |
| `normalized correlation` is low (< 0.5) and images look structurally different | likely a real coordinate/unit mismatch, not just discretization noise — first re-run §2 (synthetic, fast) and check `--nx --nz` match `data/real/marmousi_vp_12.5m.bin.hdr`, then send me the two PNGs. |
| out of memory | see the memory note in §3; raise `--store-interval` or reduce `--max-shots`. |
| `OOB detected` / `InvalidArgument` from Devito | send me the full traceback — this is an API-version mismatch in the script, not a data problem. |
