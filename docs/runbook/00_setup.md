# Runbook 00 — Pod setup, reference run, first bar

What this produces on the pod, once:

* a build with every engine in the ladder so far (`cpu`, `cuda-v0`),
* the verified fixed dataset (`marmousi12`),
* the bandwidth probe numbers for the pod's CPU and GPU,
* the **CPU reference image** `results/ref/marmousi_ref.bin` + its 1× rows,
* the first chart with one bar.

Expected time: ~5 min of commands + **15–30 min** for the reference run.

---

## 1. Create the pod

| setting | choose |
|---|---|
| Template / image | any **CUDA 12.x devel** image (needs `nvcc`). RunPod's "PyTorch" templates have it. |
| GPU | whatever is cheap; the whole job needs < 1 GB of GPU memory. |
| vCPUs | prefer **8–16** — the `cpu-opt` bar scales with this. |
| Volume | a persistent **/workspace** volume (≥ 5 GB) so the build, dataset and results survive restarts. |
| Access | SSH with your public key (RunPod: *Connect → SSH over exposed TCP*). |

Add an alias so every later command is short — in `~/.ssh/config` on your machine:

```
Host pod
    HostName <ip-from-runpod>
    Port     <port-from-runpod>
    User     root
    IdentityFile ~/.ssh/id_ed25519
```

Check: `ssh pod nvidia-smi` prints the GPU.

## 2. Send the code and the dataset

```bash
cd ~/RTM/RTM-Hardware-Acceleration
scripts/local/sync_to_pod.sh pod
```

(~50 MB: code + `data/real/marmousi_shots.bin`.)

## 3. Run the setup script

```bash
ssh pod "cd /workspace/RTM && scripts/pod/setup.sh"
```

It prints six numbered steps. What each must show:

| step | expect |
|---|---|
| 1/6 tools | GPU name from `nvidia-smi`, `cpu cores: N` |
| 2/6 build | `cpu cuda-v0` |
| 3/6 dataset checksum | three `OK` lines |
| 4/6 smoke test | two lines ending in `PASS` (reference reproducibility, then `cuda-v0` vs reference on the small synthetic model — **this is the first real GPU test**) |
| 5/6 bandwidth probe | `CPU ... GB/s` and `GPU ... GB/s` |
| 6/6 CPU reference | progress for 12 shots, then the benchmark report — the long step |

Then the summary table with one row: `cpu … 1.0x … PASS`.

Safe to re-run: every step skips when its output already exists.

## 4. Bring the results back

```bash
scripts/local/sync_from_pod.sh pod
```

## 5. Send me

* the full terminal output of step 3 (or just steps 4/6, 5/6 and the final table),
* `results/benchmarks.csv`, `results/compare.csv`, `results/ref/bandwidth.csv` (they come with step 4).

---

## If something fails

| symptom | do |
|---|---|
| `nvcc not found` | the image is a runtime image, not devel. Pick a devel/PyTorch template. |
| `cudaSetDevice(0): ...` error in the smoke test | `nvidia-smi` on the pod — if it shows no GPU the pod has none attached. |
| `cuda-v0` smoke test says `FAIL` or crashes | let the script finish (steps 5–6 only use `cpu`), then send me the smoke-test output. |
| checksum `FAILED` | the sync was incomplete: re-run `sync_to_pod.sh`, then `setup.sh`. |
| `apt-get` permission error | the image isn't root: prefix `sudo`, or pick a root template. |
| It stops during step 6 (SSH dropped) | run it under `nohup`: `ssh pod "cd /workspace/RTM && nohup scripts/pod/setup.sh > setup.log 2>&1 &"` then `ssh pod tail -f /workspace/RTM/setup.log`. |
