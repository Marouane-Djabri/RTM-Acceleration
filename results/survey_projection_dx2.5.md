# Survey projection: 1000 shots at dx = 2.5 m

Grid 6801 x 1401 (+50 sponge cells per side), 14000 time steps, 290 G point-updates per shot.

| code / device | devices | hours | cost ($) | energy (kWh) |
|---|---|---|---|---|
| cuda-v1 (NVIDIA A40), snapshot every 20 steps | 1 | 3.66 |  | 1.0 |
| cuda-v1 (NVIDIA A40), snapshot every 20 steps | 8 | 0.46 |  | 1.0 |
| cuda-v3 (NVIDIA A40), snapshot every 50 steps | 1 | 3.59 |  | 1.0 |
| cuda-v3 (NVIDIA A40), snapshot every 50 steps | 8 | 0.45 |  | 1.0 |
| devito-gpu (NVIDIA A40), snapshot every 40 steps | 1 | 5.13 |  | 0.9 |
| devito-gpu (NVIDIA A40), snapshot every 40 steps | 8 | 0.64 |  | 0.9 |
| cpu (Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 1 threads), extrapolated | 1 | 2,180.44 |  | 610.5 |
| cpu-opt (Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 8 threads), extrapolated | 1 | 1,072.54 |  | 300.3 |

Assumptions: shots are independent so time divides linearly by the number of devices; every shot has the same parameters; I/O is not a bottleneck; CPU throughput measured at 12.5 m is assumed to hold at this grid size; CPU energy uses a fixed node power of 280 W; GPU energy uses the mean nvidia-smi power of the measured run.
