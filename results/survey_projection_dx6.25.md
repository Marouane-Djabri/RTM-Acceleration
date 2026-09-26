# Survey projection: 1000 shots at dx = 6.25 m

Grid 2721 x 561 (+50 sponge cells per side), 5600 time steps, 21 G point-updates per shot.

| code / device | devices | hours | cost ($) | energy (kWh) |
|---|---|---|---|---|
| cuda-v1 (NVIDIA A40), snapshot every 10 steps | 1 | 0.28 |  | 0.0 |
| cuda-v1 (NVIDIA A40), snapshot every 10 steps | 8 | 0.04 |  | 0.0 |
| cuda-v3 (NVIDIA A40), snapshot every 20 steps | 1 | 0.27 |  | 0.1 |
| cuda-v3 (NVIDIA A40), snapshot every 20 steps | 8 | 0.03 |  | 0.1 |
| devito-gpu (NVIDIA A40), snapshot every 10 steps | 1 | 1.00 |  | 0.1 |
| devito-gpu (NVIDIA A40), snapshot every 10 steps | 8 | 0.12 |  | 0.1 |
| cpu (Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 1 threads), extrapolated | 1 | 157.01 |  | 44.0 |
| cpu-opt (Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 8 threads), extrapolated | 1 | 77.23 |  | 21.6 |

Assumptions: shots are independent so time divides linearly by the number of devices; every shot has the same parameters; I/O is not a bottleneck; CPU throughput measured at 12.5 m is assumed to hold at this grid size; CPU energy uses a fixed node power of 280 W; GPU energy uses the mean nvidia-smi power of the measured run.
