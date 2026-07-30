# TSP-D Benchmarks

Reference costs for the Traveling Salesman Problem with Drone (TSP-D), aligned with the paper’s Table 2 setting and this repo’s test files (`data/DroneTruck-size-100-len-*.txt`).

**Instance settings:** truck speed `v_t = 1.0`, drone speed `v_d = 2.0` (cost factor `0.5`), unlimited drone range, **100** instances per size.  
**Gap % (OR):** relative to the best mean cost among OR methods for that `N`.  
**Gap % (RL / NN):** relative to **TSP-ep-all** mean for that `N`.

---

## OR baselines (Julia)

Implemented in [`julia/TSPDroneBaselines`](julia/TSPDroneBaselines) (Concorde TSP tour + TSP-ep / DPS). Reproduce with:

```bash
julia --project=julia/TSPDroneBaselines \
  julia/TSPDroneBaselines/scripts/reproduce_table2.jl \
  --n 11,20,50,100 --methods TSP-ep-all,DPS/10,DPS/25
```

### N = 11

Best mean cost among methods below: **229.87**

| Method | Mean cost ± std | Gap | Time |
| --- | ---: | ---: | ---: |
| TSP-ep-all | 229.87 ± 22.74 | 0.0% | 0.01 s |
| DPS/10 | 229.95 ± 22.83 | 0.03% | 0.01 s |

### N = 20, 50, 100

| Method | N = 20 | | | N = 50 | | | N = 100 | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | Cost | Gap | Time | Cost | Gap | Time | Cost | Gap | Time |
| TSP-ep-all | 281.62 ± 18.05 | 0.64% | 0.17 s | 397.21 ± 20.19 | 1.43% | 43 s | 535.50 ± 21.83 | 0.78% | 3992 s |
| DPS/10 | 292.23 ± 19.00 | 4.48% | 0.02 s | 420.51 ± 23.98 | 7.39% | 0.08 s | 570.74 ± 20.61 | 7.55% | 0.35 s |
| DPS/25 | — | — | — | 404.78 ± 22.03 | 3.37% | 1.04 s | 548.23 ± 22.36 | 3.19% | 2.27 s |

Notes:

- **TSP-ep-all** is the strong OR reference; at N = 100 it is very slow (~1 h+ per full 100-instance pass on typical CPUs).
- **DPS/g** partitions into `N ÷ g` groups; DPS/25 is not reported for N = 20 (partition too coarse / unused in the table).
- Wall-clock times depend on CPU; paper runs used an Intel Xeon E5-2630.

---

## RL greedy eval (`scale=small`, trained matrix)

Greedy test (`action=test`) on the same 100-instance files. Training: decoder × dynamics × `N ∈ {11,20,50}`, `parameter_budget.enabled=true`, checkpoints under `~/local_db/tspdrone-rl/outputs/training/small/<wandb.name>/n<N>/`.

### Valid (checkpoint loaded)

Only **`tspd_lstm_on`** loaded weights in this eval pass. Other architectures printed `Skipping checkpoint load ... (only tspd_lstm_on may attempt load)` and ran **untrained** — those numbers are omitted.

| Method | N = 11 | Gap | N = 20 | Gap | N = 50 | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tspd_lstm_on | **273.85** | 19.1% | **351.07** | 24.7% | **574.73** | 44.7% |

### Invalid this pass (checkpoint skipped — do not use)

Same untrained/random init collapse (repeated identical makespans across architectures):

| Method | N = 11 | N = 20 | N = 50 |
| --- | ---: | ---: | ---: |
| tspd_lstm_off | 384.39 † | 646.33 † | 1545.49 † |
| attention_model_on | 384.39 † | 646.33 † | 1545.49 † |
| attention_model_off | 384.39 † | 646.33 † | 1545.49 † |
| lstm_pointer_on | 416.28 † | 674.24 † | 1666.35 † |
| lstm_pointer_off | 394.10 † | 664.76 † | 1604.27 † |

† Checkpoint not loaded. Re-run eval after pulling the load fix in `src/experiments/run.py` (`_maybe_load_weights` loads any architecture). Confirm logs show `Successfully loaded policy weights from ...` for every run.

---

## Nearest-neighbor heuristic (Python)

Weak constructive baseline in [`src/heuristics`](src/heuristics): at each step the truck, then the drone, picks the **nearest feasible** node under the same `Env` availability masks as RL training.

```bash
.venv/bin/python -m src.heuristics --all-sizes
```

Results are written under `results/heuristics/` (`nearest_neighbor_summary.md`, per-size `summary.json` + `makespans.txt`).

| N | Mean makespan ± std | Gap vs TSP-ep-all | Time |
| ---: | ---: | ---: | ---: |
| 11 | 302.32 ± 39.73 | 31.5% | &lt;0.01 s |
| 15 | 353.10 ± 39.76 | — | &lt;0.01 s |
| 20 | 406.10 ± 47.54 | 44.2% | 0.01 s |
| 50 | 649.48 ± 58.93 | 63.5% | 0.02 s |
| 100 | 913.21 ± 60.55 | 70.5% | 0.04 s |

---

## Quick comparison (mean cost)

| Method | N = 11 | N = 20 | N = 50 | N = 100 |
| --- | ---: | ---: | ---: | ---: |
| TSP-ep-all | **229.87** | **281.62** | **397.21** | **535.50** |
| DPS/10 | 229.95 | 292.23 | 420.51 | 570.74 |
| DPS/25 | — | — | 404.78 | 548.23 |
| tspd_lstm_on (greedy, small) | 273.85 | 351.07 | 574.73 | — |
| Nearest neighbor | 302.32 | 406.10 | 649.48 | 913.21 |

`tspd_lstm_on` beats nearest-neighbor at every reported `N`, but still trails TSP-ep-all (gaps 19–45%). Other RL architectures pending a clean re-eval with checkpoint load fixed.
