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

## RL greedy eval (`scale=small`)

Greedy test (`action=test`) after the decoder × dynamics × `N ∈ {11,20,50}` training matrix. All rows loaded checkpoints successfully.

| Method | N = 11 | Gap | N = 20 | Gap | N = 50 | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **tspd_lstm_on** | **273.85** | **19.1%** | **351.07** | **24.7%** | **574.73** | **44.7%** |
| tspd_lstm_off | 282.53 | 22.9% | 378.59 | 34.4% | 1050.84 | 164.6% |
| attention_model_on | 283.20 | 23.2% | 375.00 | 33.2% | 717.99 | 80.8% |
| attention_model_off | 286.46 | 24.6% | 365.53 | 29.8% | 816.93 | 105.7% |
| lstm_pointer_on | 288.97 | 25.7% | 376.37 | 33.6% | 1026.50 | 158.4% |
| lstm_pointer_off | 284.98 | 24.0% | 367.41 | 30.5% | 1049.83 | 164.3% |

**Takeaways (this `small` matrix):**

- Best RL at every `N`: **tspd_lstm_on**.
- Dynamics **on** helps most at N = 50 (especially `tspd_lstm`: 574.73 vs 1050.84).
- All RL methods beat nearest-neighbor at N = 11 and N = 20; at N = 50 only `tspd_lstm_on` and both attention_model variants beat NN (649.48).
- All still trail TSP-ep-all (best RL gaps ~19–45%).

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
| tspd_lstm_on | **273.85** | **351.07** | **574.73** | — |
| tspd_lstm_off | 282.53 | 378.59 | 1050.84 | — |
| attention_model_on | 283.20 | 375.00 | 717.99 | — |
| attention_model_off | 286.46 | 365.53 | 816.93 | — |
| lstm_pointer_on | 288.97 | 376.37 | 1026.50 | — |
| lstm_pointer_off | 284.98 | 367.41 | 1049.83 | — |
| Nearest neighbor | 302.32 | 406.10 | 649.48 | 913.21 |

---

## Alpha sweep (`tspd_lstm_on`, `scale=small`)

Greedy test (`action=test`) for **tspd_lstm** + **dynamics=on** across drone/truck speed ratios `alpha ∈ {1.0, 1.2, 1.5}` (`v_t = 1.0`, `v_d = alpha`). Checkpoints under `outputs/training/small/tspd_lstm_on_n{N}_a{alpha}/`.

These settings differ from the Table 2 / sections above (`alpha = 2.0` / `v_d = 2.0`); do not compare gaps to TSP-ep-all from those tables.

| α (`v_d`) | N = 11 | N = 20 |
| ---: | ---: | ---: |
| 1.0 | 315.38 | 387.77 |
| 1.2 | 295.50 | 379.40 |
| 1.5 | 283.78 | 361.91 |

For reference, the same architecture at **α = 2.0** (main RL table): N = 11 → **273.85**, N = 20 → **351.07**.

**Takeaways:**

- Faster drone (higher α) lowers greedy makespan at both sizes.
- N = 11: α 1.0 → 1.5 cuts ~10% (315.38 → 283.78); α 2.0 is another ~3.5% lower (273.85).
- N = 20: α 1.0 → 1.5 cuts ~6.7% (387.77 → 361.91); α 2.0 is another ~3% lower (351.07).
