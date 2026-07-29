# TSP-D Benchmarks

Reference costs for the Traveling Salesman Problem with Drone (TSP-D), aligned with the paper’s Table 2 setting and this repo’s test files (`data/DroneTruck-size-100-len-*.txt`).

**Instance settings:** truck speed `v_t = 1.0`, drone speed `v_d = 2.0` (cost factor `0.5`), unlimited drone range, **100** instances per size.  
**Gap %:** relative to the best mean cost among OR methods for that `N` (same convention as the paper / `reproduce_table2.jl`).

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

Use this as a **lower-performance floor**: any trained RL policy should beat nearest-neighbor; OR methods (especially TSP-ep-all) set the high bar.

---

## Quick comparison (mean cost)

| Method | N = 11 | N = 20 | N = 50 | N = 100 |
| --- | ---: | ---: | ---: | ---: |
| TSP-ep-all | **229.87** | **281.62** | **397.21** | **535.50** |
| DPS/10 | 229.95 | 292.23 | 420.51 | 570.74 |
| DPS/25 | — | — | 404.78 | 548.23 |
| Nearest neighbor | 302.32 | 406.10 | 649.48 | 913.21 |
