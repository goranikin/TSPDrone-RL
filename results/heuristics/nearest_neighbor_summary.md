# Nearest-neighbor (smallest-distance) TSP-D heuristic

Constructive baseline: truck then drone each pick the **nearest feasible** node under the same availability rules as the RL env.

| n_nodes | instances | mean makespan | std | unfinished | time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 100 | 302.3208 | 39.7336 | 0 | 0.003 |
| 15 | 100 | 353.1024 | 39.7558 | 0 | 0.003 |
| 20 | 100 | 406.1017 | 47.5392 | 0 | 0.005 |
| 50 | 100 | 649.4834 | 58.9256 | 0 | 0.015 |
| 100 | 100 | 913.2092 | 60.5549 | 0 | 0.036 |
