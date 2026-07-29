# TSPDroneBaselines (Julia)

Self-contained Julia implementation of the paper’s OR baselines:

| Table row | Call |
| --- | --- |
| **TSP-ep-all** | `n_groups=1`, `method="TSP-ep-all"` |
| **DPS/10** | `n_groups = N ÷ 10` |
| **DPS/25** | `n_groups = N ÷ 25` |

Algorithm sources are adapted from [TSPDrone.jl](https://github.com/kaist-comet/TSPDrone.jl) (MIT). This package does **not** import TSPDrone.jl.

**You cannot match Table 2 with zero external solvers.** The paper builds the initial TSP tour with **Concorde**. That is the only non-stdlib dependency (`Concorde.jl`).

---

## Environment setup

### Common requirements

- Julia ≥ 1.9
- A C compiler (Concorde builds from source on first `Pkg.build`)
- Network access once (download Concorde / QSOpt during build)

Paper instance settings: `truck_cost_factor=1.0`, `drone_cost_factor=0.5`, unlimited drone range.

### macOS (Apple Silicon or Intel)

```bash
# 1) Julia
brew install julia
# or: https://julialang.org/downloads/

# 2) C toolchain (needed for Concorde build)
xcode-select --install
# optional but often helpful on Apple Silicon:
brew install gcc

# 3) From repo root
cd /path/to/TSPDrone-RL
julia --project=julia/TSPDroneBaselines -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.build("Concorde")'

# smoke test
julia --project=julia/TSPDroneBaselines -e '
using TSPDroneBaselines
x = rand(11); y = rand(11)
r = solve_tspd(x, y, 1.0, 0.5; n_groups=1)
println(r.total_cost)
'
```

If `Pkg.build("Concorde")` fails on Apple Silicon, try:

```bash
export CC=gcc-14 CXX=g++-14   # match your brew gcc version
julia --project=julia/TSPDroneBaselines -e 'using Pkg; Pkg.build("Concorde"; verbose=true)'
```

### Ubuntu 24.04 (remote server)

```bash
# 1) System packages
sudo apt-get update
sudo apt-get install -y build-essential curl wget

# 2) Julia (example: 1.11 — pick a current LTS/stable from julialang.org)
curl -fsSL https://install.julialang.org | sh
# then open a new shell, or:
source ~/.bashrc

# 3) From repo root
cd ~/path/to/TSPDrone-RL
julia --project=julia/TSPDroneBaselines -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.build("Concorde")'

# smoke test (same as macOS)
julia --project=julia/TSPDroneBaselines -e '
using TSPDroneBaselines
x = rand(11); y = rand(11)
r = solve_tspd(x, y, 1.0, 0.5; n_groups=1)
println(r.total_cost)
'
```

Concorde is CPU-only (single-thread in our eval loop). No GPU / CUDA needed for these baselines.

---

## Reproduce Table 2 (OR rows)

Quick check on 5 instances of size 20:

```bash
julia --project=julia/TSPDroneBaselines \
  julia/TSPDroneBaselines/scripts/reproduce_table2.jl \
  --n 20 --methods TSP-ep-all,DPS/10 --limit 5
```

Full table (100 instances × sizes; **TSP-ep-all on N=100 is very slow**, hours):

```bash
julia --project=julia/TSPDroneBaselines \
  julia/TSPDroneBaselines/scripts/reproduce_table2.jl \
  --n 20,50,100 --methods TSP-ep-all,DPS/10,DPS/25
```

Data files used: `data/DroneTruck-size-100-len-{20,50,100}.txt`.

**Gap %** is vs the best mean cost among methods you ran for that `N` (same convention as the paper table discussion). Wall-clock times depend on CPU; paper used an Intel Xeon E5-2630.

---

## Library map

| Component | Needed? | Package |
| --- | --- | --- |
| Exact partitioning + local search (Agatz) | yes (in-repo Julia) | this package |
| DPS divide-and-conquer | yes (in-repo Julia) | this package |
| Initial TSP tour | yes | **Concorde.jl** → Concorde C solver |
| TSPDrone.jl | no | not used |
| PyTorch / this Python env | no | RL rows only |

Academic note: Concorde itself is free for academic research only.
