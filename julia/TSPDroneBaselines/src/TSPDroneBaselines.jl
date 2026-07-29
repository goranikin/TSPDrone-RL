"""
TSP-ep-all and DPS heuristics for TSP-D (Table 2 baselines).

Algorithm logic adapted from TSPDrone.jl (MIT):
https://github.com/kaist-comet/TSPDrone.jl

This package does **not** depend on TSPDrone.jl. The only external solver
dependency is Concorde.jl for the initial TSP tour (as in the paper).
"""
module TSPDroneBaselines

using Concorde
using Statistics

const MAX_TIME_LIMIT = Inf
const MAX_DRONE_RANGE = Inf

mutable struct TSPDroneResult
    total_cost::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    Ct::Matrix{Float64}
    Cd::Matrix{Float64}
    flying_range::Float64
end

include("tspd_utils.jl")
include("tsp_ep_all.jl")
include("DPS.jl")
include("solve.jl")
include("io.jl")

export TSPDroneResult,
    solve_tspd,
    load_instances,
    evaluate_dataset,
    print_summary,
    MAX_TIME_LIMIT,
    MAX_DRONE_RANGE

end # module
