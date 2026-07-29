function local_search_functions(method::String)
    if method == "TSP-ep"
        return Function[]
    elseif method == "TSP-ep-1p"
        return Function[one_point_move]
    elseif method == "TSP-ep-2p"
        return Function[two_point_move]
    elseif method == "TSP-ep-2opt"
        return Function[two_opt_move]
    elseif method == "TSP-ep-all"
        return Function[two_point_move, one_point_move, two_opt_move]
    else
        error(
            "Unknown method=$(method). Expected one of: " *
            "TSP-ep, TSP-ep-1p, TSP-ep-2p, TSP-ep-2opt, TSP-ep-all",
        )
    end
end

"""
    solve_tspd(x, y, truck_cost_factor, drone_cost_factor; kwargs...)

Solve TSP-D with depot at index 1.

- `n_groups=1` with `method="TSP-ep-all"` → table row **TSP-ep-all**
- `n_groups = N ÷ g` with `method="TSP-ep-all"` → table row **DPS/g**
"""
function solve_tspd(
    x::Vector{Float64},
    y::Vector{Float64},
    truck_cost_factor::Float64,
    drone_cost_factor::Float64;
    n_groups::Int=1,
    method::String="TSP-ep-all",
    flying_range::Float64=MAX_DRONE_RANGE,
    time_limit::Float64=MAX_TIME_LIMIT,
    initial_tour::Union{Vector{Int},Nothing}=nothing,
)
    local_search_methods = local_search_functions(method)
    Ct, Cd = cost_matrices_with_dummy(x, y, truck_cost_factor, drone_cost_factor)

    total_tspd_len, truck_route, drone_route = divide_partition_search(
        Ct,
        Cd;
        local_search_methods=local_search_methods,
        n_groups=n_groups,
        flying_range=flying_range,
        time_limit=time_limit,
        initial_tour=initial_tour,
    )

    return TSPDroneResult(
        total_tspd_len,
        truck_route,
        drone_route,
        Ct,
        Cd,
        flying_range,
    )
end

function solve_tspd(
    truck_cost_mtx::Matrix{Float64},
    drone_cost_mtx::Matrix{Float64};
    n_groups::Int=1,
    method::String="TSP-ep-all",
    flying_range::Float64=MAX_DRONE_RANGE,
    time_limit::Float64=MAX_TIME_LIMIT,
    initial_tour::Union{Vector{Int},Nothing}=nothing,
)
    Ct, Cd = cost_matrices_with_dummy(truck_cost_mtx, drone_cost_mtx)
    local_search_methods = local_search_functions(method)

    total_tspd_len, truck_route, drone_route = divide_partition_search(
        Ct,
        Cd;
        local_search_methods=local_search_methods,
        n_groups=n_groups,
        flying_range=flying_range,
        time_limit=time_limit,
        initial_tour=initial_tour,
    )

    return TSPDroneResult(
        total_tspd_len,
        truck_route,
        drone_route,
        Ct,
        Cd,
        flying_range,
    )
end
