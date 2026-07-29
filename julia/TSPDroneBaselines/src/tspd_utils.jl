# Adapted from TSPDrone.jl (MIT) — https://github.com/kaist-comet/TSPDrone.jl

function cost_matrices_with_dummy(truck_cost_mtx, drone_cost_mtx)
    Ct = [
        truck_cost_mtx truck_cost_mtx[:, 1]
        truck_cost_mtx[1, :]' 0.0
    ]
    Cd = [
        drone_cost_mtx drone_cost_mtx[:, 1]
        drone_cost_mtx[1, :]' 0.0
    ]
    return Ct, Cd
end

function _cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone)
    n_nodes = length(x)
    @assert length(x) == length(y)

    dist = zeros(Float64, n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in 1:n_nodes
            dist[i, j] = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2)
        end
    end

    Ct = speed_of_truck .* dist
    Cd = speed_of_drone .* dist
    return Ct, Cd
end

function cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone)
    xx = copy(x)
    yy = copy(y)
    push!(xx, x[1])
    push!(yy, y[1])
    return _cost_matrices_with_dummy(xx, yy, speed_of_truck, speed_of_drone)
end

function travel_cost(path::Vector{Int}, C::Matrix{T}) where {T}
    s = zero(T)
    for i in 1:(length(path) - 1)
        s += C[path[i], path[i + 1]]
    end
    return s
end

function objective_value(truck_route, drone_route, Ct, Cd)
    combined_nodes = intersect(truck_route, drone_route)
    obj_val = 0.0
    for i in 1:(length(combined_nodes) - 1)
        j1 = combined_nodes[i]
        j2 = combined_nodes[i + 1]

        t_idx1 = findfirst(x -> x == j1, truck_route)
        t_idx2 = findfirst(x -> x == j2, truck_route)
        t_cost = travel_cost(truck_route[t_idx1:t_idx2], Ct)

        d_idx1 = findfirst(x -> x == j1, drone_route)
        d_idx2 = findfirst(x -> x == j2, drone_route)
        d_cost = travel_cost(drone_route[d_idx1:d_idx2], Cd)

        obj_val += max(t_cost, d_cost)
    end
    return obj_val
end

function operation_length(origin, destination, route, cost_mtx; is_drone=false)
    o_idx = findfirst(x -> x == origin, route)
    d_idx = findfirst(x -> x == destination, route)
    sub_route = route[o_idx:d_idx]

    length = 0.0
    if !(is_drone && d_idx == o_idx + 1)
        for i in o_idx:(d_idx - 1)
            length += cost_mtx[route[i], route[i + 1]]
        end
    end
    return length, sub_route
end

function print_summary(result::TSPDroneResult)
    combined_nodes = intersect(result.truck_route, result.drone_route)
    operations = [(combined_nodes[i], combined_nodes[i + 1]) for i in 1:(length(combined_nodes) - 1)]
    op_costs = Float64[]

    for (i, op) in enumerate(operations)
        truck_length, truck_sub_route =
            operation_length(op[1], op[2], result.truck_route, result.Ct)
        drone_length, drone_sub_route =
            operation_length(op[1], op[2], result.drone_route, result.Cd; is_drone=true)
        op_length = max(truck_length, drone_length)
        push!(op_costs, op_length)

        println("Operation #$(i):")
        println("  - Truck = $(truck_length) : $(truck_sub_route)")
        println("  - Drone = $(drone_length) : $(drone_sub_route)")
        println("  - Length = $(op_length)")
    end

    @assert isapprox(sum(op_costs), result.total_cost; atol=1e-6)
    println("Total Cost = $(sum(op_costs))")
end
