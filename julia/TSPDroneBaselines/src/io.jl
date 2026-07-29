"""
Load paper-format instances from `data/DroneTruck-size-100-len-N.txt`.

Each row is triples `(x, y, demand)`. Depot is the **last** triple in the file;
this function rotates so depot is first (required by `solve_tspd`).
"""
function load_instances(path::AbstractString, n::Int)
    instances = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        vals = parse.(Float64, split(line))
        @assert length(vals) == 3n "Expected $(3n) values for n=$n, got $(length(vals))"
        xs = [vals[i] for i in 1:3:length(vals)]
        ys = [vals[i] for i in 2:3:length(vals)]
        # depot last in file → first for solver
        x = [xs[end]; xs[1:(end - 1)]]
        y = [ys[end]; ys[1:(end - 1)]]
        @assert length(x) == n
        push!(instances, (x, y))
    end
    return instances
end

"""
Evaluate one method on a dataset.

Returns named tuple `(mean_cost, std_cost, mean_time, costs, times)`.
"""
function evaluate_dataset(
    instances;
    n_groups::Int=1,
    method::String="TSP-ep-all",
    truck_cost_factor::Float64=1.0,
    drone_cost_factor::Float64=0.5,
    flying_range::Float64=MAX_DRONE_RANGE,
    time_limit::Float64=MAX_TIME_LIMIT,
    limit::Int=0,
)
    costs = Float64[]
    times = Float64[]
    n_eval = limit > 0 ? min(limit, length(instances)) : length(instances)

    for i in 1:n_eval
        x, y = instances[i]
        t0 = time()
        result = solve_tspd(
            x,
            y,
            truck_cost_factor,
            drone_cost_factor;
            n_groups=n_groups,
            method=method,
            flying_range=flying_range,
            time_limit=time_limit,
        )
        push!(times, time() - t0)
        push!(costs, result.total_cost)
        @info "instance=$i/$n_eval cost=$(result.total_cost) time=$(times[end])s"
    end

    return (
        mean_cost=mean(costs),
        std_cost=std(costs; corrected=true),
        mean_time=mean(times),
        costs=costs,
        times=times,
    )
end
