#!/usr/bin/env julia
# Reproduce Table 2 OR baselines: TSP-ep-all, DPS/10, DPS/25.
#
# Usage (from repo root):
#   julia --project=julia/TSPDroneBaselines julia/TSPDroneBaselines/scripts/reproduce_table2.jl
#   julia --project=julia/TSPDroneBaselines julia/TSPDroneBaselines/scripts/reproduce_table2.jl --n 20 --methods TSP-ep-all,DPS/10 --limit 5

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

using TSPDroneBaselines
using Statistics

function parse_args(args)
    cfg = Dict(
        "n" => [20, 50, 100],
        "methods" => ["TSP-ep-all", "DPS/10", "DPS/25"],
        "limit" => 0,
        "data_dir" => joinpath(@__DIR__, "..", "..", "..", "data"),
        "truck" => 1.0,
        "drone" => 0.5,
    )
    i = 1
    while i <= length(args)
        if args[i] == "--n" && i < length(args)
            cfg["n"] = parse.(Int, split(args[i + 1], ","))
            i += 2
        elseif args[i] == "--methods" && i < length(args)
            cfg["methods"] = split(args[i + 1], ",")
            i += 2
        elseif args[i] == "--limit" && i < length(args)
            cfg["limit"] = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--data-dir" && i < length(args)
            cfg["data_dir"] = args[i + 1]
            i += 2
        elseif args[i] in ("-h", "--help")
            println(
                """
                reproduce_table2.jl [options]
                  --n 20,50,100          problem sizes (include depot)
                  --methods TSP-ep-all,DPS/10,DPS/25
                  --limit K              evaluate first K instances (0 = all)
                  --data-dir PATH        default: <repo>/data
                """,
            )
            exit(0)
        else
            error("Unknown argument: $(args[i])")
        end
    end
    return cfg
end

function n_groups_for(method::AbstractString, n::Int)
    if method == "TSP-ep-all"
        return 1, true
    elseif startswith(method, "DPS/")
        g = parse(Int, split(method, "/")[2])
        if n < g
            return 0, false  # not applicable (paper shows –)
        end
        return max(1, n ÷ g), true
    else
        error("Unknown method label: $method")
    end
end

function main(args)
    cfg = parse_args(args)
    data_dir = abspath(cfg["data_dir"])

    println("="^72)
    println("TSP-D Table 2 baselines (TSP-ep-all / DPS)")
    println("data_dir=$data_dir")
    println("truck=$(cfg["truck"]) drone=$(cfg["drone"])")
    println("="^72)

    # Collect all costs per (n, method) for gap vs best mean later
    results = Dict{Tuple{Int,String},NamedTuple}()

    for n in cfg["n"]
        path = joinpath(data_dir, "DroneTruck-size-100-len-$n.txt")
        if !isfile(path)
            @warn "Missing $path — skip n=$n"
            continue
        end
        instances = load_instances(path, n)
        println("\nLoaded $(length(instances)) instances for N=$n")

        for method in cfg["methods"]
            ng, ok = n_groups_for(method, n)
            if !ok
                println("  $method: – (N=$n < group size)")
                continue
            end
            println("  Running $method (n_groups=$ng) ...")
            stats = evaluate_dataset(
                instances;
                n_groups=ng,
                method="TSP-ep-all",
                truck_cost_factor=cfg["truck"],
                drone_cost_factor=cfg["drone"],
                limit=cfg["limit"],
            )
            results[(n, method)] = stats
            println(
                "  → cost=$(round(stats.mean_cost; digits=2))±$(round(stats.std_cost; digits=2))  " *
                "time=$(round(stats.mean_time; digits=2)) s",
            )
        end
    end

    # Print table with gaps vs best mean cost among evaluated methods for that N
    println("\n" * "="^72)
    println("Summary (gap % vs best mean cost among methods run for that N)")
    println("="^72)
    for n in cfg["n"]
        method_keys = [m for m in cfg["methods"] if haskey(results, (n, m))]
        isempty(method_keys) && continue
        best = minimum(results[(n, m)].mean_cost for m in method_keys)
        println("\nN = $n  (best mean cost = $(round(best; digits=2)))")
        for m in method_keys
            s = results[(n, m)]
            gap = 100 * (s.mean_cost - best) / abs(best)
            println(
                "  $(rpad(m, 12))  " *
                "$(lpad(string(round(s.mean_cost; digits=2)), 8))±" *
                "$(rpad(string(round(s.std_cost; digits=2)), 6))  " *
                "gap=$(lpad(string(round(gap; digits=2)), 5))%  " *
                "($(round(s.mean_time; digits=2)) s)",
            )
        end
    end
end

main(ARGS)
