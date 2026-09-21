mga_enabled(case::Case) = case.settings.MGA.Enabled

function validate_mga(case::Case)
    mga_enabled(case) || return nothing
    expansion_horizon(case) isa PerfectForesight ||
        error("MGA requires PerfectForesight; myopic runs are not supported.")
    solution_algorithm(case) isa Monolithic ||
        error("MGA currently requires the Monolithic solution algorithm.")
    return nothing
end

function run_mga(
    case::Case,
    EP::Model,
    path::AbstractString;
    rng = nothing,
    least_cost_original = nothing
)
    validate_mga(case)
    # An externally supplied baseline permits MGA without a least-cost solve.
    if isnothing(least_cost_original)
        termination_status(EP) == MOI.OPTIMAL || error("MGA requires an optimal least-cost solution first.")
    else
        isfinite(least_cost_original) && least_cost_original > 0 ||
            throw(ArgumentError("least_cost_original must be finite and positive."))
    end
    haskey(EP, :vMGA) && !isempty(EP[:vMGA]) || error("No MGA groups were added to the model.")
    println("MGA Module")

    mga_settings = case.settings.MGA
    slack = mga_settings.Epsilon
    if isnothing(rng)
        seed = get(mga_settings, :RandomSeed, nothing)
        rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    end

    # Sort the (period, group) keys so random weights and output group indices
    # do not depend on dictionary insertion order
    mga_groups = sort!(collect(keys(EP[:vMGA])))

    # Every MGA solve can cost up to epsilon more than the least-cost solution.
    # abs keeps positive epsilon a relaxation when the objective is negative.
    parameter_scale = parameter_scaling_factor(get_settings(case))
    least_cost = isnothing(least_cost_original) ? objective_value(EP) :
        least_cost_original / parameter_scale^2
    budget_limit = least_cost + slack * abs(least_cost)

    jobs = create_mga_jobs(mga_groups, mga_settings, rng)

    # Save the cost expression before replacing the objective for each MGA job.
    # Add the shared budget once to the baseline model.
    system_cost = objective_function(EP)
    scaling = first(get_periods(case)).settings.ConstraintScaling
    constraints_before = scaling ?
        Set(JuMP.all_constraints(EP; include_variable_in_set_constraints = true)) : nothing

    cost_coefficients = [abs(coefficient) for (coefficient, _) in JuMP.linear_terms(system_cost)
                         if !iszero(coefficient)]
    isempty(cost_coefficients) && error("The system cost has no variable coefficients.")
    budget_row_scale = max(1.0, min(least_cost, minimum(cost_coefficients) / 1e-3))
    mkpath(path)
    open(joinpath(path, "mga_budget_coefficients.txt"), "w") do io
        report_mga_budget_coefficients(io, system_cost; divisor=budget_row_scale,
            budget_limit=budget_limit)
    end
    println("MGA budget coefficient report: ", joinpath(path, "mga_budget_coefficients.txt"))
    @constraint(EP, mga_budget,
        system_cost / budget_row_scale <= budget_limit / budget_row_scale)
    println("Parameter scale=$parameter_scale; MGA budget row: terms=$(length(cost_coefficients)), " *
            "divisor=$budget_row_scale, RHS=$(budget_limit / budget_row_scale), " *
            "coefficient range=[$(minimum(cost_coefficients) / budget_row_scale), " *
            "$(maximum(cost_coefficients) / budget_row_scale)]")
    if scaling && budget_row_scale == 1.0
        # Scaling may replace the budget with several constraints. Record all
        # of them so later cost-based pricing can remove the MGA budget.
        set_name(mga_budget, "mga_budget")
        scale_constraints!(ConstraintRef[mga_budget])
        EP[:cMGABudget] = filter(
            constraint -> constraint ∉ constraints_before,
            JuMP.all_constraints(EP; include_variable_in_set_constraints = true))
    else
        EP[:cMGABudget] = ConstraintRef[mga_budget]
    end

    # Reuse EP, changing only its objective for each MGA solve.
    results = NamedTuple[]
    output_dirs = Dict{String,String}()
    for job in jobs
        (; iteration, group_index, weights, direction, sense) = job
        # RandomVector uses every group; VariableMinMax uses one group.
        if group_index == 0
            @objective(EP, sense,
                sum(weights[k] * EP[:vMGA][group] for (k, group) in enumerate(mga_groups)))
        else
            @objective(EP, sense, EP[:vMGA][mga_groups[group_index]])
        end

        optimize!(EP)

        #### START TEMPORARY DIAGNOSITCS
        suffix = group_index == 0 ? "" : "_group_$(group_index)"
        direction_path = get!(output_dirs, direction) do
            create_mga_output_dir(joinpath(path, "results"), direction)
        end
        outpath = joinpath(direction_path, "MGA_$(slack)_$(iteration)$(suffix)")
        mkpath(outpath)

        # Save diagnostics before validation so failed solves also leave a report.
        diagnostics = open(joinpath(outpath, "numerical_diagnostics.txt"), "w") do io
            redirect_stdout(io) do
                diagnose_numerics(
                    EP;
                    top_n = 30,
                    tiny_threshold = 1e-8,
                    large_threshold = 1e8,
                )
            end
        end

        ### END TEMPORARY DIAGNOSTICS

        if has_values(EP)
            model_cost = value(system_cost)
            println("MGA $direction: status=$(termination_status(EP)), " *
                    "system cost=$(model_cost * parameter_scale^2), " *
                    "budget=$(budget_limit * parameter_scale^2)")
            if !isnothing(least_cost_original)
                model_cost <= budget_limit + max(1e-6 * budget_row_scale, 0.05 * slack * abs(least_cost)) || error("MGA $direction violated the cost budget.")
            end
        end

        status = termination_status(EP)

        status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) || error("MGA $direction iteration $iteration group $group_index failed: $status")

        primal_status(EP) == MOI.FEASIBLE_POINT || error("MGA $direction returned an invalid primal solution: $(primal_status(EP))")

        postprocess!(case, EP)
        write_outputs(outpath, case, EP)

        # Keep the original system cost alongside the MGA objective value.
        summary = (
            iteration = iteration,
            direction = direction,
            mga_objective = objective_value(EP),
            objective_function = value(system_cost),
            least_cost = least_cost,
            budget_limit = budget_limit
        )
        if group_index != 0
            summary = merge(summary,
                (group_index = group_index, group = string(mga_groups[group_index])))
        end

        CSV.write(joinpath(outpath, "mga_summary.csv"), DataFrame([summary]))
        push!(results, summary)
    end
    return results
end

"""Report extreme nonzero budget coefficients before constraint-scaling transformations.

Keeps only 2*top_n terms in memory, even for multi-million-term budgets.
Coefficients are ranked by magnitude; signed values and variable names are retained.
"""
function report_mga_budget_coefficients(io::IO, cost; divisor=1.0,
    budget_limit=nothing, top_n::Int=30)
    top_n > 0 || throw(ArgumentError("top_n must be positive"))
    isfinite(divisor) && divisor > 0 || throw(ArgumentError("divisor must be finite and positive"))
    smallest, largest = [], []
    count = 0
    for (coefficient, variable) in JuMP.linear_terms(cost)
        iszero(coefficient) && continue
        count += 1
        magnitude = abs(coefficient)
        entry = (; magnitude, coefficient, variable)
        if length(smallest) < top_n || magnitude < last(smallest).magnitude
            push!(smallest, entry)
            sort!(smallest; by=x -> x.magnitude)
            length(smallest) > top_n && pop!(smallest)
        end
        if length(largest) < top_n || magnitude > last(largest).magnitude
            push!(largest, entry)
            sort!(largest; by=x -> x.magnitude, rev=true)
            length(largest) > top_n && pop!(largest)
        end
    end
    println(io, "MGA budget coefficients BEFORE constraint scaling/presolve")
    println(io, "Ranked by absolute coefficient, not by coefficient × solution value.")
    println(io, "Coefficients are in model units; no physical-unit conversion is applied.")
    println(io, "Nonzero terms: ", count, "; row divisor: ", divisor)
    println(io, "Cost constant: ", JuMP.constant(cost))
    if !isnothing(budget_limit)
        println(io, "Budget limit / divisor: ", budget_limit / divisor)
        println(io, "RHS after moving constant: ", (budget_limit - JuMP.constant(cost)) / divisor)
    end
    if count > 0
        println(io, "Max/min coefficient magnitude ratio: ", first(largest).magnitude / first(smallest).magnitude)
    end
    for (label, entries) in (("SMALLEST", smallest), ("LARGEST", largest))
        println(io, "\n", label, " ", length(entries), " NONZERO COEFFICIENT MAGNITUDES")
        println(io, "rank\tcost_coefficient\tbudget_row_coefficient\tvariable_index\tvariable_name")
        for (rank, entry) in enumerate(entries)
            variable_name = JuMP.name(entry.variable)
            isempty(variable_name) && (variable_name = string(entry.variable))
            println(io, rank, '\t', entry.coefficient, '\t', entry.coefficient / divisor,
                '\t', JuMP.index(entry.variable).value, '\t', variable_name)
        end
    end
    return nothing
end

"""Reserve the next numbered MGA folder without overwriting another run."""
function create_mga_output_dir(results_path::AbstractString, direction::AbstractString)
    mkpath(results_path)
    prefix = "MGAResults_$(direction)_"
    run_number = 1
    for entry in readdir(results_path)
        startswith(entry, prefix) || continue
        number = tryparse(Int, chop(entry; head = length(prefix), tail = 0))
        isnothing(number) || (run_number = max(run_number, number + 1))
    end
    while true
        outpath = joinpath(results_path, prefix * lpad(string(run_number), 2, '0'))
        try
            mkdir(outpath)
            return outpath
        catch
            # Another process may have reserved this number since readdir.
            ispath(outpath) || rethrow()
            run_number += 1
        end
    end
end

"""Create MGA objectives in solve order, pairing max and min for each vector or group."""
function create_mga_jobs(groups, settings, rng)
    jobs = NamedTuple[]
    if settings.MGAAlgorithm == "RandomVector"
        for iteration in 1:settings.NumIterations
            weights = rand(rng, length(groups))
            for (direction, sense) in (("max", MOI.MAX_SENSE), ("min", MOI.MIN_SENSE))
                push!(jobs, (; iteration, group_index = 0, weights, direction, sense))
            end
        end
    elseif settings.MGAAlgorithm == "VariableMinMax"
        for group_index in eachindex(groups)
            for (direction, sense) in (("max", MOI.MAX_SENSE), ("min", MOI.MIN_SENSE))
                push!(jobs, (; iteration = 1, group_index, weights = nothing, direction, sense))
            end
        end
    else
        throw(ArgumentError("Unknown MGAAlgorithm: $(settings.MGAAlgorithm). Expected RandomVector or VariableMinMax."))
    end
    return jobs
end

const MGA_VALID_GROUPINGS = ("technology", "location", "custom")

"""
    mga_group_component(e::AbstractEdge, grouping::AbstractString, edge_asset_map)

Return the Symbol identifying which sub-group `e` belongs to for one grouping
dimension ("technology", "location", or "custom").
"""
function mga_group_component(e::AbstractEdge, grouping::AbstractString, edge_asset_map)
    if grouping == "technology"
        return nameof(typeof(edge_asset_map[id(e)][]))
    elseif grouping == "location"
        if ismissing(e.location)
            @warn "Edge $(id(e)) has no location for MGA grouping; will be grouped with other edges without a location."
        end
        return coalesce(e.location, :none)
    elseif grouping == "custom"
        group = mga_group(e)
        if ismissing(group)
            @warn "Edge $(id(e)) has no custom group for MGA grouping; will be grouped with other edges without a custom group."
        end
        return coalesce(group, :none)
    else
        error("Unknown MGA grouping: \"$grouping\". Allowed values are $MGA_VALID_GROUPINGS.")
    end
end

"""
    add_mga_variables(system::System, EP::Model, settings::NamedTuple)

Add capacity or weighted annual-flow aggregates for each MGA group in this
period. Edges are included in MGA if `mga_enabled` is true or `mga_group` is set.
Groups are formed by combining the dimensions in `settings.Groupings`
(any subset of "technology", "location", "custom").
"""
function add_mga_variables(system::System, EP::Model, settings::NamedTuple)
    groupings = settings.Groupings
    quantity = settings.Quantity
    period = period_index(system)

    # Make sure MGA groupngs are valid
    unknown = setdiff(groupings, MGA_VALID_GROUPINGS)
    isempty(unknown) || error("Unknown MGA grouping(s): $unknown. Allowed values are $MGA_VALID_GROUPINGS.")

   # Get aray of edges and the map of edges to assets (so you can find the technology type of an edge if needed)
    edges, edge_asset_map = get_edges(system; return_ids_map=true)

    # Find edges to be included in mga, which are any edges that have mga_enabled = true or a non-missing mga_group
    edges_included_in_mga = filter(e -> (mga_enabled(e) == true) || (mga_group(e) !== missing), edges)

    # Define a function that makes a tuple of the grouping components for an edge
    # Eg: mga_group_key(e) -> (:SolarPV, :Zone1)
    mga_group_key(e) = Tuple(mga_group_component(e, g, edge_asset_map) for g in groupings)

    # Sort edges into groups based on their mga_group_key
    edge_groups = Dict{NTuple{length(groupings), Symbol}, Vector{AbstractEdge}}()
    for e in edges_included_in_mga
        key = mga_group_key(e)

        # If this key doesn't exist in the dictionary yet, create a new entry with an empty vector
        if !haskey(edge_groups, key)
            edge_groups[key] = Vector{AbstractEdge}()
        end

        push!(edge_groups[key], e)
    end

    # Store each period's variables and constraints
    if !haskey(EP, :vMGA)
        EP[:vMGA] = Dict{Tuple{Int,NTuple{length(groupings),Symbol}},VariableRef}()
        EP[:cMGA] = Dict{Tuple{Int,NTuple{length(groupings),Symbol}},ConstraintRef}()
    end

    # Add MGA variables and constraints for each group in this period, based on the selected quantity (capacity or annual flow)
    for (group, mga_edges) in edge_groups
        key = (period, group)
        group_name = join(group, "_")

        # Define MGA groups. Only capacity aggregates get a lower bound of zero, since annual flow can be negative for bidirectional edges.
        vMGA = @variable(EP, base_name = "vMGA_$(period)_$(group_name)")
        if quantity == "capacity"
            set_lower_bound(vMGA, 0)
        end
        EP[:vMGA][key] = vMGA

        # Constraint to compute total annual flow or capacity for this group
        if quantity == "annual_flow"
            EP[:cMGA][key] = @constraint(EP,
                vMGA == sum(flow(e, t) * subperiod_weight(e, current_subperiod(e, t))
                    for e in mga_edges for t in time_interval(e)))
        elseif quantity == "capacity"
            EP[:cMGA][key] = @constraint(EP,
                vMGA == sum(capacity(e) for e in mga_edges))
        else
            error("Unknown MGA quantity: \"$quantity\". Allowed values are \"annual_flow\", \"capacity\".")
        end
    end

    return nothing
end
