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
    rng = Random.default_rng()
)
    validate_mga(case)
    # Make sure the least-cost model is ready before starting MGA
    termination_status(EP) == MOI.OPTIMAL || error("MGA requires an optimal least-cost solution first.")
    haskey(EP, :vMGA) && !isempty(EP[:vMGA]) || error("No MGA groups were added to the model.")
    println("MGA Module")

    mga_settings = case.settings.MGA
    slack = mga_settings.Epsilon

    # Sort the (period, group) keys so random weights and output group indices
    # do not depend on dictionary insertion order
    mga_groups = sort!(collect(keys(EP[:vMGA])))

    # Every MGA solve can cost up to epsilon more than the least-cost solution.
    # abs keeps positive epsilon a relaxation when the objective is negative.
    least_cost = objective_value(EP)
    budget_limit = least_cost + slack * abs(least_cost)

    jobs = create_mga_jobs(mga_groups, mga_settings, rng)

    # Save the cost expression before replacing the objective for each MGA job.
    # Add the shared budget once to the baseline model.
    system_cost = objective_function(EP)
    scaling = first(get_periods(case)).settings.ConstraintScaling
    constraints_before = scaling ?
        Set(JuMP.all_constraints(EP; include_variable_in_set_constraints = true)) : nothing

    @constraint(EP, mga_budget, system_cost <= budget_limit)
    if scaling
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
        termination_status(EP) == MOI.OPTIMAL ||
            error("MGA $direction iteration $iteration group $group_index failed: $(termination_status(EP))")

        suffix = group_index == 0 ? "" : "_group_$(group_index)"
        outpath = joinpath(path, "MGAResults_$direction", "MGA_$(slack)_$(iteration)$(suffix)")
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
