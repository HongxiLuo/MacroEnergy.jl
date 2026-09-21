"""
    diagnose_numerics(model; top_n=20, tiny_threshold=1e-8, large_threshold=1e8)

Diagnose numerical issues after an optimization solve.

Reports:
1. Solver/solution status
2. Largest primal constraint violations
3. Largest variable values
4. Smallest nonzero variable values
5. Variables with extreme magnitudes

Useful for diagnosing Gurobi uncrush violations and scaling problems.
"""
function diagnose_numerics(
    model::JuMP.Model;
    top_n::Int = 20,
    tiny_threshold::Float64 = 1e-8,
    large_threshold::Float64 = 1e8,
)

    println("\n", "="^80)
    println("NUMERICAL DIAGNOSTICS")
    println("="^80)

    # ------------------------------------------------------------------
    # 1. SOLVER STATUS
    # ------------------------------------------------------------------

    println("\n--- Solver status ---")
    println("Termination status: ", termination_status(model))
    println("Primal status:      ", primal_status(model))
    println("Dual status:        ", dual_status(model))

    if !has_values(model)
        println("\nNo primal solution is available.")
        return nothing
    end

    # ------------------------------------------------------------------
    # 2. CONSTRAINT VIOLATIONS
    # ------------------------------------------------------------------

    println("\n--- Constraint feasibility ---")

    report = primal_feasibility_report(model)

    violations = sort(
        collect(report),
        by = x -> x.second,
        rev = true,
    )

    if isempty(violations)
        println("No constraint violations reported.")
    else
        max_viol = violations[1].second

        println("Maximum primal violation: ", max_viol)
        println("\nTop $(min(top_n, length(violations))) constraint violations:")

        for (i, pair) in enumerate(Iterators.take(violations, top_n))
            con  = pair.first
            viol = pair.second

            println("\n[$i]")
            println("  Violation:  ", viol)

            # Constraint name, if available
            try
                println("  Name:       ", name(con))
            catch
                println("  Name:       <unavailable>")
            end

            println("  Constraint: ", con)
        end
    end

    # ------------------------------------------------------------------
    # 3. COLLECT VARIABLE INFORMATION
    # ------------------------------------------------------------------

    var_info = []

    for var in all_variables(model)
        val = value(var)

        if !isfinite(val)
            continue
        end

        lb = has_lower_bound(var) ? lower_bound(var) : -Inf
        ub = has_upper_bound(var) ? upper_bound(var) : Inf

        push!(
            var_info,
            (
                variable = var,
                name = name(var),
                value = val,
                abs_value = abs(val),
                lower_bound = lb,
                upper_bound = ub,
            ),
        )
    end

    # ------------------------------------------------------------------
    # 4. LARGEST VARIABLES
    # ------------------------------------------------------------------

    largest = sort(
        var_info,
        by = x -> x.abs_value,
        rev = true,
    )

    println("\n", "-"^80)
    println("LARGEST VARIABLE VALUES")
    println("-"^80)

    for (i, x) in enumerate(Iterators.take(largest, top_n))
        println(
            "[$i] |x| = ", x.abs_value,
            "   x = ", x.value,
            "   [", x.lower_bound, ", ", x.upper_bound, "]",
            "   ", x.name,
        )
    end

    # ------------------------------------------------------------------
    # 5. SMALLEST NONZERO VARIABLES
    # ------------------------------------------------------------------

    nonzero = filter(x -> x.abs_value > 0.0, var_info)

    smallest = sort(
        nonzero,
        by = x -> x.abs_value,
    )

    println("\n", "-"^80)
    println("SMALLEST NONZERO VARIABLE VALUES")
    println("-"^80)

    for (i, x) in enumerate(Iterators.take(smallest, top_n))
        println(
            "[$i] |x| = ", x.abs_value,
            "   x = ", x.value,
            "   [", x.lower_bound, ", ", x.upper_bound, "]",
            "   ", x.name,
        )
    end

    # ------------------------------------------------------------------
    # 6. EXTREME VARIABLES
    # ------------------------------------------------------------------

    tiny_vars = filter(
        x -> 0.0 < x.abs_value < tiny_threshold,
        var_info,
    )

    large_vars = filter(
        x -> x.abs_value > large_threshold,
        var_info,
    )

    println("\n", "-"^80)
    println("EXTREME VARIABLE SUMMARY")
    println("-"^80)

    println(
        "Tiny nonzero variables (0 < |x| < ",
        tiny_threshold,
        "): ",
        length(tiny_vars),
    )

    println(
        "Large variables (|x| > ",
        large_threshold,
        "): ",
        length(large_vars),
    )

    # ------------------------------------------------------------------
    # 7. MAGNITUDE DISTRIBUTION
    # ------------------------------------------------------------------

    println("\n", "-"^80)
    println("VARIABLE MAGNITUDE DISTRIBUTION")
    println("-"^80)

    bins = [
        (0.0,      1e-12),
        (1e-12,    1e-9),
        (1e-9,     1e-6),
        (1e-6,     1e-3),
        (1e-3,     1.0),
        (1.0,      1e3),
        (1e3,      1e6),
        (1e6,      1e9),
        (1e9,      1e12),
        (1e12,     Inf),
    ]

    zero_count = count(x -> x.abs_value == 0.0, var_info)
    println("exactly zero: ", zero_count)

    for (lo, hi) in bins
        n = count(
            x -> x.abs_value > lo && x.abs_value <= hi,
            var_info,
        )

        println(
            "(", lo, ", ", hi, "] : ",
            n,
        )
    end

    println("\n", "="^80)
    println("END NUMERICAL DIAGNOSTICS")
    println("="^80)

    return (
        violations = violations,
        variables = var_info,
        tiny_variables = tiny_vars,
        large_variables = large_vars,
    )
end
