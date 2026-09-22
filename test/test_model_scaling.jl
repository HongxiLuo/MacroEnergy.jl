import JuMP
import HiGHS
import MacroEnergyScaling

function largest_affine_constraint_coefficient(model)
    largest = 0.0
    for (function_type, set_type) in JuMP.list_of_constraint_types(model)
        for constraint in JuMP.all_constraints(model, function_type, set_type)
            func = JuMP.constraint_object(constraint).func
            func isa JuMP.AffExpr || continue
            for coefficient in values(func.terms)
                largest = max(largest, abs(coefficient))
            end
        end
    end
    return largest
end

Test.@testset "Objective scaling and MGA budget" begin
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, 1 <= x <= 10)
    JuMP.@variable(model, 1 <= y <= 10)
    JuMP.@constraint(model, 1e12 * x + y >= 1e12 + 1)
    JuMP.@objective(model, Min, 1e-4 * x + 1e7 * y)
    Test.@test largest_affine_constraint_coefficient(model) == 1e12

    settings = MacroEnergyScaling.ScalingSettings(scale_objective_uniformly=false)
    MacroEnergyScaling.scale_objective!(model, settings)
    MacroEnergy.scale_constraints!(model, settings)
    MacroEnergy.remove_proxy_bounds!(settings)
    Test.@test largest_affine_constraint_coefficient(model) <= settings.coeff_ub
    JuMP.optimize!(model)

    Test.@test JuMP.termination_status(model) == JuMP.MOI.OPTIMAL
    cost = JuMP.objective_function(model)
    least_cost = JuMP.objective_value(model)
    Test.@test least_cost ≈ 1e-4 + 1e7
    Test.@test settings.objective_scaling_factor == 1.0

    budget_limit = 1.1 * least_cost
    budget = JuMP.@constraint(model, cost <= budget_limit)
    MacroEnergy.scale_constraints!([budget], settings)
    MacroEnergy.remove_proxy_bounds!(settings)
    Test.@test largest_affine_constraint_coefficient(model) <= settings.coeff_ub
    JuMP.@objective(model, Max, x)
    JuMP.optimize!(model)

    Test.@test JuMP.termination_status(model) == JuMP.MOI.OPTIMAL
    Test.@test JuMP.value(cost) <= budget_limit + 1e-5
end
