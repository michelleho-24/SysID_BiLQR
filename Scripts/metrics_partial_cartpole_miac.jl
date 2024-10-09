using Statistics 
using JLD2
using LinearAlgebra
include("../Cartpole/cartpole_sysid.jl")

pomdp = CartpoleMDP()

Σ_true = 2.0

function state_cost(state_vec)
    γ = 0.9 
    Q = pomdp.Q
    cost = 0

    for t in 1:length(state_vec)
        s = state_vec[t]
        cost += -γ^t * (s' * Q * s)
    end

    return cost 
end 

@load "bilqr_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

# find average trace and std
# trace_bilqr_avg = mean([tr(ΣΘΘ) for ΣΘΘ in all_ΣΘΘ])
# trace_bilqr_std = std([tr(ΣΘΘ) for ΣΘΘ in all_ΣΘΘ])

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

@load "mpc_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ
trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

@load "random_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

@load "reg_cartpolepartial_miac_results.jld2" all_b_ends all_mp_estimates all_mp_variances all_ΣΘΘ

trace_regression_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
trace_regression_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_std)
println("MPC: ", trace_mpc_avg, " ± ", trace_mpc_std)
println("Random: ", trace_random_avg, " ± ", trace_random_std)
println("Regression: ", trace_regression_avg, " ± ", trace_regression_std)

println("RMSE of ΣΘΘ")
println("BILQR: ", rmse_bilqr_avg, " ± ", rmse_bilqr_std)
println("MPC: ", rmse_mpc_avg, " ± ", rmse_mpc_std)
println("Random: ", rmse_random_avg, " ± ", rmse_random_std)
println("Regression: ", rmse_regression_avg, " ± ", rmse_regression_std)

# # Plot the estimated mass of the pole over time
# plot(time_steps, mp_estimates, ribbon=sqrt.(mp_variances), label="Estimated mp ± 1 std dev", xlabel="Time Step", ylabel="Estimated mp", title="EKF Estimation of mp")
# hline!([mp_true], label="True mp", linestyle=:dash)

# # Show the plot
# savefig("time_mp.png")  
