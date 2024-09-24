using Statistics 
using JLD2
using LinearAlgebra

Σ_true = 2.0

function rmse(P_pred, P_true = Σ_true)
    # Ensure both matrices are square and diagonal
    # P_pred = Diagonal(P_pred)
    @assert size(P_pred) == size(P_true) "Matrices must have the same size"
    
    # Extract diagonal elements
    # diag_pred = diagm(P_pred)
    # diag_true = diagm(P_true)
    diag_pred = P_pred
    diag_true = P_true
    
    # Compute the squared differences
    squared_diff = (diag_pred .- diag_true).^2
    
    # Compute the mean squared error (MSE)
    mse = mean(squared_diff)
    
    # Compute and return the root mean squared error (RMSE)
    return sqrt(mse)
end

@load "bilqr_cartpolepartial_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

# find average trace and std
# trace_bilqr_avg = mean([tr(ΣΘΘ) for ΣΘΘ in all_ΣΘΘ])
# trace_bilqr_std = std([tr(ΣΘΘ) for ΣΘΘ in all_ΣΘΘ])

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

# find average rmse and std
rmse_bilqr_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

@load "mpc_cartpolepartial_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ
trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_mpc_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_mpc_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

@load "random_cartpolepartial_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_random_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_random_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

@load "reg_cartpolefull_sysid_results.jld2" all_b_ends all_mp_estimates all_mp_variances all_ΣΘΘ

trace_regression_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_regression_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_regression_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_regression_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

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
