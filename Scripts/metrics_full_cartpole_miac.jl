using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings

Σ_true = 2.0

function rmse(P_pred, P_true = Σ_true)
    @assert size(P_pred) == size(P_true) "Matrices must have the same size"
    diag_pred = P_pred
    diag_true = P_true
    squared_diff = (diag_pred .- diag_true).^2
    mse = mean(squared_diff)
    return sqrt(mse)
end

function log_prob_gaussian(x, mean, variance)
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
t = 50
plotting_seed = 6
time_steps = collect(1:t)

@load "regression_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

# Calculate trace statistics
trace_values = [ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]
trace_regression_avg = mean(trace_values)
trace_regression_std = std(trace_values)
trace_regression_ste = trace_regression_std / sqrt(length(trace_values))

# Calculate RMSE statistics
rmse_values = [rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]
rmse_regression_avg = mean(rmse_values)
rmse_regression_std = std(rmse_values)
rmse_regression_ste = rmse_regression_std / sqrt(length(rmse_values))

log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot(time_steps, log_probs_regression, label="Regression", title = "MIAC")

@load "random_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std/sqrt(length(all_ΣΘΘ))

rmse_random_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_random_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_random_ste = rmse_random_std/sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

@load "bilqr_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

rmse_bilqr_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_ste = rmse_bilqr_std/sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_bilqr, label="BiLQR")

@load "mpcreg_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

# Calculate trace statistics
trace_values = [ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]
trace_mpcreg_avg = mean(trace_values)
trace_mpcreg_std = std(trace_values)
trace_mpcreg_ste = trace_mpcreg_std / sqrt(length(trace_values))

# Calculate RMSE statistics
rmse_values = [rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]
rmse_mpcregavg = mean(rmse_values)
rmse_mpcreg_std = std(rmse_values)
rmse_mpcreg_ste = rmse_mpcreg_std / sqrt(length(rmse_values))

log_probs_mpcreg = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_mpcreg, label="MPC + Regression")

@load "mpc_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_ste = trace_mpc_std/sqrt(length(all_ΣΘΘ))

rmse_mpc_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_mpc_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_mpc_ste = rmse_mpc_std/sqrt(length(all_ΣΘΘ))

log_probs_mpc = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_mpc, label="MPC + EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random + Regression: ", trace_regression_avg, " ± ", trace_regression_ste)
println("Random + EKF: ", trace_random_avg, " ± ", trace_random_ste)
println("MPC + Regression: ", trace_mpcreg_avg, " ± ", trace_mpcreg_ste)
println("MPC + EKF: ", trace_mpc_avg, " ± ", trace_mpc_ste)

println("RMSE of ΣΘΘ")
println("BiLQR: ", rmse_bilqr_avg, " ± ", rmse_bilqr_ste)
println("Random + Regression: ", rmse_regression_avg, " ± ", rmse_regression_ste)
println("Random + EKF: ", rmse_random_avg, " ± ", rmse_random_ste)
println("MPC + Regression: ", rmse_mpcregavg, " ± ", rmse_mpcreg_ste)
println("MPC + EKF: ", rmse_mpc_avg, " ± ", rmse_mpc_ste)

# hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# Set default DPI and save the plot
default(dpi=1000)
savefig("time_mp.png")