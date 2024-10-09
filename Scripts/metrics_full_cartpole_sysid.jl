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
plotting_seed = 1
time_steps = collect(1:t)

@load "regression_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_regression_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_regression_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_regression_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_regression_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot(time_steps, log_probs_regression, label="Linear Regression", title = "System Identification")

@load "random_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_random_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_random_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

@load "bilqr_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])

rmse_bilqr_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot!(time_steps, log_probs_bilqr, label="BiLQR")

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_std/sqrt(50))
println("Random: ", trace_random_avg, " ± ", trace_random_std/sqrt(50))
println("Regression: ", trace_regression_avg, " ± ", trace_regression_std/sqrt(50))

println("RMSE of ΣΘΘ")
println("BiLQR: ", rmse_bilqr_avg, " ± ", rmse_bilqr_std/sqrt(50))
println("Random: ", rmse_random_avg, " ± ", rmse_random_std/sqrt(50))
println("Regression: ", rmse_regression_avg, " ± ", rmse_regression_std/sqrt(50))

# hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# Set default DPI and save the plot
default(dpi=1000)
savefig("time_mp.png")