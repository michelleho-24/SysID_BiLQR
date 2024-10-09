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

# @load "mpcreg_cartpolefull_miac_results.jld2" all_mp_estimates all_mp_variances all_mp_true 

# log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
# plot(time_steps, log_probs_regression, label="Regression", title = "System Identification")

@load "mpc_cartpolefull_miac_results.jld2" all_mp_estimates all_mp_variances all_mp_true

log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot(time_steps, log_probs_regression, label="MPC", title = "Adaptive Control")

@load "random_cartpolefull_sysid_results.jld2" all_mp_estimates all_mp_variances all_mp_true

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

@load "bilqr_cartpolefull_miac_results.jld2" all_mp_estimates all_mp_variances all_mp_true

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
plot!(time_steps, log_probs_bilqr, label="BiLQR")

# hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# Set default DPI and save the plot
default(dpi=1000)
savefig("time_mp_miac.png")