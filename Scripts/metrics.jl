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

# Define the number of time steps
t = 50
plotting_seed = 1
time_steps = collect(1:t)

@load "regression_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

# Extract theta (second element in the state vector) for the plotting seed
theta_regression = [all_s[plotting_seed][i][2] for i in 1:t]
plot(time_steps, theta_regression, label="Regression", title="MIAC", xlabel="Time Step", ylabel=L"\theta")

@load "random_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

theta_random = [all_s[plotting_seed][i][2] for i in 1:t]
plot!(time_steps, theta_random, label="EKF")

@load "bilqr_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

theta_bilqr = [all_s[plotting_seed][i][2] for i in 1:t]
plot!(time_steps, theta_bilqr, label="BiLQR")

@load "mpcreg_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

theta_mpcreg = [all_s[plotting_seed][i][2] for i in 1:t]
plot!(time_steps, theta_mpcreg, label="MPC + Regression")

@load "mpc_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

theta_mpc = [all_s[plotting_seed][i][2] for i in 1:t]
plot!(time_steps, theta_mpc, label="MPC + EKF")


# Set default DPI and save the plot
default(dpi=1000)
savefig("theta_trajectory.png")