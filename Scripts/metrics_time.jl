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

@load "bilqr_cartpolefull_sysid_time_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

rmse_bilqr_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
rmse_bilqr_ste = rmse_bilqr_std/sqrt(length(all_ΣΘΘ))

plot(time_steps, all_mp_estimates[plotting_seed], ribbon=sqrt.(all_mp_variances[plotting_seed]), label="BILQR", xlabel="Time steps", ylabel=L"$m_p$ estimate",
    title="Estimated mass over time", fillalpha=0.2, color=:blue, legend=:topleft)

true_mass = all_mp_true[plotting_seed]
hline!([true_mass], label="True mass (1-20)", linestyle=:dash, xlims=(1, 20))
hline!([true_mass .+ 1], label="True mass + 1 (21-50)", linestyle=:dash, xlims=(21, 50))

# Set default DPI and save the plot
default(dpi=1000)
savefig("track_mp.png")