using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings
using Distributions

function log_prob_gaussian(x, mean, variance)
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
# t = 20
# plotting_seed = 3
# time_steps = collect(1:t)

@load "bilqr_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)
ste_last_log_prob_bilqr = std_last_log_prob_bilqr/sqrt(length(last_log_probs_bilqr))

println("Trace of ΣΘΘ")
println("bilqr: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)

println("Final Log Probability")
println("bilqr: ", avg_last_log_prob_bilqr, " ± ", ste_last_log_prob_bilqr)