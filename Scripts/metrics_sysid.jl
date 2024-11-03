using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings
using Distributions

function log_prob_gaussian(x, mean, variance)
    if variance[1] <= 1e-10
        return 0  # Return negative infinity for invalid variance
    end
    
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
# t = 20
# plotting_seed = 3
# time_steps = collect(1:t)

method = "bilqr"

@load "$(method)_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

last_log_probs = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

filtered_log_probs = [last_log_probs[i] for i in 1:length(last_log_probs) if i != 107]
# println(sort(filtered_log_probs))

trace_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_ste = trace_std/sqrt(length(all_ΣΘΘ))

avg_last_log_prob = mean(filtered_log_probs)
std_last_log_prob = std(filtered_log_probs)
ste_last_log_prob = std_last_log_prob/sqrt(length(last_log_probs))

println("Trace of ΣΘΘ")
println("$(method): ", trace_avg, " ± ", trace_ste)

println("Final Log Probability")
println("$(method): ", avg_last_log_prob, " ± ", ste_last_log_prob)