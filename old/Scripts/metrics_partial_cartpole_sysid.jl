using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings

function log_prob_gaussian(x, mean, variance)
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
t = 50
plotting_seed = 1
time_steps = collect(1:t)

@load "random_cartpolepartial_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std/sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")
plot(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="EKF", xlabel="Time Step", ylabel="Mass Estimate")

last_log_probs_random = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

@load "bilqr_cartpolepartial_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[1][i]) for i in 1:t]
# plot!(time_steps, log_probs_bilqr, label="BiLQR")
plot!(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="BiLQR", xlabel="Time Step", ylabel="Mass Estimate")

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random: ", trace_random_avg, " ± ", trace_random_ste)

println("Average End Log Probabilities")
println("BiLQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)

hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# Set default DPI and save the plot
default(dpi=1000)
savefig("est_partial_mp_seed1.png")