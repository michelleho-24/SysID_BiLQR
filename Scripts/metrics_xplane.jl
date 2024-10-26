using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings

function log_prob_gaussian(x::AbstractVector, mean::AbstractVector, covariance::AbstractMatrix)
    k = length(x)
    diff = x - mean
    log_det_cov = logdet(covariance)
    inv_cov = inv(covariance)
    exponent = diff' * inv_cov * diff
    return -0.5 * (k * log(2 * π) + log_det_cov + exponent)
end

# Define the number of time steps
t = 50
plotting_seed = 1
time_steps = collect(1:t)

@load "regression_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

# Calculate trace statistics
trace_values = [ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]
trace_regression_avg = mean(trace_values)
trace_regression_std = std(trace_values)
trace_regression_ste = trace_regression_std / sqrt(length(trace_values))

log_probs_regression = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot(time_steps, log_probs_regression, label="Linear Regression", title = "System Identification")

last_log_probs_regression = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_regression= mean(last_log_probs_regression)
std_last_log_prob_regression = std(last_log_probs_regression)

@load "random_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std/sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_random = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

@load "bilqr_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed ], all_mp_estimates[plotting_seed ][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_bilqr, label="BiLQR")

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates)  if haskey(all_mp_estimates, seed)]

Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random: ", trace_random_avg, " ± ", trace_random_ste)
println("Regression: ", trace_regression_avg, " ± ", trace_regression_ste)

println("Final Log Probability")
println("BILQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)
println("Regression: ", avg_last_log_prob_regression, " ± ", std_last_log_prob_regression)

# hline!([all_mp_true[plotting_seed ]], label="True mass", linestyle=:dash)

# # Set default DPI and save the plot
# default(dpi=1000)
# savefig("time_mp.png")