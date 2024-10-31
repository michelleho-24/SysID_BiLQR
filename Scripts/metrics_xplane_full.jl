using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings
using Distributions

function log_multivariate_normal_pdf(x::Vector, μ::Vector, Σ::Matrix)
    # Create a Multivariate Normal distribution
    dist = MvNormal(μ, Σ)

    # Compute the log probability
    log_prob = logpdf(dist, x)

    return log_prob
end

# function log_prob_gaussian(x, mean, variance)
#     variance = variance[1][1]
#     mean = mean[1]
#     x = x[1]
#     return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
# end

# Define the number of time steps
t = 50
plotting_seed = 3
time_steps = collect(1:t)

@load "bilqr_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

trace_bilqr_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std/sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_multivariate_normal_pdf(all_ABtrue[plotting_seed ], all_AB_estimates[plotting_seed ][i], all_AB_variances[plotting_seed][i]) for i in 1:t]
plot(time_steps, log_probs_bilqr, label="BiLQR")

last_log_probs_bilqr = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end-35], all_AB_variances[seed][end-35]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]

# Calculate the average and standard deviation
# non_negative_indices = findall(x -> x >= 0, last_log_probs_bilqr)
# filtered_log_probs_bilqr = filter(x -> x >= 0, last_log_probs_bilqr)
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)


@load "random_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

# println(size(all_AB_variances[1][1]))
# println(size(all_AB_estimates[1][1]))
# println(size(all_ABtrue[1]))

trace_random_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std/sqrt(length(all_ΣΘΘ))

log_probs_random = [log_multivariate_normal_pdf(all_ABtrue[plotting_seed ], all_AB_estimates[plotting_seed ][i], all_AB_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_random = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end-35], all_AB_variances[seed][end-35]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]
# filtered_last_log_probs_random = last_log_probs_random[non_negative_indices]
# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

# # sort log probs 
# sorted_last_log_probs_bilqr = sort(last_log_probs_bilqr)
# println(sorted_last_log_probs_bilqr)
# println(sortperm([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]))

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random: ", trace_random_avg, " ± ", trace_random_ste)

println("Final Log Probability")
println("BILQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)

# hline!([all_ABtrue[plotting_seed ]], label="True mass", linestyle=:dash)

# # Set default DPI and save the plot
default(dpi=1000)
savefig("xplane_est.png")