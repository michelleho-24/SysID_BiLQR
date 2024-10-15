using Statistics
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings

Σ_true = 2.0

function log_prob_gaussian(x, mean, variance)
    return -0.5 * log(2 * π * variance) - 0.5 * ((x - mean)^2 / variance)
end

# Define the number of time steps
t = 50
plotting_seed = 6
time_steps = collect(1:t)

# Helper function to calculate the expected reward
function calculate_expected_reward(all_s)
    counts = []

    for seed in keys(all_s)
        s_seed = all_s[seed]
        count = 0
        for t in 1:length(s_seed)
            angle = s_seed[t][2]  # Assuming angle is the second element
            angle_diff = angle - (π / 2)
            angle_diff_deg = angle_diff * (180 / π)
            if abs(angle_diff_deg) <= 45
                count += 1
            end
        end
        push!(counts, count)
    end

    mean_count = mean(counts)
    std_count = std(counts)
    return mean_count, std_count
end

# Random + EKF Algorithm
@load "random_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std / sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_random = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

# Calculate expected reward
mean_counts_random, std_counts_random = calculate_expected_reward(all_s)

# BiLQR Algorithm
@load "bilqr_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std / sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_bilqr, label="BiLQR")

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)

# Calculate expected reward
mean_counts_bilqr, std_counts_bilqr = calculate_expected_reward(all_s)

# MPC + Regression Algorithm
@load "mpcreg_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

# Calculate trace statistics
trace_values = [ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]
trace_mpcreg_avg = mean(trace_values)
trace_mpcreg_std = std(trace_values)
trace_mpcreg_ste = trace_mpcreg_std / sqrt(length(trace_values))

log_probs_mpcreg = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_mpcreg, label="MPC + Regression")

last_log_probs_mpcreg = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_mpcreg = mean(last_log_probs_mpcreg)
std_last_log_prob_mpcreg = std(last_log_probs_mpcreg)

# Calculate expected reward
mean_counts_mpcreg, std_counts_mpcreg = calculate_expected_reward(all_s)

# MPC + EKF Algorithm
@load "mpc_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_ste = trace_mpc_std / sqrt(length(all_ΣΘΘ))

log_probs_mpc = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
plot!(time_steps, log_probs_mpc, label="MPC + EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_mpc = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_mpc = mean(last_log_probs_mpc)
std_last_log_prob_mpc = std(last_log_probs_mpc)

# Calculate expected reward
mean_counts_mpc, std_counts_mpc = calculate_expected_reward(all_s)

println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random + EKF: ", trace_random_avg, " ± ", trace_random_ste)
println("MPC + Regression: ", trace_mpcreg_avg, " ± ", trace_mpcreg_ste)
println("MPC + EKF: ", trace_mpc_avg, " ± ", trace_mpc_ste)

println("\nLog Probabilities")
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)
println("BiLQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("MPC + Regression: ", avg_last_log_prob_mpcreg, " ± ", std_last_log_prob_mpcreg)
println("MPC + EKF: ", avg_last_log_prob_mpc, " ± ", std_last_log_prob_mpc)

println("\nExpected Reward (number of time steps within ±12 degrees of the top position)")
println("Random + EKF: ", mean_counts_random, " ± ", std_counts_random)
println("BiLQR: ", mean_counts_bilqr, " ± ", std_counts_bilqr)
println("MPC + Regression: ", mean_counts_mpcreg, " ± ", std_counts_mpcreg)
println("MPC + EKF: ", mean_counts_mpc, " ± ", std_counts_mpc)

# hline!([all_mp_true[plotting_seed]], label="True mass", linestyle=:dash)

# # Set default DPI and save the plot
# default(dpi=1000)
# savefig("time_mp.png")
