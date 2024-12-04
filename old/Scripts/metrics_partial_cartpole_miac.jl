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
plotting_seed = 20
time_steps = collect(1:t)

function calculate_expected_reward(all_s, all_u; α=1.0, β=1.0, γ=1.0, δ=0.1, ϵ=0.01)

    rewards = []
    for seed in keys(all_s)
        s_seed = all_s[seed]
        reward = 0
        for t in 1:length(all_u[seed])
            x, θ, x_dot, θ_dot = s_seed[t]
            u = all_u[seed][t]
            
            # +1 reward for balancing pole
            if abs(pi/2 - θ) < pi/180*12
                reward += 1 - u[1]/4
            end
        end
        push!(rewards, reward)
    end

    mean_reward = mean(rewards)
    std_reward = std(rewards)
    return mean_reward, std_reward
end

# Random + EKF Algorithm
@load "random_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_random_ste = trace_random_std / sqrt(length(all_ΣΘΘ))

log_probs_random = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot(time_steps, log_probs_random, label="EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_random = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]
# Calculate the average and standard deviation
avg_last_log_prob_random = mean(last_log_probs_random)
std_last_log_prob_random = std(last_log_probs_random)

# Calculate expected reward
mean_counts_random, std_counts_random = calculate_expected_reward(all_s, all_u)

# plot mass estimates with standard deviation ribbon
# plot(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="EKF", xlabel="Time Step", ylabel="Mass Estimate")

# BiLQR Algorithm
@load "bilqr_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_bilqr_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_ste = trace_bilqr_std / sqrt(length(all_ΣΘΘ))

log_probs_bilqr = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_bilqr, label="BiLQR")

last_log_probs_bilqr = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]
# 39, 22, 35
# Calculate the average and standard deviation
avg_last_log_prob_bilqr = mean(last_log_probs_bilqr)
std_last_log_prob_bilqr = std(last_log_probs_bilqr)

# Calculate expected reward
mean_counts_bilqr, std_counts_bilqr = calculate_expected_reward(all_s, all_u)

# plot!(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="BiLQR")

# MPC + EKF Algorithm
@load "mpc_cartpolepartial_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s all_u all_mp_true

trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpc_ste = trace_mpc_std / sqrt(length(all_ΣΘΘ))

log_probs_mpc = [log_prob_gaussian(all_mp_true[plotting_seed], all_mp_estimates[plotting_seed][i], all_mp_variances[plotting_seed][i]) for i in 1:t]
# plot!(time_steps, log_probs_mpc, label="MPC + EKF", xlabel="Time Step", ylabel=L"\log(p(\hat{\theta} \mid a_{1:t}, o_{1:t}))")

last_log_probs_mpc = [log_prob_gaussian(all_mp_true[seed], all_mp_estimates[seed][end], all_mp_variances[seed][end]) for seed in 1:length(all_mp_estimates) if haskey(all_mp_estimates, seed)]

# Calculate the average and standard deviation
avg_last_log_prob_mpc = mean(last_log_probs_mpc)
std_last_log_prob_mpc = std(last_log_probs_mpc)

# Calculate expected reward
mean_counts_mpc, std_counts_mpc = calculate_expected_reward(all_s, all_u)

# plot!(time_steps, [(all_mp_estimates[plotting_seed][i]) for i in 1:t], ribbon=[(all_mp_variances[plotting_seed][i]) for i in 1:t], label="MPC")


println("Trace of ΣΘΘ")
println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_ste)
println("Random + EKF: ", trace_random_avg, " ± ", trace_random_ste)
println("MPC + EKF: ", trace_mpc_avg, " ± ", trace_mpc_ste)

println("\nLog Probabilities")
println("Random: ", avg_last_log_prob_random, " ± ", std_last_log_prob_random)
println("BiLQR: ", avg_last_log_prob_bilqr, " ± ", std_last_log_prob_bilqr)
println("MPC + EKF: ", avg_last_log_prob_mpc, " ± ", std_last_log_prob_mpc)

println("\nExpected Reward (number of time steps within ±12 degrees of the top position)")
println("Random + EKF: ", mean_counts_random, " ± ", std_counts_random)
println("BiLQR: ", mean_counts_bilqr, " ± ", std_counts_bilqr)
println("MPC + EKF: ", mean_counts_mpc, " ± ", std_counts_mpc)

# hline!([all_mp_true[plotting_seed]], label="True mass", linestyle=:dash)

# # Set default DPI and save the plot
# default(dpi=1000)
# savefig("miacpartial_mp_est.png")