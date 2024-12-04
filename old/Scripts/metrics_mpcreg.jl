using Statistics 
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings
using Distributions

include("../XPlane/Cessna_SysID_miac.jl")

pomdp = XPlanePOMDP()

function log_multivariate_normal_pdf(x::Vector, μ::Vector, Σ::Matrix)
    # Create a Multivariate Normal distribution
    dist = MvNormal(μ, Σ)

    # Compute the log probability
    log_prob = logpdf(dist, x)

    return log_prob
end

function calculate_expected_reward(all_s, all_u, pomdp)
    rewards = []
    for seed in keys(all_s)
        reward = 0
        for t in 1:length(all_s[seed])
            u, w, θ, q = all_s[seed][t]
            a = all_u[seed][t]
            
            # +1 for theta between 0 and pi/6, +1 for w non negative, +1 for u[2] between 0 and 1
            reward += (θ >= 0 && θ <= pi/6) + (w >= 0) + (a[2] >= 0 && a[2] <= 1)
        end
        push!(rewards, reward)
    end
    mean_reward = mean(rewards)
    std_reward = std(rewards)
    return mean_reward, std_reward
end

# Define the number of time steps
t = 20
plotting_seed = 3
time_steps = collect(1:t)

@load "mpcreg_xplanefull_miac_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

trace_mpcreg_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpcreg_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_mpcreg_ste = trace_mpcreg_std/sqrt(length(all_ΣΘΘ))

last_log_probs_mpcreg = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end], all_AB_variances[seed][end]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]

# Calculate the average and standard deviation
# non_negative_indices = findall(x -> x >= 0, last_log_probs_mpcreg)
# filtered_log_probs_mpcreg = filter(x -> x >= 0, last_log_probs_mpcreg)
avg_last_log_prob_mpcreg = mean(last_log_probs_mpcreg)
std_last_log_prob_mpcreg = std(last_log_probs_mpcreg)

avg_reward_mpcreg, std_reward_mpcreg = calculate_expected_reward(all_s, all_u, pomdp)

# @load "regression_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

# trace_mpcreg_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
# trace_mpcreg_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
# trace_mpcreg_ste = trace_mpcreg_std/sqrt(length(all_ΣΘΘ))

# last_log_probs_mpcreg = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end], all_AB_variances[seed][end]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]

# # Calculate the average and standard deviation
# # non_negative_indices = findall(x -> x >= 0, last_log_probs_mpcreg)
# # filtered_log_probs_mpcreg = filter(x -> x >= 0, last_log_probs_mpcreg)
# avg_last_log_prob_mpcreg = mean(last_log_probs_mpcreg)
# std_last_log_prob_mpcreg = std(last_log_probs_mpcreg)

# avg_reward_mpcreg, std_reward_mpcreg = calculate_expected_reward(all_s, all_u, pomdp)

# # sort log probs 
# sorted_last_log_probs_mpcreg = sort(last_log_probs_mpcreg)
# println(sorted_last_log_probs_mpcreg)
# println(sortperm([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]))

println("Trace of ΣΘΘ")
println("mpcreg: ", trace_mpcreg_avg, " ± ", trace_mpcreg_ste)

println("Final Log Probability")
println("mpcreg: ", avg_last_log_prob_mpcreg, " ± ", std_last_log_prob_mpcreg)

println("Expected Reward")
println("mpcreg: ", avg_reward_mpcreg, " ± ", std_reward_mpcreg)

# hline!([all_ABtrue[plotting_seed ]], label="True mass", linestyle=:dash)

# # Set default DPI and save the plot
# default(dpi=1000)
# savefig("xplane_est.png")