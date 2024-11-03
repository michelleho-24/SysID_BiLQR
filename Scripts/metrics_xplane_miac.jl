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

function calculate_expected_reward(all_s, all_u)
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
    # Create the histogram
    p = histogram(rewards, bins=6, title="Reward Distribution", xlabel="Reward Value", ylabel="Frequency")

    # Save the plot as a PNG file
    savefig(p, "bilqr_reward_histogram.png")

    mean_reward = mean(rewards)
    std_reward = std(rewards)
    return mean_reward, std_reward
end

method = "bilqr"

@load "$(method)_xplanefull_miac_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

last_log_probs = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end], all_AB_variances[seed][end]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]
trace_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_ste = trace_std/sqrt(length(last_log_probs))

avg_last_log_prob = mean(last_log_probs)
std_last_log_prob = std(last_log_probs)
ste_last_log_prob = std_last_log_prob/sqrt(length(last_log_probs))

mean_reward, std_reward = calculate_expected_reward(all_s, all_u)

println("Trace of ΣΘΘ")
println("$(method): ", trace_avg, " ± ", trace_ste)

println("Final Log Probability")
println("$(method): ", avg_last_log_prob, " ± ", ste_last_log_prob)

println("Expected Reward")
println("$(method): ", mean_reward, " ± ", std_reward)

