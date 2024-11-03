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
                reward += 1 - u[1]/10
            end
        end
        push!(rewards, reward)
    end

    # # Create the histogram
    # p = histogram(rewards, bins=6, title="Reward Distribution", xlabel="Reward Value", ylabel="Frequency")

    # # Save the plot as a PNG file
    # savefig(p, "random_reward_histogram.png")

    mean_reward = mean(rewards)
    std_reward = std(rewards)
    return mean_reward, std_reward
end

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

mean_reward, std_reward = calculate_expected_reward(all_s, all_u)

println("Trace of ΣΘΘ")
println("$(method): ", trace_avg, " ± ", trace_ste)

println("Final Log Probability")
println("$(method): ", avg_last_log_prob, " ± ", ste_last_log_prob)

println("Expected Reward")
println("$(method): ", mean_reward, " ± ", std_reward)