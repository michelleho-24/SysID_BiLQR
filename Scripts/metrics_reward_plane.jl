using Statistics
using JLD2
using LinearAlgebra
using Plots; pgfplotsx()
using LaTeXStrings
using Distributions


function calculate_cumulative_reward_for_seed(all_s, all_u, seed)
    rewards = []
    for t in 1:length(all_s[seed])
        u, w, θ, q = all_s[seed][t]
        a = all_u[seed][t]
        r = (θ >= 0 && θ <= pi/6) + (w >= 0) + (a[2] >= 0 && a[2] <= 1)
        push!(rewards, r)
    end
    return cumsum(rewards)
end

# === Choose a consistent seed to compare ===
seed = 2  # Change this to any shared seed index

# === Load all three files ===
files = [
    ("BiLQR_xplanepartial_miac_results.jld2", "BiLQR"),
    ("Random+EKF_xplanepartial_miac_results.jld2", "Random+EKF"),
    ("MPC+EKF_xplanepartial_miac_results.jld2", "MPC+EKF")]

cumulative_curves = []

for (filename, label) in files
    @load filename all_s all_u
    if haskey(all_s, seed) && haskey(all_u, seed)
        cumulative_r = calculate_cumulative_reward_for_seed(all_s, all_u, seed)
        push!(cumulative_curves, (cumulative_r, label))
    else
        @warn "Seed $seed not found in $filename"
    end
end

# === Plotting ===

plt = plot(; legend=:topleft, xlabel="Time Step", ylabel="Accumulated Reward", 
            titlefontsize=14, guidefontsize=12, legendfontsize=12, lw=2)

for (cumulative_r, label) in cumulative_curves
    T = 1:length(cumulative_r)
    plot!(plt, T, cumulative_r, label=label)
end

savefig("cumulative_reward_seed$(seed).tex")
