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
    return -0.5 * log(2 * π * variance[1]) - 0.5 * ((x - mean[1])^2 / variance[1])
end

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

method = "random"

@load "$(method)_xplanefull_sysid_results.jld2" all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 

last_log_probs = [log_multivariate_normal_pdf(all_ABtrue[seed], all_AB_estimates[seed][end], all_AB_variances[seed][end]) for seed in 1:length(all_AB_estimates)  if haskey(all_AB_estimates, seed)]

trace_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_ste = trace_std/sqrt(length(last_log_probs))

avg_last_log_prob = mean(last_log_probs)
std_last_log_prob = std(last_log_probs)
ste_last_log_prob = std_last_log_prob/sqrt(length(last_log_probs))

println("Trace of ΣΘΘ")
println("$(method): ", trace_avg, " ± ", trace_ste)

println("Final Log Probability")
println("$(method): ", avg_last_log_prob, " ± ", ste_last_log_prob)