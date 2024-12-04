using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ilqr_types.jl")
include("cartpole_sysid.jl")
include("../BiLQR/bilqr.jl")
include("../BiLQR/ekf.jl")
include("../Baselines/Regression.jl")

"""
    system_identification(pomdp::CartpoleSysIDPOMDP, policy, num_steps)

Simulates system identification for the Cartpole using the given policy.

# Arguments
- `pomdp`: The Cartpole system identification POMDP.
- `policy`: The policy to be used (e.g., BiLQR, MPC, etc.).
- `num_steps`: Number of simulation steps.

# Returns
- A tuple `(all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, mp_true)`.
"""

function step_through(pomdp::iLQRPOMDP, time_steps::Int, policy)
    b = initialstate_distribution(pomdp).support[1]
    s_true = mdp.s_init

    # Data storage

    ##TODO: fix for general iLQRPOMDP
    mp_estimates = [b[num_states(pomdp)]]
    mp_variances = [b[end - 1]]
    all_s = [s_true]
    all_b = [b]
    all_u = []

    # Simulation loop
    for t in 1:time_steps
        # Get action from policy
        a, action_info = POMDPs.action(policy, pomdp, b)

        # Store the action
        push!(all_u, a)

        # Step the POMDP
        b, _, _ = POMDPs.gen(pomdp, b, a, Random.default_rng())

        # Store belief and system parameters
        push!(all_b, b)
        push!(mp_estimates, b[num_states(pomdp)])
        push!(mp_variances, b[end - 1])

        # Simulate true state (process noise included)
        s_true = b[1:num_states(pomdp)]
        push!(all_s, s_true)
    end

    ΣΘΘ = b[end]
    info_dict = Dict(:all_b => all_b, :mp_estimates => mp_estimates, :mp_variances => mp_variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :mp_true => pomdp.mp_true)
    return all_b, info_dict
end