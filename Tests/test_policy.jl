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
    simulate(pomdp::iLQRPOMDP, policy, num_steps)

Simulates system identification for the Cartpole using the given policy.

# Arguments
- `pomdp`: The Cartpole system identification POMDP.
- `policy`: The policy to be used (e.g., BiLQR, MPC, etc.).
- `num_steps`: Number of simulation steps.

# Returns
- A tuple `(all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, mp_true)`.
"""

function simulate(pomdp::iLQRPOMDP, time_steps::Int, policy)
    b = initialstate_distribution(pomdp).support[1]
    s = mdp.s_init

    # Data storage
    vec_estimates = [b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]]
    variances = [diagm(b[end-num_sysvars(pomdp) + 1:end])]
    all_s = [s]
    all_b = [b]
    all_u = []

    # Simulation loop
    for t in 1:time_steps
        # Get action from policy
        a, action_info = action_info(policy, b)

        # Store the action
        push!(all_u, a)

        # Step the POMDP
        s, _, _ = POMDPs.gen(pomdp, s, a, Random.default_rng())

        # Observation 
        z = POMDPs.observation(pomdp, s, a, Random.default_rng())

        # Update belief
        b = ekf(pomdp, b, a, z)
        if b === nothing
            return nothing
        end
        
        # Extract the mean and covariance from belief
        m = b[1:num_states(pomdp)]
        Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))
        
        # Store estimates
        push!(vec_estimates, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]) # 8 x 1
        push!(variances, diagm(b[end-num_sysvars(pomdp) + 1:end])) # 8x8
        push!(all_b, b)

        # Simulate true state (process noise included)
        push!(all_s, s)
    end

    ΣΘΘ = b[end]
    info_dict = Dict(:all_b => all_b, :mp_estimates => mp_estimates, :mp_variances => mp_variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :mp_true => pomdp.mp_true)
    return all_b, info_dict
end