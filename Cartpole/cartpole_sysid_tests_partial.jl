# using Plots
using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ilqr_types.jl")
include("cartpole_sysid_partial.jl")
include("../BiLQR/bilqr.jl")
include("../BiLQR/ekf.jl")
include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
# include("../Baselines/Regression.jl")

global b, s_true

function system_identification(seed)
    
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = CartpoleMDP()

    # True mass of the pole (unknown to the estimator)
    mp_true = 2.0  # True mass of the pole

    # Initial true state
    s_true = pomdp.s_init  # [x, θ, dx, dθ, mp]

    # Initial belief state
    Σ0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.1])  # Initial covariance
    b = vcat(pomdp.s_init, Σ0[:])  # Belief vector containing mean and covariance

    # Simulation parameters
    num_steps = 100

    # Data storage for plotting
    mp_estimates = zeros(num_steps)
    mp_variances = zeros(num_steps)
    all_s = []


    for t in 1:num_steps

        # a = mpc(pomdp, b,10)
        # a = [rand() * 20.0 - 10.0]
        a, info_dict = bilqr(pomdp, b)
        
        # Simulate the true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise to the true state
        noise_state = rand(MvNormal(pomdp.W_state_process))
        noise_total = vcat(noise_state, 0.0)
        s_next_true = s_next_true + noise_total
        
        # Generate observation from the true next state
        z = obs_mean(pomdp, s_next_true)
        
        # Add observation noise
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise
        
        # Use your ekf function to update the belief
        b = ekf(pomdp, b, a, z)
        
        # Extract the mean and covariance from belief
        m = b[1:num_states(pomdp)]
        Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))
        
        # Store estimates
        mp_estimates[t] = m[num_states(pomdp)]
        mp_variances[t] = Σ[num_states(pomdp), num_states(pomdp)]
        
        # Update the true state for the next iteration
        s_true = s_next_true

        # Store the true state for plotting
        push!(all_s, s_true)
    end

    ΣΘΘ = b[end]
    
    return b, mp_estimates, mp_variances, ΣΘΘ, all_s

end