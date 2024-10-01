# using Plots
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
include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
# include("../Baselines/Regression.jl")

global b, s_true

function system_identification(method)

    # Initialize the Cartpole MDP
    pomdp = CartpoleMDP()

    # True mass of the pole (unknown to the estimator)
    mp_true = 2.0  # True mass of the pole

    # huge prior on the mass to begin with, let seed select mass from the distribution

    # Initial true state sampled from initial belief 
    # b0 = pomdp.b0
    s_true = rand(MvNormal(pomdp.μ0, pomdp.Σ0))
    # s_true = pomdp.s_init  # [x, θ, dx, dθ, mp]

    # Initial belief state
    # Σ0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.1])  # Initial covariance normally
    
    # Initial covariance
    Σ0 = Diagonal(vcat(fill(1e-6, num_states(pomdp) - num_sysvars(pomdp)), [pomdp.Σ0[end]]))

    # Initialize belief as true state vector, mean of the log of pole mass, and new Σ0
    b = vec(vcat(s_true[1:num_states(pomdp)-num_sysvars(pomdp)], pomdp.μ0[end], Σ0[:]))

    # pomdp.s_goal = 

    # Simulation parameters
    num_steps = 100

    # Data storage for plotting
    log_mp_estimates = zeros(num_steps)
    log_mp_variances = zeros(num_steps)
    all_s = []
    all_b = []

    for t in 1:num_steps

        if method == "bilqr"
            # Use your bilqr function to get the optimal action
            a, info_dict = bilqr(pomdp, b)
        elseif method == "mpc"
            a = mpc(pomdp, b, 10)
        elseif method == "random"
            a = [rand() * 20.0 - 10.0]
        end
        
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
        log_mp_estimates[t] = m[num_states(pomdp)]
        log_mp_variances[t] = Σ[num_states(pomdp), num_states(pomdp)]
        
        # Update the true state for the next iteration
        s_true = s_next_true

        # Store the true state for plotting
        push!(all_s, s_true)
        push!(all_b, b)
        # println(b)
    end

    ΣΘΘ = b[end]
    
    return all_b, log_mp_estimates, log_mp_variances, ΣΘΘ, all_s

end