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
include("../Baselines/Regression.jl")

global b, s_true

function system_identification(seed)
    
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = CartpoleMDP()

    # True mass of the pole (unknown to the estimator)
    mp_true = 2.0  # True mass of the pole

    # so if b_initial has mu_0 = [mu_x, mu_theta] and Sigma_0 = [Sxx, 0; 0, S_thth], 
    # in both cases you draw s_0 from b_initial, in partially observable, b_0=b_initial, 
    # in fully observable b_0 has mean [s_0_x, mu_theta] and covariance [.001I, 0; 0, S_thth]

    # Initial true state
    # s_true = pomdp.s_init  # [x, θ, dx, dθ, mp]
    # s_true = [0.0, π/2, 0.0, 0.0, 2.0]

    # always start s0 from upright position, but draw mp from a normal distribution
    # mp_prior = Normal(2.0, 1.0)
    # mp = abs(rand(mp_prior))
    # println("True mass: ", mp)
    # s0 = [0.0, π/2, 0.0, 0.0, 2.0]
    # s_true = s0

    # # Initial belief state
    # # Σ0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.1])  # Initial covariance normally
    # Σ0 = Diagonal([1e-4, 1e-4, 1e-4, 1e-4, 1.0])  # Initial covariance
    # b = vcat(pomdp.s_init[1:4], mp, Σ0[:])  # Belief vector containing mean and covariance

    # define initial distribution for total belief state 
    s_true = pomdp.s_init
    
    b = vcat(s_true[1:4], pomdp.mp_true, pomdp.Σ0[:])
    # Simulation parameters
    num_steps = 100

    # Data storage for plotting
    mp_estimates = zeros(num_steps)
    mp_variances = zeros(num_steps)
    all_s = []
    all_b = []
    all_u = []

    for t in 1:num_steps

        # Store the true state for plotting
        push!(all_s, s_true)
        push!(all_b, b)

        # a, info_dict = bilqr(pomdp, b)
        # a = mpc(pomdp, b, 10)
        a = random_policy(pomdp, b)
        # all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true = regression(pomdp, b)
        # return all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true

        push!(all_u, a)
        
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
    
    return all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, pomdp.mp_true

end