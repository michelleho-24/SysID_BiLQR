# using Plots
using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ilqr_types.jl")
include("Cessna_SysID.jl")
include("../BiLQR/bilqr_xplane.jl")
include("../BiLQR/ekf_xplane.jl")
# include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
include("../Baselines/regression_xplane.jl")

global b, s_true

function system_identification(seed, method)
    
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = XPlanePOMDP()

    # define initial distribution for total belief state 
    s_true = pomdp.s_init
    
    # 12 + 12 vector 
    b = vcat(s_true[1:end - num_sysvars(pomdp)], pomdp.s_init[num_states(pomdp) - num_sysvars(pomdp) + 1:end], pomdp.Σ0)
    # println(size(b))
    # println(size(pomdp.Σ0))
    # Simulation parameters
    num_steps = 15

    # Data storage for plotting
    AB_vec_estimates = []
    AB_variances = []
    
    push!(AB_vec_estimates, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]) # 8 x 1
    push!(AB_variances, diagm(b[end-num_sysvars(pomdp) + 1:end])) # 8x8

    all_s = []
    all_b = []
    all_u = []

    for t in 1:num_steps

        # Store the true state for plotting
        push!(all_s, s_true)
        push!(all_b, b)

        if method == "bilqr"
            results = bilqr(pomdp, b)
            if results === nothing
                return nothing
            else 
                a, info_dict = results
            end
        elseif method == "mpc"
            a = mpc(pomdp, b, 10)

        elseif method == "random"
            a = xplane_random_policy(pomdp, b)
        elseif method == "regression" || method == "mpcreg"
            results = regression(pomdp, b, method)
            return results 
        end 

        push!(all_u, a)
        
        # Simulate the true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        
        # Add process noise to the true state
        noise_state = rand(MvNormal(pomdp.W_state_process))
        noise_total = vcat(noise_state, zeros(num_sysvars(pomdp)))
        s_next_true = s_next_true + noise_total
        
        # Generate observation from the true next state
        z = obs_mean(pomdp, s_next_true)
        
        # Add observation noise
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise
        
        # Use your ekf function to update the belief
        b = ekf(pomdp, b, a, z)

        if b === nothing
            return nothing
        end
        
        push!(AB_vec_estimates, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)]) # 8 x 1
        push!(AB_variances, diagm(b[end-num_sysvars(pomdp) + 1:end])) 

        # println(size(b[5:12]))
        # println(size(diagm(b[13:end])))
        
        # Update the true state for the next iteration
        s_true = s_next_true
    end

    ΣΘΘ = diagm(b[end-num_sysvars(pomdp) + 1:end])
    # println(typeof(all_b))
    # println(typeof(AB_vec_estimates))
    # println(typeof(AB_variances))
    # println(typeof(ΣΘΘ))
    # println(typeof(all_s))
    # println(typeof(all_u))
    # println(typeof(pomdp.AB_true))
    
    return all_b, AB_vec_estimates, AB_variances, ΣΘΘ, all_s, all_u, pomdp.AB_true

end