# using Plots
using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../BiLQR/ilqr_types.jl")
include("Cessna_SysID_partial.jl")
include("../BiLQR/bilqr_xplane.jl")
include("../BiLQR/ekf_xplane.jl")
# include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
# include("../Baselines/Regression.jl")

global b, s_true

function system_identification(seed, method)
    
    Random.seed!(seed)

    # Initialize the Cartpole MDP
    pomdp = XPlanePOMDP()

    # define initial distribution for total belief state 
    s_true = pomdp.s_init
    
    # 24 + 24 vector 
    b = vcat(s_true[1:end - num_sysvars(pomdp)], pomdp.s_init[num_states(pomdp) - num_sysvars(pomdp) + 1:end], pomdp.Σ0)
    # Simulation parameters
    num_steps = 50

    # Data storage for plotting
    AB_vec_estimates = []
    AB_variances = []
    
    push!(AB_vec_estimates, b[num_states(pomdp) - num_sysvars(pomdp):num_states(pomdp)]) # 16 x 1
    push!(AB_variances, Diagonal(b[end-(num_sysvars(pomdp) + 1):end])) # 16 x 16

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

        #TODO: might want to change random range for this problem 
        elseif method == "random"
            a = xplane_random_policy(pomdp, b)
        elseif method == "regression" || method == "mpcreg"
            all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true = regression(pomdp, b, method)
            return all_b, mp_estimated_list, variance_mp_list, ΣΘΘ, all_s, all_u, pomdp.mp_true
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
        
        push!(AB_vec_estimates, b[num_states(pomdp) - num_sysvars(pomdp):num_states(pomdp)]) # 16 x 1
        push!(AB_variances, diagm(b[end-(num_sysvars(pomdp) + 1):end])) # 16 x 16
        
        # Update the true state for the next iteration
        s_true = s_next_true

        # Store the true state for plotting
        push!(all_s, s_true)
    end

    ΣΘΘ = diagm(b[end-(num_sysvars(pomdp)-1):end])
    
    return all_b, AB_vec_estimates, AB_variances, ΣΘΘ, all_s, all_u, pomdp.AB_true

end