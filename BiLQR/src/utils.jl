# saving 

# loading 

# plotting 

# JLD2??

# evaluations 

# reduce lines in experiments.jl

using POMDPs
using POMDPTools
"""
    create_sample_cartpole()

Create a sample CartpoleMDP instance with predefined parameters.

# Returns
- `CartpoleMDP`: An instance of the CartpoleMDP with predefined matrices and parameters.

# Example
```julia
pomdp = create_sample_cartpole()
```
"""

function create_sample_cartpole()
    # Define the matrices and parameters
    Q = 1e-4 * I(5)
    R = 1e-4 * I(1)
    Q_N = Diagonal([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    Λ = Diagonal(vcat(fill(1e-4, 24), [1]))  # 5^2

    m0 = [0.0, π / 2, 0.0, 0.0, 2.0]         # Initial mean of the belief
    Σ0 = [1e-4, 1e-4, 1e-4, 1e-4, 2.0]  # Initial covariance matrix

    δt = 0.1        # Time step
    mc = 1.0        # Cart mass
    g = 9.81        # Gravitational acceleration
    l = 1.0         # Pole length

    # Noise covariance matrices
    # W_state_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    W_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    W_obs = diagm([1e-2, 1e-2, 1e-2, 1e-2])
    # W_obs_ekf = diagm([1e-2, 1e-2, 1e-2, 1e-2])

    # Number of states, actions, observations, and system variables
    num_states = 5
    num_actions = 1
    num_observations = 4
    num_sysvars = 1

    cartpole_mdp = CartpoleMDP(Q, R, Q_N, Λ, m0, Σ0, δt, mc, g, l, W_process, W_obs, 
                                num_states, num_actions, num_observations, num_sysvars)
    return cartpole_mdp
end 


"""
    simulate(pomdp::iLQRPOMDP, num_steps, policy)

Simulates system identification for the Cartpole using the given policy.

# Arguments
- `pomdp`: The Cartpole system identification POMDP.
- `num_steps`: Number of simulation steps.
- `policy`: The policy to be used (e.g., BiLQR, MPC, etc.).

# Returns
- A tuple `(all_b, mp_estimates, mp_variances, ΣΘΘ, all_s, all_u, mp_true)`.
"""

function simulate(time_steps::Int, policy, belief_updater)
    pomdp = policy.pomdp

    # remember Σ0 now is a vector of diagonal elements of covariance matrix (not vector of all elements in covariance matrix)
    s = pomdp.s_init
    # b = vcat(s[1:end - num_sysvars(pomdp)], pomdp.mp_true, pomdp.Σ0[:])

    b = vcat(s[1:end - pomdp.num_sysvars], pomdp.s_init[pomdp.num_states - pomdp.num_sysvars + 1:end], diagm(pomdp.Σ0)[:])

    # Data storage
    vec_estimates = [b[pomdp.num_states - pomdp.num_sysvars + 1:pomdp.num_states]]
    variances = [diagm(b[end-pomdp.num_sysvars + 1:end])]
    all_s = [s]
    all_b = [b]
    all_u = []
    means = []
    variances = []

    # Simulation loop
    # stepthrough(pomdp, policy; up=updater, b0=initial_belief, s0=initial_state, max_steps=10, kwargs...)
    # for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=10)
    #     println("in state $s")
    #     println("took action $o")
    #     println("received observation $o and reward $r")
    # end
    i = 0

    # for (current_s, a, z, r) in stepthrough(pomdp, policy, belief_updater, b, s, "s,a,o,r", max_steps=time_steps)
    for t in 1:time_steps

        println("timestep: ", t)
        
        push!(all_b, b)
        push!(all_s, s)

        a, action_dict = action_info(policy, b)
        push!(all_u, a)

        s = rand(POMDPs.transition(pomdp, s, a))

        # Generate observation from the true next state
        z = rand(POMDPs.observation(pomdp, s, a))

        # Update the belief using the updater
        b = update(belief_updater, b, a, z)
        if b === nothing
            println("Belief update failed; terminating simulation.")
            return nothing
        end

        # Extract mean and covariance from belief
        m = b[1:num_states(pomdp)]
        Σ = reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))

        # Store estimates
        push!(means, b[num_states(pomdp) - num_sysvars(pomdp) + 1:num_states(pomdp)])
        push!(variances, diagm(b[end - num_sysvars(pomdp) + 1:end]))

        # i += 1
    end

    ΣΘΘ = b[end]

    ## TODO: not mp_true
    info_dict = Dict(:all_b => all_b, :means => means, :variances => variances, :ΣΘΘ => ΣΘΘ, :all_s => all_s, :all_u => all_u, :true_params => pomdp.mp_true)
    return all_b, info_dict

end
