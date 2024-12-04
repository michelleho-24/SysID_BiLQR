using POMDPs

"""
    struct RandomPolicy

A policy that generates random actions within specified ranges.

# Fields
- `action_ranges::Vector{Tuple{Float64, Float64}}`: A vector of tuples where each tuple specifies the range (min, max) for each action dimension.
"""
struct RandomPolicy
    action_ranges::Vector{Tuple{Float64, Float64}}
end

"""
    POMDPs.action(policy::RandomPolicy, b)

Generate a random action based on the specified ranges in the `RandomPolicy`.

# Arguments
- `policy::RandomPolicy`: The policy object specifying action ranges.
- `b`: The belief or state (not used in this random policy, but required by the POMDPs.jl interface).

# Returns
- A tuple `(action, action_info)` where:
  - `action`: The randomly generated action vector.
  - `action_info`: A dictionary containing additional information about the action generation.
"""
function POMDPs.action(policy::RandomPolicy, pomdp::iLQRPOMDP, b)
    # Extract action ranges
    action_ranges = policy.action_ranges

    # Generate a random action for each dimension
    action = [rand(range[1]:range[2]) for range in action_ranges]

    # Compile action_info
    action_info = Dict(
        :action_ranges => action_ranges,  # The ranges used to generate the actions
        :num_actions => length(action_ranges),  # Number of action dimensions
        :random_seed => Random.default_rng().state.seed,  # RNG seed for reproducibility
        :distribution => "Uniform",  # Indicates the type of random distribution
        :action => action  # The generated action (useful for logging)
    )

    return action, action_info
end

# solver = QMDPSolver()
# policy = solve(solver, pomdp)
