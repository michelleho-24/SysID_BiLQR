using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
using Optim

function sigma_pred(pomdp, x::AbstractVector, u::AbstractVector, Σ::AbstractMatrix)
    At = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, u), x)
    return At * Σ * At' + dyn_noise(pomdp, x,u)
end

function sigma_update(pomdp, x::AbstractVector, Σ::AbstractMatrix)
    Ct = ForwardDiff.jacobian(x -> obs_mean(pomdp,x), x)
    K = Σ * Ct' * inv(Ct * Σ * Ct' + obs_noise(pomdp, x))
    return (I - K * Ct) * Σ
    # Sigma - Sigma C' inv() C Sigma 
end

function update_belief(pomdp::iLQGPOMDP, belief::AbstractVector, u::AbstractVector)
    # extract mean and covariance from belief state
    n_states = num_states(pomdp)
    x = belief[1:n_states]
    Σ = reshape(belief[n_states+1:end], n_states, n_states)

    # mean update belief using most likely observation (no measurement gain)
    x_new = dyn_mean(pomdp, x, u)
    Σ_pred = sigma_pred(pomdp, x,u,Σ)
    Σ_new = sigma_update(pomdp, x_new,Σ_pred)

    return form_belief_vector(x_new,Σ_new)
end
function cost(Q, R, Q_N, s, u, s_goal)
    # Compute the cost of a state-action pair
    return (s - s_goal)' * Q * (s - s_goal) + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end

function mpc(pomdp::iLQGPOMDP, initial_belief::AbstractVector, horizon::Int)
    n_actions = num_actions(pomdp)
    initial_actions = [rand(n_actions) for _ in 1:horizon]
    flat_initial_actions = reduce(vcat, initial_actions)

    function cost_function(flat_actions)
        actions = [flat_actions[(i-1)*n_actions+1:i*n_actions] for i in 1:horizon]
        belief = initial_belief
        total_cost = 0.0

        for t in 1:horizon
            action = actions[t]
            state = dyn_mean(pomdp, belief[1:num_states(pomdp)], action) # + dyn_noise(pomdp, belief[1:num_states(mdp)], action)
            # noise_state = rand(MvNormal(pomdp.W_state_process))
            # noise_total = vcat(noise_state, 0.0)
            # state = state + noise_total
            # observation = obs_mean(pomdp, state) + obs_noise(pomdp, state)
            # belief = update_belief(pomdp, belief, action)

            total_cost += cost(pomdp.Q, pomdp.R, pomdp.Q_N, state, action, pomdp.s_goal[1:num_states(pomdp)])
        end

        return total_cost
    end

    result = optimize(cost_function, flat_initial_actions, method=BFGS())
    opt_action = result.minimizer[1:n_actions]

    return opt_action
end
