using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
# using DifferentialEquations
# include("super_state.jl")
# include("ilqr_tests.jl")

form_belief_vector(x,Σ) = [x..., vec(Σ)...]

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

function update_state(pomdp::MPCPOMDP, belief::AbstractVector, u::AbstractVector)
    # extract mean and covariance from belief state
    n_states = num_states(pomdp)
    x = belief[1:n_states]
    Σ = reshape(belief[n_states+1:end], n_states, n_states)

    # mean update belief using most likely observation (no measurement gain)
    x_new = dyn_mean(pomdp, x, u)
    Σ_new = sigma_pred(pomdp, x, u, Σ)
    # create new belief state
    belief_new = form_belief_vector(x_new, Σ_new)
    return belief_new
end

function optimize_control(pomdp::MPCPOMDP, x::AbstractVector, horizon::Int)
    # This function should optimize control actions over a given horizon.
    # For simplicity, assume a linear or quadratic cost function and linear dynamics.
    
    u_seq = []
    
    # Dummy loop to represent optimization (replace with actual optimization code)
    for t in 1:horizon
        # Calculate control for the t-th step (this should be an optimization problem)
        u_t = -x  # Placeholder: In reality, this would be the result of solving an optimization problem.
        push!(u_seq, u_t)
    end
    
    return u_seq
end

function mpc_control(pomdp::MPCPOMDP, x::AbstractVector, horizon::Int)
    u_seq = optimize_control(pomdp, x, horizon)
    
    # Return the first action calculated
    return u_seq[1]
end
