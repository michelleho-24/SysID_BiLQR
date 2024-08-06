using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Infiltrator
include("cartpole.jl")
include("ilqr_types.jl")

function ekf(pomdp, b, a, z)

    # separate belief into mean and covariance 
    m, Σ = b[1:num_states(pomdp)], reshape(b[num_states(pomdp) + 1:end], num_states(pomdp), num_states(pomdp))

    # predict mean 
    m_pred = dyn_mean(pomdp, m, a)

    # calculate jacobians
    A = ForwardDiff.jacobian(s -> dyn_mean(pomdp, m, a), m)
    C = ForwardDiff.jacobian(s -> obs_mean(pomdp, s), m)

    # noise matrices 
    W_obs = pomdp.W_obs
    W_proc = pomdp.W_process
    
    # predict covariance
    Σ_pred = A * Σ * A' + W_proc

    # Update step 
    K = Σ_pred * C' * inv(C * Σ_pred * C' + W_obs)
    m_new = m_pred + K * (z - (C * m_pred)) 
    Σ_new = (I - K * C) * Σ_pred

    return vcat(m_new, Σ_new[:])

end 

