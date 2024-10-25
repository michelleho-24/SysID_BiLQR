
using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Infiltrator
include("ilqr_types.jl")

function ekf(pomdp, b, a, z)
    # Separate belief into mean and covariance
    num_state = num_states(pomdp)
    m = b[1:num_state]
    Σ_vec = b[num_state+1:end]

    # Predict mean
    m_pred = dyn_mean(pomdp, m, a)

    # Calculate Jacobians
    # Correctly compute the Jacobian of dyn_mean with respect to s
    A = ForwardDiff.jacobian(s -> dyn_mean(pomdp, s, a), m)

    # Compute the Jacobian of obs_mean with respect to s, evaluated at m_pred
    C = ForwardDiff.jacobian(s -> obs_mean(pomdp, s), m_pred)

    # Noise matrices
    W_obs = pomdp.W_obs
    W_proc = pomdp.W_process

    # Predict covariance
    Σ_pred = A * Diagonal(Σ_vec) * A' + W_proc

    # Update step
    # Compute the innovation (difference between actual observation and predicted observation)
    y = z - obs_mean(pomdp, m_pred)

    # Compute the innovation covariance
    S = C * Σ_pred * C' + W_obs

    if any(isnan, S) || abs(det(S)) < 1e-12
        println("S is nan, next seed...")
        return nothing
    end

    # Compute the Kalman Gain
    K = Σ_pred * C' * inv(S)

    # Update the mean estimate
    m_new = m_pred + K * y

    # Update the covariance estimate
    Σ_new = (I - K * C) * Σ_pred

    return vcat(m_new, diag(Σ_new))
end
