# ==============================================================================
# AirplanePOMDP Definition
# A POMDP model representing the dynamics and observations of an airplane system
# ==============================================================================
mutable struct AirplanePOMDP <: iLQRPOMDP{AbstractVector, AbstractVector, AbstractVector}

    # Cost matrices
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Q_N::Matrix{Float64}
    Λ::Matrix{Float64}

    # Initial belief and goal state
    Σ0::Vector{Float64}
    b0::MvNormal
    s_init::Vector{Float64}
    AB_true::Vector{Float64}
    s_goal::Vector{Float64}

    # Physical constants
    m::Float64
    g::Float64
    l::Float64
    δt::Float64

    # Noise covariance matrices
    W_state_process::Matrix{Float64}
    W_process::Matrix{Float64}
    W_obs::Matrix{Float64}
    W_obs_ekf::Matrix{Float64}

    # Dimensionality
    num_states::Int
    num_actions::Int
    num_observations::Int
    num_sysvars::Int

    # Constructor
    function AirplanePOMDP(
        Q::Matrix{Float64}, R::Matrix{Float64}, Q_N::Matrix{Float64}, Λ::Matrix{Float64},
        Σ0::Vector{Float64}, δt::Float64, m::Float64, g::Float64, l::Float64,
        W_state_process::Matrix{Float64}, W_process::Matrix{Float64},
        W_obs::Matrix{Float64}, W_obs_ekf::Matrix{Float64}, num_states::Int=12, num_actions::Int=2, 
        num_observations::Int=4, num_sysvars::Int=4
    )
        # Use centralized dimension methods for validation
        model = new(
            Q, R, Q_N, Λ, Σ0, MvNormal(rand(12), diagm(Σ0)), rand(12), rand(12), rand(12),
            m, g, l, δt, W_state_process, W_process, W_obs, W_obs_ekf
        )

        # # Validate dimensions using methods
        # @assert size(Q) == (num_states(model), num_states(model)) "Q matrix must match num_states"
        # @assert size(R) == (num_actions(model), num_actions(model)) "R matrix must match num_actions"
        # @assert size(Q_N) == (num_states(model), num_states(model)) "Q_N matrix must match num_states"
        # @assert size(Λ) == (num_states(model), num_states(model)) "Λ matrix must match num_states"
        # @assert length(Σ0) == num_states(model) "Σ0 vector must match num_states"
        # @assert size(W_state_process) == (num_states(model), num_states(model)) "W_state_process must match num_states"
        # @assert size(W_obs) == (num_observations(model), num_observations(model)) "W_obs must match num_observations"
        # @assert size(W_obs_ekf) == (num_observations(model), num_observations(model)) "W_obs_ekf must match num_observations"

        return model
    end
end

# ==============================================================================
# Dynamics and Observations
# Define system dynamics and observation functions for the AirplanePOMDP
# ==============================================================================

"""
    dyn_mean(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)

Compute the mean dynamics update for the airplane system.
"""
function dyn_mean(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)
    # Extract state and parameters
    s_true = s[1:4]
    A = s[5:8]
    B = s[9:end]

    # Define system matrices
    col2 = [0.0, -0.1, -0.5, 0.0]
    col3 = [-9.81, 1.0, -0.1, 1.0]
    col4 = [0.0, 0.0, 0.0, 0.0]
    colB = [0.0, 0.0, 0.0, 0.0]

    # Compute dynamics
    ds = hcat(A, col2, col3, col4) * s_true + hcat(B, colB) * a
    s_new = s_true + p.δt * ds

    # Return updated state
    return vcat(s_new, A, B)
end

"""
    dyn_noise(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)

Return the process noise covariance for the airplane system.
"""
dyn_noise(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process

"""
    obs_mean(p::AirplanePOMDP, sp::AbstractVector)

Return the mean observation for the airplane system.
"""
obs_mean(p::AirplanePOMDP, sp::AbstractVector) = sp[1:4]

"""
    obs_noise(p::AirplanePOMDP, sp::AbstractVector)

Return the observation noise covariance for the airplane system.
"""
obs_noise(p::AirplanePOMDP, sp::AbstractVector) = p.W_obs
