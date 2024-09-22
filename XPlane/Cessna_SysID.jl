using Parameters
using LinearAlgebra
using SparseArrays
include("../BiLQR/ilqr_types.jl")

# from Josh's paper: 
# state space consists of the velocity, angle of attack, angle of 
# sideslip, roll, pitch, yaw, roll rate, pitch rate, yaw rate, east 
# position, north position, altitude, and engine power setting
# The controls are the throttle percentage, elevator, aileron and 
# rudder deflections.

# from my talk with Josh: 
# state: position, altitude, angle of attack, control surface positions , velocity
# assume max throttle - (elevator, aileron, rudder)
# action: assume max throttle so elevator, aileron, rudder controls 

@with_kw mutable struct XPlanePOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    Q::Matrix{Float64} = 1e-6 * Matrix{Float64}(I, 165, 165)
    R::Matrix{Float64} = 1e-6 * Matrix{Float64}(I, 3, 3)
    # Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    # Λ::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 121, 121)
    Q_N::Matrix{Float64} = Diagonal(vcat(fill(1e-10, 11), fill(0.1, 154)))
    # Λ::Matrix{Float64} = Diagonal(vec([i ≥ 12 && j ≥ 12 ? 0.1 : 1e-10 for i in 1:165, j in 1:165])) # total 165^2
    Λ::Matrix{Float64} = spdiagm(0 => vec([i ≥ 12 && j ≥ 12 ? 1.0 : 1e-10 for i in 1:165, j in 1:165]))
    # start and end positions
    s_init::Vector{Float64} = vcat([0.0, 0.0, 1000.0, 100.0, pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                   vec(0.01 * Matrix{Float64}(I, 11, 11)), 
                                   vec(0.01 * Matrix{Float64}(ones(11, 3))))
    
    s_goal::Vector{Float64} = vcat(s_init, 
                                   vec(Matrix{Float64}(I, 11, 11)), 
                                   vec(Matrix{Float64}(ones(11, 3))))
    
    # mechanics
    m::Float64 = 6500.0
    g::Float64 = 9.81
    l::Float64 = 1.0
    δt::Float64 = 0.1

    # noise
    W_state_process::Matrix{Float64} = 1e-4 * Matrix{Float64}(I, 11, 11)
    W_process::Matrix{Float64} = Diagonal(vcat(fill(1e-4, 11), 
                                               vec(0.0 * Matrix{Float64}(I, 11, 11)), 
                                               vec(0.0 * Matrix{Float64}(ones(11, 3)))))
    W_obs::Matrix{Float64} = 1e-2 * Matrix{Float64}(I, 11, 11)
end

function dyn_mean(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    # Ax + Bu = x_new, A' = A, B' = B

    s_true, A_vec, B_vec = s[1:11], s[12:11^2+11], s[11^2+12:end]
    A = reshape(A_vec, 11, 11)
    B = reshape(B_vec, 11, 3)

    s_new = A * s_true + B * a
    return vcat(s_new, A_vec, B_vec)
end

dyn_noise(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::XPlanePOMDP, sp::AbstractVector) = sp[1:11]
obs_noise(p::XPlanePOMDP, sp::AbstractVector) = p.W_obs
num_states(p::XPlanePOMDP) = 11 + 11^2 + 33
num_actions(p::XPlanePOMDP) = 3
num_observations(p::XPlanePOMDP) = 11

mdp = XPlanePOMDP()