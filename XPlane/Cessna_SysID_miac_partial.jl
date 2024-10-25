using Parameters
using LinearAlgebra
using SparseArrays
using Distributions
include("../BiLQR/ilqr_types.jl")

# from Josh's paper: 
# state space consists of the velocity, angle of attack, angle of 
# sideslip, roll, pitch, yaw, roll rate, pitch rate, yaw rate, east 
# position, north position, altitude, and engine power setting
# The controls are the throttle percentage, elevator, aileron and 
# rudder deflections.

# from my talk with Josh: 
# state: position, altitude, angle of attack, control surface positions , velocity
# x, z, theta, delta_e, delta_a, delta_r, x_dot, z_dot
# assume max throttle - (elevator, aileron, rudder)
# action: assume max throttle so elevator, aileron, rudder controls 

@with_kw mutable struct XPlanePOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    # weight x z, theta positions high
    Q::Matrix{Float16} = diagm(vcat(fill(0.1, 3), fill(1e-5, 5), fill(1e-5, 16)))
    R::Matrix{Float16} = diagm(0 => fill(0.05, 3))
    Q_N::Matrix{Float16} = diagm(vcat(fill(0.1, 3), fill(1e-5, 5), fill(0.1, 16)))
    Λ::Matrix{Float16} = Diagonal(vcat(fill(0.05, 8), fill(1, 16))) 
    
    # Σ0::Matrix{Float64} = Diagonal(vcat(fill(1e-10, 8), fill(2, 16)))
    Σ0::Vector{Float64} = vcat(fill(1e-5, 8), fill(2, 16))
    b0::MvNormal = MvNormal(
        vcat([0.0, 1000.0, pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
             [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], # A diagonal means 
             [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]), # B first column means 
        Σ0)
    s_init::Vector{Float64} = rand(b0)
    # A_true::Matrix{Float64} = Diagonal(s_init[8:15])
    # B_true::Matrix{Float64} = hcat(s_init[16:end], ones(8, 2))
    AB_true::Vector{Float64} = vcat(s_init[8:end])

    # move to new x position, keep same height, and same angle of attack 
    s_goal::Vector{Float64} = vcat([5000], s_init[2:end] ..., vec(zeros(24))...)
    
    # mechanics
    m::Float16 = 6500.0
    g::Float16 = 9.81
    l::Float16 = 1.0
    δt::Float16 = 0.1

    # noise
    W_state_process::Matrix{Float16} = 1e-3 * Matrix{Float16}(I, 8, 8)
    W_process::Matrix{Float16} = Diagonal(vcat(fill(1e-3, 8), 
                                            fill(0, 8), 
                                            fill(0, 8)) )
    W_obs::Matrix{Float16} = 1e-1 * Matrix{Float16}(I, 6, 6)
end

function dyn_mean(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    # Ax + Bu = x_new, A' = A, B' = B

    s_true = s[1:8]
    A = s[9:16]
    B = s[17:end]

    s_new = Diagonal(A) * s_true + hcat(B, ones(8,2)) * a
    return vcat(s_new, A, B)
end

dyn_noise(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::XPlanePOMDP, sp::AbstractVector) = sp[1:8]
obs_noise(p::XPlanePOMDP, sp::AbstractVector) = p.W_obs
num_states(p::XPlanePOMDP) = 8 + 8 + 8 
num_actions(p::XPlanePOMDP) = 3
num_observations(p::XPlanePOMDP) = 6
num_sysvars(p::XPlanePOMDP) = 8 + 8
