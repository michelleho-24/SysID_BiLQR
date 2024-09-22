using Parameters

# from Josh's paper: 
# state space consists of the velocity, angle of attack, angle of 
# sideslip, roll, pitch, yaw, roll rate, pitch rate, yaw rate, east 
# position, north position, altitude, and engine power setting
# The controls are the throttle percentage, elevator, aileron and 
# rudder deflections.

# from my talk with Josh: 
# state: position, altitude, angle of attack, control surface positions, velocity
# assume max throttle - (elevator, aileron, rudder)
# action: assume max throttle so elevator, aileron, rudder controls 


@with_kw mutable struct XPlanePOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    # TODO: change sizes 
    Q::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    R::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    Λ::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 121, 121)
    
    # start and end positions
    # A = I(11)
    # B = ones(11, 3)
    s_init::Vector{Float64} = [0.0, 0.0, 1000.0, 100.0, pi/2.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 
        vec(0.01*(Matrix{Float64}(I, 11, 11)))..., vec(0.01(Matrix{Float64}(ones(11, 3))))...]
    
    s_goal::Vector{Float64} = [s_init..., vec(Matrix{Float64}(I, 11, 11))..., vec(Matrix{Float64}(ones(11, 3)))...]
    
    # mechanics
    m::Float64 = 6500.0
    g::Float64 = 9.81
    l::Float64 = 1.0
    δt::Float64 = 0.1

    # noise
    # W_process::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11 + 11^2 + 33, 11 + 11^2 + 33)
    # W_obs::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    W_state_process::Matrix{Float64} = 1e-4 * Matrix{Float64}(I, 11, 11)
    W_process::Matrix{Float64} = Diagonal(vcat(fill(1e-4, 4), vec(0.0*(Matrix{Float64}(I, 11, 11)))..., vec(0.0(Matrix{Float64}(ones(11, 3)))...)))
    W_obs::Matrix{Float64} = 1e-2 * Matrix{Float64}(I, 8, 8)
end

function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    # Ax + Bu = x_new, A' = A, B' = B

    s_true, A_vec, B_vec = s[1:11], s[12:11^2+11], s[11^2+12:end]
    A = reshape(A_vec, 11, 11)
    B = reshape(B_vec, 11, 3)

    s_new = A * s_true + B * a
    return [s_new, A_vec, B_vec]
end

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp[1:8]
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 11 + 11^2 + 33
num_actions(p::CartpoleMDP) = 3
num_observations(p::CartpoleMDP) = 8
