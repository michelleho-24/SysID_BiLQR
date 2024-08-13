using Parameters

# from Josh's paper: 
# state space consists of the velocity, angle of attack, angle of 
# sideslip, roll, pitch, yaw, roll rate, pitch rate, yaw rate, east 
# position, north position, altitude, and engine power setting
# The controls are the throttle percentage, elevator, aileron and 
# rudder deflections.

# from my talk with Josh: 
# state: position, velocity, altitude, angle of attack, control surface positions 
# assume max throttle - (elevator, aileron, rudder)
# action: assume max throttle so elevator, aileron, rudder controls 


@with_kw mutable struct F16POMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    # TODO: change sizes 
    Q::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    R::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    Λ::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 121, 121)
    
    # start and end positions
    # A = I(11)
    # B = zeros(11, 3)
    s_init::Vector{Float64} = [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 100.0, pi/2.0, 0.0, 0.0, 0.0]
    
    s_goal::Vector{Float64} = [s_init..., vec(Matrix{Float64}(I, 11, 11))..., vec(Matrix{Float64}(zeros(11, 3)))...]
    
    # mechanics
    m::Float64 = 6500.0
    g::Float64 = 9.81
    l::Float64 = 1.0
    δt::Float64 = 0.1

    # noise
    W_process::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11 + 11^2 + 33, 11 + 11^2 + 33)
    W_obs::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
end

function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    x, θ, dx, dθ = s
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + p.mp * (sinθ^2)
    ds = [
        dx,
        dθ,
        (p.mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,
        -((p.mc + p.mp) * p.g * sinθ + p.mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l)
    ]
    return s + p.δt * ds
end

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp[1:2]
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 4
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 2

"""Return a tuple of the Cartesian positions of the cart and the counterweight"""
function visualize(p::CartpoleMDP, s::AbstractVector) 
    return ([s[1], 0.], [s[1]+p.l*sin(s[2]), p.l*cos(s[2])])
end

