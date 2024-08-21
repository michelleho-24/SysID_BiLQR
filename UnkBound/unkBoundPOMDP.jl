using Parameters
using LinearAlgebra

@with_kw mutable struct UnkBoundPOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    # cost matrices
    Q::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    R::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 2, 2)
    Λ::Matrix{Float64} = Diagonal([ones(8); 5.0])
    
    # start and end positions
    s_init::Vector{Float64} = [0.0, 0.0, 5.0]
    s_goal::Vector{Float64} = vcat([10.0, 0.0, 5.0], vec(Matrix{Float64}(I, 3, 3)))
    
    # mechanics
    scale::Float64 = 10.0
    shift::Float64 = 0.5

    # noise
    W_process::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    W_obs::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 2, 2)
end

function dyn_mean(p::UnkBoundPOMDP, s::AbstractVector, a::AbstractVector)
    x, y, h = s
    ax, ay = a

    # soft barrier function
    sigmoid(x) = 1 / (1 + exp(-x))

    # Calculate α using a soft barrier function based on the sigmoid
    # Adjust the scale and shift to control the smoothness and threshold
    scale = p.scale  # Controls the steepness of the transition
    shift = h - p.shift  # Adjusts the midpoint of the transition, set to h for this example
    α = 1 + sigmoid(scale * (y - shift))

    # x_next = max(x + ax * α,0.0)
    # y_next = max(y + ay,0.0)
    x_next = x + ax * α
    y_next = y + ay

    h_next = h + p.W_process[3,3]

    return [x_next, y_next, h_next]
    
end

dyn_noise(p::UnkBoundPOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::UnkBoundPOMDP, sp::AbstractVector) = sp[1:2]
obs_noise(p::UnkBoundPOMDP, sp::AbstractVector) = p.W_obs
num_states(p::UnkBoundPOMDP) = 3
num_actions(p::UnkBoundPOMDP) = 2
num_observations(p::UnkBoundPOMDP) = 2

"""Return a tuple of the Cartesian positions of the cart and the counterweight"""
function visualize(p::UnkBoundPOMDP, s::AbstractVector) 
    return ([s[1], 0.], [s[1]+p.l*sin(s[2]), p.l*cos(s[2])])
end

