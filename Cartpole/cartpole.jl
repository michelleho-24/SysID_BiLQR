using Parameters

@with_kw mutable struct CartpoleMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    # reward
    Q::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 4, 4)
    Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 4, 4)
    R::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 1, 1)
    Λ::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 16, 16)
    
    # start and end positions
    s_init::Vector{Float64} = [0.0, π/2, 0.0, 0.0]
    #TODO: change to be with 4 later
    s_goal::Vector{Float64} = [s_init..., vec(Matrix{Float64}(I, 4, 4))...]
    
    # mechanics
    δt::Float64 = 0.1
    mp::Float64 = 2.0
    mc::Float64 = 1.0
    g::Float64 = 9.81
    l::Float64 = 1.0

    # noise
    W_process::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 4, 4)
    W_obs::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 2, 2)
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

