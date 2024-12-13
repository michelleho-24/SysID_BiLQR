using Parameters
include("../BiLQR/ilqr_types.jl")

# TODO: look into how @with_kw works - do they all need to be defined as parameters? 

# can do without @with_kw and function inside cartpolemdp 
    # CartpoleMDP(Q, R, Qn, Lambda; kw1=1., kw2 = 2.,…) = new(Q, R, Qn, Lamda, kw1, kw2, …)
    # in the new you would need to pass in all the parameters
@with_kw mutable struct CartpoleMDP <: iLQGPOMDP{AbstractVector, AbstractVector, AbstractVector}
    
    #TODO: can i specify the size of these matrices here? 
    Q::Matrix{Float64} 
    R::Matrix{Float64}
    Q_N::Matrix{Float64} 
    Λ::Matrix{Float64} 
    
    Σ0::Matrix{Float64} # = diagm([1e-4, 1e-4, 1e-4, 1e-4, 2.0])
    b0::MvNormal = MvNormal([0.0, π/2, 0.0, 0.0, 2.0], Σ0)
    s_init::Vector{Float64} = begin
        s = rand(b0)
        s[end] = abs(s[end])  # Ensure the last element is positive
        s
    end

    mp_true::Float64 = s_init[end]
    s_goal::Vector{Float64} = [s_init..., vec(zeros(5, 5))...]
    
    # mechanics
    δt::Float64 = 0.1
    mc::Float64 = 1.0
    g::Float64 = 9.81
    l::Float64 = 1.0
    
    # noise covariance matrices
    W_state_process::Matrix{Float64} 
    W_process::Matrix{Float64} 
    W_obs::Matrix{Float64} 
    W_obs_ekf::Matrix{Float64} 
end

function dyn_mean(p::CartpoleMDP, s::AbstractVector, a::AbstractVector)
    x, θ, dx, dθ, mp = s
    sinθ, cosθ = sin(θ), cos(θ)
    h = p.mc + mp * (sinθ^2)
    ds = [
        dx,
        dθ,
        (mp * sinθ * (p.l * (dθ^2) + p.g * cosθ) + a[1]) / h,
        -((p.mc + mp) * p.g * sinθ + mp * p.l * (dθ^2) * sinθ * cosθ + a[1] * cosθ) / (h * p.l), 
        0.0
    ]
    
    s_new = s + p.δt * ds

    return s_new
end

dyn_noise(p::CartpoleMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::CartpoleMDP, sp::AbstractVector) = sp[1:4]
obs_noise(p::CartpoleMDP, sp::AbstractVector) = p.W_obs
num_states(p::CartpoleMDP) = 5
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 4
num_sysvars(p::CartpoleMDP) = 1