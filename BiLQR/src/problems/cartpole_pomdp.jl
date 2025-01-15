
#     Σ0::Matrix{Float64} # = diagm([1e-4, 1e-4, 1e-4, 1e-4, 2.0])
#     b0::MvNormal = MvNormal([0.0, π/2, 0.0, 0.0, 2.0], Σ0)
#     δt::Float64 = 0.1
#     mc::Float64 = 1.0
#     g::Float64 = 9.81
#     l::Float64 = 1.0

mutable struct CartpoleMDP <: iLQRPOMDP{AbstractVector, AbstractVector, AbstractVector}
    #TODO: add checks for matrix sizes 
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Q_N::Matrix{Float64}
    Λ::Matrix{Float64}
    m0::Vector{Float64}
    Σ0::Matrix{Float64}
    b0::MvNormal
    s_init::Vector{Float64}
    mp_true::Float64
    s_goal::Vector{Float64}
    δt::Float64
    mc::Float64
    g::Float64
    l::Float64
    W_state_process::Matrix{Float64}
    W_process::Matrix{Float64}
    W_obs::Matrix{Float64}
    W_obs_ekf::Matrix{Float64}

    function CartpoleMDP(
        Q::Matrix{Float64}, R::Matrix{Float64}, Q_N::Matrix{Float64}, Λ::Matrix{Float64},
        m0::Vector{Float64}, Σ0::Matrix{Float64}, δt::Float64, mc::Float64, g::Float64, l::Float64,
        W_state_process::Matrix{Float64}, W_process::Matrix{Float64},
        W_obs::Matrix{Float64}, W_obs_ekf::Matrix{Float64}
    )
        b0 = MvNormal(m0, Σ0)
        s_init = rand(b0)
        s_init[end] = abs(s_init[end])
        mp_true = s_init[end]
        s_goal = [s_init...; vec(zeros(5))...]
        new(Q, R, Q_N, Λ, m0, Σ0, b0, s_init, mp_true, s_goal, δt, mc, g, l, W_state_process, W_process, W_obs, W_obs_ekf)
    end
end

num_states(p::CartpoleMDP) = 5
num_actions(p::CartpoleMDP) = 1
num_observations(p::CartpoleMDP) = 4
num_sysvars(p::CartpoleMDP) = 1

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
obs_mean(p::CartpoleMDP, sp::AbstractVector, a::AbstractVector) = sp[1:4]
obs_noise(p::CartpoleMDP, sp::AbstractVector, a::AbstractVector) = p.W_obs
