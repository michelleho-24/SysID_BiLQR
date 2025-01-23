
mutable struct AirplanePOMDP <: iLQRPOMDP{AbstractVector, AbstractVector, AbstractVector}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Q_N::Matrix{Float64}
    Λ::Matrix{Float64}
    Σ0::Vector{Float64}
    b0::MvNormal
    s_init::Vector{Float64}
    AB_true::Vector{Float64}
    s_goal::Vector{Float64}
    m::Float64
    g::Float64
    l::Float64
    δt::Float64
    W_state_process::Matrix{Float64}
    W_process::Matrix{Float64}
    W_obs::Matrix{Float64}
    W_obs_ekf::Matrix{Float64}

    function AirplanePOMDP(
        Q::Matrix{Float64}, R::Matrix{Float64}, Q_N::Matrix{Float64}, Λ::Matrix{Float64},
        Σ0::Vector{Float64}, δt::Float64, m::Float64, g::Float64, l::Float64,
        W_state_process::Matrix{Float64}, W_process::Matrix{Float64},
        W_obs::Matrix{Float64}, W_obs_ekf::Matrix{Float64}
    )
        b0 = MvNormal(
            vcat([1.0, 0.5, 0.1, 0.05], [-0.05, 0, 0, 0], [0, 1, 0, 0]),
            diagm(Σ0)
        )
        s_init = rand(b0)
        AB_true = s_init[5:end]
        s_goal = vcat([100, 0, 0, 0], AB_true, vec(zeros(12))...)

        new(
            Q, R, Q_N, Λ, Σ0, b0, s_init, AB_true, s_goal,
            m, g, l, δt, W_state_process, W_process, W_obs, W_obs_ekf
        )
    end
end

num_states(p::AirplanePOMDP) = 12
num_actions(p::AirplanePOMDP) = 2
num_observations(p::AirplanePOMDP) = 4
num_sysvars(p::AirplanePOMDP) = 4

function dyn_mean(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector)
    s_true = s[1:4]
    A = s[5:8]
    B = s[9:end]

    col2 = [0.0, -0.1, -0.5, 0.0]
    col3 = [-9.81, 1.0, -0.1, 1.0]
    col4 = [0.0, 0.0, 0.0, 0.0]
    colB = [0.0, 0.0, 0.0, 0.0]

    ds = hcat(A, col2, col3, col4) * s_true + hcat(B, colB) * a
    s_new = s_true + p.δt * ds

    return vcat(s_new, A, B)
end

dyn_noise(p::AirplanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::AirplanePOMDP, sp::AbstractVector) = sp[1:4]
obs_noise(p::AirplanePOMDP, sp::AbstractVector) = p.W_obs
