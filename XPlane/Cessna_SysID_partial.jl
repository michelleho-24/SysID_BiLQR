using Parameters
using LinearAlgebra
using SparseArrays
include("../BiLQR/ilqr_types.jl")

@with_kw mutable struct XPlanePOMDP <: iLQGPOMDP{AbstractVector,AbstractVector,AbstractVector}
    
    Q::Matrix{Float16} = 1e-6 * Matrix{Float16}(I, 96, 96)
    R::Matrix{Float16} = 1e-6 * Matrix{Float16}(I, 3, 3)
    # Q_N::Matrix{Float16} = 1e-3 * Matrix{Float16}(I, 11, 11)
    # Λ::Matrix{Float16} = 1e-3 * Matrix{Float16}(I, 121, 121)
    Q_N::Matrix{Float16} = Diagonal(vcat(fill(1e-10, 8), fill(0.1, 88)))
    # Λ::Matrix{Float16} = Diagonal(vec([i ≥ 12 && j ≥ 12 ? 0.1 : 1e-10 for i in 1:165, j in 1:165])) # total 165^2
    Λ::Matrix{Float16} = spdiagm(0 => vec([i ≥ 9 && j ≥ 9 ? 1.0 : 1e-10 for i in 1:96, j in 1:96])) # total 96^2
    # start and end positions
    s_init::Vector{Float16} = vcat([0.0, 1000.0, pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                   vec(0.01 * Matrix{Float16}(I, 8, 8)), 
                                   vec(0.01 * Matrix{Float16}(ones(8, 3))))
    
    s_goal::Vector{Float16} = vcat(s_init ..., vec(Matrix{Float16}(I, 96, 96))...)
    
    # mechanics
    m::Float16 = 6500.0
    g::Float16 = 9.81
    l::Float16 = 1.0
    δt::Float16 = 0.1

    # noise
    W_state_process::Matrix{Float16} = 1e-4 * Matrix{Float16}(I, 8, 8)
    W_process::Matrix{Float16} = Diagonal(vcat(fill(1e-4, 8), 
                                               vec(0.0 * Matrix{Float16}(I, 8, 8)), 
                                               vec(0.0 * Matrix{Float16}(ones(8, 3)))))
    W_obs::Matrix{Float16} = 1e-2 * Matrix{Float16}(I, 6, 6)
end

function dyn_mean(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    # Ax + Bu = x_new, A' = A, B' = B

    s_true, A_vec, B_vec = s[1:8], s[9:8^2+8], s[8^2+8+1:end]
    A = reshape(A_vec, 8, 8)
    B = reshape(B_vec, 8, 3)

    s_new = A * s_true + B * a
    return vcat(s_new, A_vec, B_vec)
end

dyn_noise(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector) = p.W_process
obs_mean(p::XPlanePOMDP, sp::AbstractVector) = sp[1:6]
obs_noise(p::XPlanePOMDP, sp::AbstractVector) = p.W_obs
num_states(p::XPlanePOMDP) = 8 + 8^2 + 8*3
num_actions(p::XPlanePOMDP) = 3
num_observations(p::XPlanePOMDP) = 6

mdp = XPlanePOMDP()