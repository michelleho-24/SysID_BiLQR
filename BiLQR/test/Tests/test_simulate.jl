using POMDPs
using Random
using LinearAlgebra
using ForwardDiff
using Distributions
using Plots

include("../POMDPs/cartpole_ilqrpomdp.jl")
include("../Policies/bilqr_policy.jl")

# Define the matrices and parameters
Q = diagm([1.0, 1.0, 1.0, 1.0, 1.0])
R = diagm([1.0])
Q_N = diagm([1.0, 1.0, 1.0, 1.0, 1.0])
Λ = diagm([1.0, 1.0, 1.0, 1.0, 1.0])

m0 = [0.0, π / 2, 0.0, 0.0, 2.0]         # Initial mean of the belief
Σ0 = diagm([1e-4, 1e-4, 1e-4, 1e-4, 2.0])  # Initial covariance matrix

δt = 0.1        # Time step
mc = 1.0        # Cart mass
g = 9.81        # Gravitational acceleration
l = 1.0         # Pole length

# Noise covariance matrices
W_state_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
W_process = diagm([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
W_obs = diagm([1e-2, 1e-2, 1e-2, 1e-2])
W_obs_ekf = diagm([1e-2, 1e-2, 1e-2, 1e-2])

# Create the CartpoleMDP instance
cartpole_mdp = CartpoleMDP(
    Q, R, Q_N, Λ, m0, Σ0, δt, mc, g, l,
    W_state_process, W_process, W_obs, W_obs_ekf
)

horizon = 100
N = 10
eps = 1e-6
max_iters = 100

policy = BiLQRPolicy(pomdp = cartpole_mdp, N = N, eps = eps, max_iters = max_iters)
simulate(pomdp::CartpoleMDP, policy.max_iters, policy)